# /// script
# requires-python = ">=3.12"
# dependencies = ["lerobot[dataset]==0.6.1", "av", "numpy", "pyyaml", "tqdm", "wandb>=0.24"]
# ///
"""
Collate episodes saved by RoboRollout into a LeRobot (v3.0) dataset.

Episodes are read from local Hydra output directories (searched recursively for episode directories),
or from W&B projects given as wandb:<entity>/<project>, in which case each run is one episode.
Episodes can be filtered by date and by any metadata (info.json values, config values, or W&B run fields).

Each episode becomes one LeRobot episode, with:

    observation.images.<camera>   the observation-synced video of each camera, at full resolution
    observation.state             the --state-keys of observations.npz, concatenated
    action                        the --action-keys of actions.npz, concatenated
    task                          the episode's task

The source and metadata of each episode are saved to meta/robo_rollout_sources.json.

Examples:

    # list the successful episodes from the last week that would be collated
    uv run scripts/collate_lerobot.py outputs/ wandb:my-entity/droid-evals \\
        --repo-id me/droid-pick --since 2026-10-07 --filter success=true --dry-run

    # collate them and upload to the Hugging Face Hub
    uv run scripts/collate_lerobot.py outputs/ wandb:my-entity/droid-evals \\
        --repo-id me/droid-pick --since 2026-10-07 --filter success=true --upload
"""

import argparse
import json
import operator
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import av
import numpy as np
import yaml
from tqdm import tqdm

SOURCES_FILE = "meta/robo_rollout_sources.json"
WANDB_PREFIX = "wandb:"
DEFAULT_CACHE_DIR = Path("~/.cache/robo_rollout/wandb").expanduser()
# per-step policy server timing, which isn't part of the action
DEFAULT_EXCLUDED_ACTION_PREFIXES = ("server_timing/",)


@dataclass
class Episode:
    source: str  # local path or W&B run path, for display
    date: datetime  # local time, naive
    metadata: dict  # info.json values, plus "config" and (for W&B) "wandb"
    # returns the local paths of observations.npz, actions.npz, and each camera's synced video
    fetch: Callable[[list[str]], tuple[Path, Path, dict[str, Path]]] = field(repr=False)

    @property
    def cameras(self) -> list[str]:
        return list(self.metadata["config"].get("robot", {}).get("cameras", {}))

    @property
    def fps(self) -> int:
        # synced videos are written at round(1 / policy_dt) fps, see robo_rollout.rollout.create_writer
        return round(1.0 / self.metadata["config"]["policy_dt"])


# ── Sources ───────────────────────────────────────────────────────────


def find_disk_episodes(root: Path) -> Iterator[Episode]:
    """Find episodes under root, which may be an episode directory, a run directory, or any parent of them."""
    info_paths = [root / "info.json"] if (root / "info.json").is_file() else sorted(root.rglob("info.json"))
    for info_path in info_paths:
        episode_dir = info_path.parent
        if not (episode_dir / "observations.npz").is_file():
            continue
        config_path = episode_dir.parent / ".hydra" / "config.yaml"
        if not config_path.is_file():
            print(f"WARN: Skipping {episode_dir}: no config at {config_path}")
            continue
        metadata = json.loads(info_path.read_text())
        metadata["config"] = yaml.safe_load(config_path.read_text())

        def fetch(cameras: list[str], episode_dir=episode_dir):
            videos = {cam: episode_dir / f"{cam}.mp4" for cam in cameras}
            return episode_dir / "observations.npz", episode_dir / "actions.npz", videos

        yield Episode(
            source=str(episode_dir),
            # info.json is written when the episode ends
            date=datetime.fromtimestamp(info_path.stat().st_mtime),
            metadata=metadata,
            fetch=fetch,
        )


def find_wandb_episodes(project_path: str, since: datetime | None, cache_dir: Path) -> Iterator[Episode]:
    """Find episodes logged as runs of a W&B project, given as <entity>/<project>."""
    import wandb

    api = wandb.Api()
    filters = {}
    if since is not None:
        # only an optimization, since dates are filtered again locally
        filters["createdAt"] = {"$gte": since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}
    for run in api.runs(project_path, filters=filters, lazy=False):
        summary = run.summary_metrics
        if "task" not in summary:
            print(f"WARN: Skipping W&B run {'/'.join(run.path)}: no episode info logged (did it crash?)")
            continue
        metadata = {k: v for k, v in summary.items() if not k.startswith(("_", "video/"))}
        metadata["config"] = dict(run.config)
        metadata["wandb"] = {"id": run.id, "name": run.name, "tags": list(run.tags), "state": run.state}

        def fetch(cameras: list[str], run=run, summary=summary):
            run_dir = cache_dir / Path(*run.path)
            names = {"observations": "observations.npz", "actions": "actions.npz"}
            for cam in cameras:
                if f"video/{cam}" not in summary:
                    raise FileNotFoundError(f"no video logged for camera {cam}")
                names[cam] = summary[f"video/{cam}"]["path"]
            paths = {}
            for key, name in names.items():
                file = run.file(name)
                path = run_dir / name
                # re-download partial downloads
                replace = path.exists() and path.stat().st_size != file.size
                file.download(root=str(run_dir), replace=replace, exist_ok=True)
                paths[key] = path
            return paths.pop("observations"), paths.pop("actions"), paths

        yield Episode(
            source=f"{WANDB_PREFIX}{'/'.join(run.path)}",
            date=parse_utc(run.created_at),
            metadata=metadata,
            fetch=fetch,
        )


def parse_utc(timestamp: str) -> datetime:
    """Parse a W&B UTC timestamp into a naive local datetime."""
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().replace(tzinfo=None)


# ── Filtering ─────────────────────────────────────────────────────────


def lookup(metadata: dict, key: str) -> Any:
    """Get a dotted key from nested metadata, or KeyError. Keys containing dots are matched whole first."""
    if key in metadata:
        return metadata[key]
    for i, char in enumerate(key):
        if char == "." and key[:i] in metadata and isinstance(metadata[key[:i]], dict):
            try:
                return lookup(metadata[key[:i]], key[i + 1:])
            except KeyError:
                pass
    raise KeyError(key)


def regex_match(value: Any, pattern: Any) -> bool:
    return re.search(str(pattern), str(value)) is not None


# longer operators first, so that e.g. ">=" isn't parsed as ">"
FILTER_OPS = {
    "!=": operator.ne,
    "~=": regex_match,
    ">=": operator.ge,
    "<=": operator.le,
    "=": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}
FILTER_RE = re.compile(r"^(.+?)(" + "|".join(re.escape(op) for op in FILTER_OPS) + r")(.*)$")


@dataclass
class MetadataFilter:
    key: str
    op: str
    value: Any

    @classmethod
    def parse(cls, spec: str) -> "MetadataFilter":
        match = FILTER_RE.match(spec)
        if match is None:
            raise argparse.ArgumentTypeError(f"invalid filter {spec!r}, expected KEY<op>VALUE")
        key, op, value = match.groups()
        if op != "~=":
            # parse values like YAML, so that e.g. "true" and "1" compare equal to booleans and numbers
            value = yaml.safe_load(value)
            # but keep dates as strings, to compare with the ISO-formatted "date" key
            if isinstance(value, date):
                value = value.isoformat()
        return cls(key.strip(), op, value)

    def __call__(self, metadata: dict) -> bool:
        try:
            actual = lookup(metadata, self.key)
        except KeyError:
            return self.op == "!="
        compare = FILTER_OPS[self.op]
        if isinstance(actual, list):
            # lists (e.g. tags) match if any element does, or for "!=", if no element does
            if self.op == "!=":
                return all(compare(x, self.value) for x in actual)
            return any(self._safe_compare(compare, x) for x in actual)
        return self._safe_compare(compare, actual)

    def _safe_compare(self, compare, actual) -> bool:
        try:
            return bool(compare(actual, self.value))
        except TypeError:  # e.g. comparing a string to a number
            return False


def parse_date(value: str, end: bool = False) -> datetime:
    """Parse a date or datetime. A bare date as an end bound includes the whole day."""
    dt = datetime.fromisoformat(value)
    if end and len(value) <= len("YYYY-MM-DD"):
        dt += timedelta(days=1)
    return dt


# ── Conversion ────────────────────────────────────────────────────────


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def select_keys(data: dict[str, np.ndarray], keys: list[str] | None, kind: str, exclude=()) -> list[str]:
    """Pick the keys to concatenate. Defaults to every numeric key with at most one dimension per step."""
    if keys is not None:
        missing = [k for k in keys if k not in data]
        if missing:
            raise KeyError(f"{kind} keys {missing} not found, available keys: {list(data)}")
        return keys
    selected = []
    for key, value in data.items():
        if key.startswith(exclude):
            continue
        if not (np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_) or value.ndim > 2:
            print(f"WARN: Not including {kind} key {key} (dtype {value.dtype}, shape {value.shape})")
            continue
        selected.append(key)
    return selected


def concat_keys(data: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Concatenate per-step arrays into a (n_steps, dim) float32 array. Scalars per step become one dimension."""
    n = min(len(data[k]) for k in keys)
    return np.concatenate([data[k][:n].reshape(n, -1) for k in keys], axis=1).astype(np.float32)


def dim_names(data: dict[str, np.ndarray], keys: list[str]) -> list[str]:
    names = []
    for key in keys:
        dim = int(np.prod(data[key].shape[1:]))
        names += [key] if dim == 1 else [f"{key}.{i}" for i in range(dim)]
    return names


def video_info(path: Path) -> tuple[int, int, int]:
    """Get the (n_frames, height, width) of a video."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        n_frames = stream.frames or sum(1 for _ in container.decode(stream))
        return n_frames, stream.height, stream.width


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


@dataclass
class Layout:
    """Which keys and cameras make up each frame, fixed by the first collated episode."""
    cameras: list[str]
    state_keys: list[str]
    action_keys: list[str]
    state_names: list[str]
    action_names: list[str]
    image_shapes: dict[str, tuple[int, int, int]]

    def features(self) -> dict:
        features = {
            f"observation.images.{cam}": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
            for cam, shape in self.image_shapes.items()
        }
        features["observation.state"] = {
            "dtype": "float32", "shape": (len(self.state_names),), "names": self.state_names,
        }
        features["action"] = {
            "dtype": "float32", "shape": (len(self.action_names),), "names": self.action_names,
        }
        return features


def prepare_episode(episode: Episode, layout: Layout | None, args) -> tuple[Layout, dict]:
    """
    Fetch and load an episode, returning its layout and data. Raises ValueError if it can't be collated.
    Observations, actions, and videos are trimmed to the same length, since an interrupt can leave
    them one step apart (e.g. an observation was recorded, but the episode ended before its action was).
    """
    cameras = layout.cameras if layout is not None else args.cameras
    obs_path, act_path, video_paths = episode.fetch(cameras)
    obs, actions = load_npz(obs_path), load_npz(act_path)
    if not obs or not actions:
        raise ValueError("no steps recorded")

    if layout is None:
        state_keys = select_keys(obs, args.state_keys, "observation")
        action_keys = select_keys(actions, args.action_keys, "action", DEFAULT_EXCLUDED_ACTION_PREFIXES)
        if not state_keys or not action_keys:
            raise ValueError(f"no state keys ({state_keys}) or no action keys ({action_keys})")
        image_shapes = {}
        for cam, path in video_paths.items():
            _, height, width = video_info(path)
            image_shapes[cam] = (height, width, 3)
        layout = Layout(
            cameras=cameras,
            state_keys=state_keys,
            action_keys=action_keys,
            state_names=dim_names(obs, state_keys),
            action_names=dim_names(actions, action_keys),
            image_shapes=image_shapes,
        )

    for kind, data, keys, names in (
        ("observation", obs, layout.state_keys, layout.state_names),
        ("action", actions, layout.action_keys, layout.action_names),
    ):
        if missing := [k for k in keys if k not in data]:
            raise ValueError(f"missing {kind} keys {missing}")
        if dim_names(data, keys) != names:
            raise ValueError(f"{kind} dimensions {dim_names(data, keys)} don't match {names}")

    state = concat_keys(obs, layout.state_keys)
    action = concat_keys(actions, layout.action_keys)
    lengths = {"observations": len(state), "actions": len(action)}
    for cam, path in video_paths.items():
        n_frames, height, width = video_info(path)
        if (height, width, 3) != layout.image_shapes[cam]:
            raise ValueError(f"{cam} video is {width}x{height}, expected {layout.image_shapes[cam][1::-1]}")
        lengths[f"video/{cam}"] = n_frames

    n = min(lengths.values())
    if n == 0:
        raise ValueError(f"no steps recorded ({lengths})")
    if max(lengths.values()) - n > 1:
        print(f"WARN: Trimming {episode.source} by more than one step: {lengths}")
    return layout, {
        "n_frames": n,
        "lengths": lengths,
        "state": state[:n],
        "action": action[:n],
        "videos": video_paths,
    }


def write_episode(dataset, episode: Episode, data: dict):
    task = episode.metadata["task"]
    readers = [iter_frames(path) for path in data["videos"].values()]
    keys = [f"observation.images.{cam}" for cam in data["videos"]]
    try:
        for t, images in zip(range(data["n_frames"]), zip(*readers)):
            frame = dict(zip(keys, images))
            frame["observation.state"] = data["state"][t]
            frame["action"] = data["action"][t]
            frame["task"] = task
            dataset.add_frame(frame)
        dataset.save_episode()
    except BaseException:
        dataset.clear_episode_buffer()
        raise
    finally:
        for reader in readers:
            reader.close()


# ── Main ──────────────────────────────────────────────────────────────


def print_episodes(episodes: list[Episode]):
    print(f"{'#':>4}  {'date':<19}  {'success':<7}  {'steps':>5}  {'task':<40}  source")
    for i, ep in enumerate(episodes):
        success = ep.metadata.get("success", "?")
        steps = ep.metadata.get("episode_length_steps", "?")
        task = str(ep.metadata.get("task", "?"))
        task = task if len(task) <= 40 else task[:37] + "..."
        print(f"{i:>4}  {ep.date:%Y-%m-%d %H:%M:%S}  {str(success):<7}  {steps:>5}  {task:<40}  {ep.source}")


def json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sources", nargs="+",
        help="Directories to search for saved episodes, or W&B projects as wandb:<entity>/<project>",
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo ID, e.g. <hf_user>/<dataset_name>")
    parser.add_argument("--root", type=Path, help="Where to write the dataset (default: $HF_LEROBOT_HOME/<repo-id>)")
    parser.add_argument("--overwrite", action="store_true", help="Delete the dataset at --root if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Only list the episodes that would be collated")
    parser.add_argument("--upload", action="store_true", help="Upload the dataset to the Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="Make the uploaded dataset private")

    filtering = parser.add_argument_group("filtering")
    filtering.add_argument(
        "--filter", dest="filters", type=MetadataFilter.parse, action="append", default=[], metavar="KEY<op>VALUE",
        help="Only include episodes whose metadata matches, where <op> is one of = != ~= (regex) < <= > >=. "
             "Repeat to require several filters. e.g. success=true, config.policy_metadata.model~=pi0",
    )
    filtering.add_argument("--since", type=parse_date, help="Only include episodes from this date or datetime on")
    filtering.add_argument(
        "--until", type=lambda s: parse_date(s, end=True),
        help="Only include episodes up to this date (inclusive) or datetime",
    )

    contents = parser.add_argument_group("dataset contents")
    contents.add_argument("--cameras", nargs="+", help="Cameras to include (default: those in every episode)")
    contents.add_argument(
        "--state-keys", nargs="+",
        help="observations.npz keys to concatenate into observation.state, e.g. qpos/arm qpos/gripper "
             "(default: every numeric key)",
    )
    contents.add_argument(
        "--action-keys", nargs="+",
        help="actions.npz keys to concatenate into action, e.g. arm gripper "
             "(default: every numeric key, except server_timing/*)",
    )
    contents.add_argument("--robot-type", help="Robot type to save in the dataset metadata, e.g. franka")
    contents.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Where to download W&B episode files",
    )
    args = parser.parse_args()

    episodes = []
    for source in args.sources:
        if source.startswith(WANDB_PREFIX):
            episodes += find_wandb_episodes(source[len(WANDB_PREFIX):], args.since, args.cache_dir)
        elif Path(source).is_dir():
            episodes += find_disk_episodes(Path(source))
        else:
            parser.error(f"{source} is neither a directory nor a W&B project ({WANDB_PREFIX}<entity>/<project>)")

    episodes = [
        ep for ep in episodes
        if (args.since is None or ep.date >= args.since)
        and (args.until is None or ep.date < args.until)
        and all(f({**ep.metadata, "date": ep.date.isoformat(), "source": ep.source}) for f in args.filters)
    ]
    episodes.sort(key=lambda ep: ep.date)
    if not episodes:
        parser.error("no episodes matched")
    print_episodes(episodes)
    n_success = sum(ep.metadata.get("success") is True for ep in episodes)
    print(f"\n{len(episodes)} episodes ({n_success} successful)")

    fps_values = {ep.fps for ep in episodes}
    if len(fps_values) > 1:
        parser.error(f"episodes were recorded at different rates ({sorted(fps_values)} fps), "
                     f"filter on config.policy_dt to collate one rate")
    fps = fps_values.pop()

    common_cameras = [cam for cam in episodes[0].cameras if all(cam in ep.cameras for ep in episodes)]
    if args.cameras is None:
        args.cameras = common_cameras
        dropped = sorted({cam for ep in episodes for cam in ep.cameras} - set(common_cameras))
        if dropped:
            print(f"WARN: Not including cameras missing from some episodes: {dropped}")
    elif missing := [cam for cam in args.cameras if cam not in common_cameras]:
        parser.error(f"cameras {missing} are missing from some episodes (cameras in every episode: {common_cameras})")
    print(f"Cameras: {args.cameras}, fps: {fps}")

    if args.dry_run:
        return

    if args.upload:
        from huggingface_hub import HfApi

        if "/" not in args.repo_id:
            parser.error("--repo-id must be <hf_user_or_org>/<dataset_name> to upload")
        # fail before collating, rather than after
        print(f"Uploading as Hugging Face user {HfApi().whoami()['name']}")

    # the AV1 encoder logs its config to stderr for every episode and camera, which buries our warnings
    os.environ.setdefault("SVT_LOG", "2")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = args.root or HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        if not args.overwrite:
            parser.error(f"{root} exists, pass --overwrite to replace it")
        shutil.rmtree(root)

    dataset, layout = None, None
    collated, skipped = [], []
    for episode in tqdm(episodes, desc="Collating"):
        try:
            episode_layout, data = prepare_episode(episode, layout, args)
        except Exception as e:
            tqdm.write(f"WARN: Skipping {episode.source}: {e}")
            skipped.append({"source": episode.source, "reason": str(e)})
            continue
        if dataset is None:
            layout = episode_layout
            print(f"observation.state: {layout.state_names}")
            print(f"action: {layout.action_names}")
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=fps,
                features=layout.features(),
                root=root,
                robot_type=args.robot_type,
                use_videos=True,
                streaming_encoding=True,
            )
        write_episode(dataset, episode, data)
        collated.append({
            "episode_index": len(collated),
            "source": episode.source,
            "date": episode.date,
            "n_frames": data["n_frames"],
            "lengths": data["lengths"],
            "metadata": episode.metadata,
        })

    if dataset is None:
        raise SystemExit("No episodes could be collated")
    dataset.finalize()
    sources_path = root / SOURCES_FILE
    sources_path.write_text(json.dumps({"episodes": collated, "skipped": skipped}, indent=1, default=json_default))
    print(f"Collated {len(collated)} episodes ({len(skipped)} skipped) into {root}")

    if args.upload:
        dataset.push_to_hub(private=args.private)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
