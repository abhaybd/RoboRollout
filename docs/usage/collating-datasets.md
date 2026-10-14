# Collating datasets

[`scripts/collate_lerobot.py`](https://github.com/abhaybd/RoboRollout/blob/main/scripts/collate_lerobot.py) collates saved episodes into a [LeRobot](https://github.com/huggingface/lerobot) dataset, for example to fine-tune a policy on its own successful rollouts. Episodes can come from local output directories, from W&B, or from both, and can be filtered by date and by any of their metadata.

## Installation

The script declares its own dependencies (including LeRobot, which needs Python 3.12), so it runs separately from the robot environments. Install [uv](https://docs.astral.sh/uv/) if needed, and run it with:

```bash
uv run scripts/collate_lerobot.py --help
```

uv creates an environment with the script's dependencies on the first run.

To read episodes from W&B, log in with `wandb login`. To upload datasets, log in to Hugging Face with `hf auth login`.

## Usage

Pass any number of sources, and the dataset's `--repo-id`:

```bash
uv run scripts/collate_lerobot.py outputs/ wandb:my-entity/droid-evals \
    --repo-id my-hf-user/droid-pick-mug \
    --since 2026-10-07 --filter success=true --filter "task~=mug"
```

The script lists the matching episodes, then writes the dataset to `$HF_LEROBOT_HOME/<repo-id>` (by default `~/.cache/huggingface/lerobot/<repo-id>`), or to `--root` if given. If the dataset already exists, pass `--overwrite` to replace it.

| Flag | Description |
| --- | --- |
| `--dry-run` | Only list the episodes that would be collated, without downloading or writing anything |
| `--upload` | Upload the dataset to the Hugging Face Hub as `--repo-id` after collating it. Add `--private` to make it private. |
| `--root` | Where to write the dataset |
| `--overwrite` | Replace the dataset at `--root` if it exists |
| `--robot-type` | Robot type to save in the dataset's metadata, e.g. `franka` |
| `--cache-dir` | Where to download episodes from W&B (default: `~/.cache/robo_rollout/wandb`) |

!!! tip
    Run with `--dry-run` first to check that your filters select the episodes you expect.

### Sources

- **Local directories** are searched recursively for episodes, so a source can be a single episode directory, a run's output directory, or a parent such as `outputs/`. Each episode's config is read from `.hydra/config.yaml` in its run directory.
- **W&B projects** are given as `wandb:<entity>/<project>`, and every run in the project is treated as an episode. Only the files needed for the dataset are downloaded: `observations.npz`, `actions.npz`, and the observation-synced videos. Downloads are cached in `--cache-dir`, so collating the same runs again doesn't download them again. Runs that crashed before logging their episode info are skipped.

### Filtering

Use `--since` and `--until` to select episodes by date. Each takes a date (`2026-10-07`) or a datetime (`2026-10-07T14:30`) in local time, and both bounds are inclusive, so `--until 2026-10-14` includes the whole day. A local episode's date is when its `info.json` was written, at the end of the episode. A W&B episode's date is when its run was created, at the start of the episode.

Use `--filter KEY<op>VALUE` to select episodes by their metadata. Repeat it to require several filters. The operators are:

| Operator | Matches when the value |
| --- | --- |
| `=`, `!=` | equals, or doesn't equal, `VALUE` |
| `~=` | contains a match for the regex `VALUE` |
| `<`, `<=`, `>`, `>=` | compares to `VALUE` as shown |

Values are parsed like YAML, so `success=true` matches the boolean `true`, and `episode_length_steps>=100` compares numbers. Quote filters that contain spaces or shell characters, e.g. `--filter "task=put the red mug in the black bowl"` or `--filter "episode_length_steps>=100"`. If the metadata value is a list, such as tags, a filter matches if any element does (or for `!=`, if no element does). An episode without the key fails every filter except `!=`.

Keys are joined with `.` and can refer to:

| Key | Description |
| --- | --- |
| `task`, `success`, `episode_length_steps`, ... | Everything in [`info.json`](running-evaluations.md#infojson) |
| `config.<key>` | The episode's config, e.g. `config.policy_dt` or `config.wandb.tags` |
| `config.policy_metadata.<key>` | The policy server's metadata. Only available for W&B episodes, since it isn't saved locally. |
| `wandb.id`, `wandb.name`, `wandb.tags`, `wandb.state` | The W&B run's ID, name, tags, and state. Only available for W&B episodes. |
| `date` | The episode's date as an ISO string, e.g. `2026-10-07T14:30:00` |
| `source` | The episode's directory, or `wandb:<entity>/<project>/<run_id>` |

For example, to collate the successful episodes of one policy over a week:

```bash
uv run scripts/collate_lerobot.py wandb:my-entity/droid-evals --repo-id my-hf-user/pi0-successes \
    --since 2026-10-07 --until 2026-10-13 \
    --filter success=true --filter config.policy_metadata.model_name=pi0
```

## Dataset contents

Each episode becomes one LeRobot episode, with these features:

| Feature | Description |
| --- | --- |
| `observation.images.<camera>` | Each camera's observation-synced video (`<camera>.mp4`), at full resolution |
| `observation.state` | The `--state-keys` of `observations.npz`, concatenated in order |
| `action` | The `--action-keys` of `actions.npz`, concatenated in order |
| `task` | The episode's task |

Frame `t` holds the observation at step `t` and the action the policy took from it. The dataset's frame rate is `round(1 / policy_dt)`, the rate of the observation-synced videos, so all episodes must have the same `policy_dt`. If they don't, filter on `config.policy_dt`.

By default, every numeric key of `observations.npz` is included in `observation.state`, and every numeric key of `actions.npz` except `server_timing/*` is included in `action`. Keys with more than one dimension per step (such as action chunks) are left out. To choose the keys and their order, pass them explicitly. For example, for the DROID robot:

```bash
uv run scripts/collate_lerobot.py outputs/ --repo-id my-hf-user/droid-pick-mug \
    --state-keys qpos/arm qpos/gripper --action-keys arm gripper
```

The names of each dimension (e.g. `qpos/arm.0`, ..., `qpos/gripper`) are saved in the dataset's feature metadata, and printed when collation starts.

By default, every camera that all episodes have is included. To pick cameras, pass `--cameras`.

The key layout and image sizes are set by the first episode collated. Any later episode that doesn't match them is skipped with a warning, as are episodes with missing files or no steps.

### Provenance

The source of each collated episode is saved to `meta/robo_rollout_sources.json` in the dataset, along with its date, metadata (including its config), and the length of each of its observations, actions, and videos before trimming. Skipped episodes are listed with the reason they were skipped.
