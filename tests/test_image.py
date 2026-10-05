import numpy as np
import pytest

from robo_rollout.utils.image import downsample, resize_with_crop, resize_with_pad


def test_downsample_takes_every_nth_pixel():
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    np.testing.assert_array_equal(downsample(img, 2, 3), img[::2, ::2])


def test_downsample_rejects_non_integer_factor():
    with pytest.raises(AssertionError):
        downsample(np.zeros((720, 1280, 3), np.uint8), 224, 224)


def test_resize_with_pad_letterboxes_wide_image():
    out = resize_with_pad(np.full((50, 100, 3), 255, np.uint8), 100, 100)
    assert out.shape == (100, 100, 3)
    # the 100x50 image is centered vertically, with 25 rows of zero padding above and below
    assert (out[:25] == 0).all() and (out[75:] == 0).all()
    assert (out[25:75] == 255).all()


def test_resize_with_crop_keeps_center_of_wide_image():
    # four vertical stripes of 25 columns each
    img = np.repeat(np.array([10, 20, 30, 40], np.uint8), 25)[None, :, None].repeat(50, axis=0).repeat(3, axis=2)
    out = resize_with_crop(img, 50, 50)
    assert out.shape == (50, 50, 3)
    # only the middle two stripes remain, undistorted and unpadded
    assert (out[:, :20] == 20).all() and (out[:, 30:] == 30).all()


def test_resize_with_crop_fills_target_for_non_integer_scale():
    # 720p to 255x255 scales by a factor that truncates to 254 rows; rounding must not leave an unfilled edge
    out = resize_with_crop(np.full((720, 1280, 3), 255, np.uint8), 255, 255)
    assert out.shape == (255, 255, 3)
    assert (out == 255).all()


@pytest.mark.parametrize("resize", [resize_with_pad, resize_with_crop])
def test_resize_preserves_batch_dims(resize):
    out = resize(np.zeros((2, 3, 50, 100, 3), np.uint8), 64, 64)
    assert out.shape == (2, 3, 64, 64, 3)
