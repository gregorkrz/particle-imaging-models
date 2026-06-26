"""Feature, color, and energy transforms."""

from .common import *


@TRANSFORMS.register_module()
class NormalizeColor(object):
    """Scale per-point colors from ``[0, 255]`` into ``[0, 1]``.

    Reads and overwrites ``data_dict["color"]`` in place by dividing it by
    ``255``. A no-op when no ``"color"`` key is present. Registered as
    ``NormalizeColor`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> data = {"color": np.array([[0., 128., 255.]], dtype="f4")}
            >>> NormalizeColor()(data)["color"].round(3)
            array([[0.   , 0.502, 1.   ]], dtype=float32)  # divided by 255 -> [0, 1]
    """

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 255
        return data_dict

@TRANSFORMS.register_module()
class LogTransform(object):
    """Compress scalar features (e.g. energy) onto ``[-1, 1]``.

    For each key in ``keys`` that is present, replaces the value with either a
    logarithmic (``log=True``) or linear (``log=False``) rescaling onto
    ``[-1, 1]`` derived from ``min_val``/``max_val``. The log map is
    ``2 * (log10(x + min_val) - log10(min_val)) / (log10(max_val + min_val) -
    log10(min_val)) - 1``. Raises ``ValueError`` if a requested key is missing.
    Registered as ``LogTransform`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        min_val (float): Lower reference value of the input range; also the
            additive offset inside the log. Defaults to ``1.0e-2``.
        max_val (float): Upper reference value of the input range. Defaults to
            ``20.0``.
        log (bool): If ``True`` use the logarithmic map, otherwise the linear
            map. Defaults to ``True``.
        keys (tuple): Keys to transform; a single string is wrapped into a
            tuple. Defaults to ``("energy",)``.
        clip (bool): If ``True``, clip inputs before mapping (to
            ``[0, max_val]`` for log, ``[min_val, max_val]`` for linear).
            Defaults to ``False``.

    Note:
        The correct ``min_val`` matters: for PoLAr-MAE/PILArNet energy it must
        equal the energy threshold (e.g. ``0.13``), not ``0.01`` — a wrong
        value degrades downstream results.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> data = {"energy": np.array([[0.13], [1.0], [20.0]], dtype="f4")}
            >>> LogTransform(min_val=0.13, max_val=20.0)(data)["energy"].round(3)
            array([[-0.725],
                   [-0.142],
                   [ 1.   ]])  # log-compressed: min_val->-1, max_val->+1
    """

    def __init__(self, min_val=1.0e-2, max_val=20.0, log=True, keys=("energy",), clip=False):
        self.min_val = min_val
        self.max_val = max_val
        self.log = log
        self.clip = clip
        keys = tuple(keys) if isinstance(keys, (list, tuple)) else (keys,)
        self.keys = keys

    def log_transform(self, x):
        """Transform energy to logarithmic scale on [-1,1]"""
        if self.clip:
            x = np.clip(x, 0.0, self.max_val)
        # [emin, emax] -> [-1,1]
        y0 = np.log10(self.min_val)
        y1 = np.log10(self.max_val + self.min_val)
        return 2 * (np.log10(x + self.min_val) - y0) / (y1 - y0) - 1

    def linear_transform(self, x):
        """Transform energy to linear scale on [-1,1]"""
        if self.clip:
            x = np.clip(x, self.min_val, self.max_val)
        return 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k] = (
                    self.log_transform(data_dict[k])
                    if self.log
                    else self.linear_transform(data_dict[k])
                )
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class RelativeLogNormalize(object):
    """Per-event relative log normalization (e.g. for hit times).

    For each key in ``keys``, subtracts the per-event minimum, clips to
    ``[0, max_val]``, applies ``log1p(x / scale)`` normalized by
    ``log1p(max_val / scale)``, then maps onto ``[out_min, out_max]``. Useful
    for trigger-offset-removed timing where only relative spacing matters.
    Raises ``ValueError`` if a requested key is missing. Registered as
    ``RelativeLogNormalize`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        keys (tuple): Keys to transform; a single string is wrapped into a
            tuple. Defaults to ``("time",)``.
        scale (float): Log soft-knee scale (``log1p(x / scale)``); must be
            positive. Defaults to ``50.0``.
        max_val (float): Upper clip applied before the log; must be positive.
            Defaults to ``4000.0``.
        out_min (float): Lower bound of the output range. Defaults to ``-1.0``.
        out_max (float): Upper bound of the output range; must exceed
            ``out_min``. Defaults to ``1.0``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> data = {"time": np.array([[100.], [150.], [4100.]], dtype="f4")}
            >>> RelativeLogNormalize(keys=("time",), scale=50.0)(data)["time"].round(3)
            array([[-1.   ],
                   [-0.685],
                   [ 1.   ]], dtype=float32)  # per-event min->-1, clipped max->+1
    """

    def __init__(
        self,
        keys=("time",),
        scale=50.0,
        max_val=4000.0,
        out_min=-1.0,
        out_max=1.0,
    ):
        self.scale = float(scale)
        self.max_val = float(max_val)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.max_val <= 0:
            raise ValueError("max_val must be positive")
        if self.out_max <= self.out_min:
            raise ValueError("out_max must be greater than out_min")
        keys = tuple(keys) if isinstance(keys, (list, tuple)) else (keys,)
        self.keys = keys
        self.denom = np.log1p(self.max_val / self.scale)

    def relative_log_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = x - np.min(x)
        x = np.clip(x, 0.0, self.max_val)
        y = np.log1p(x / self.scale) / self.denom
        y = self.out_min + y * (self.out_max - self.out_min)
        return np.clip(y, self.out_min, self.out_max).astype(
            np.float32, copy=False
        )

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k] = self.relative_log_transform(data_dict[k])
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class MomentumTransform(object):
    """Log10-compress strictly-positive momentum values.

    For each key in ``keys`` that is present, replaces positive entries with
    ``log10(clip(x, 1e-6, None))`` while leaving non-positive entries
    unchanged (so sentinel/zero values pass through). Registered as
    ``MomentumTransform`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        keys (tuple): Keys to transform. Defaults to ``("momentum",)``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> data = {"momentum": np.array([[0.0], [1.0], [100.0]], dtype="f4")}
            >>> MomentumTransform()(data)["momentum"].round(3)
            array([[0.],
                   [0.],
                   [2.]], dtype=float32)  # positive -> log10; 0 sentinel passes through
    """

    def __init__(self, keys=("momentum",)):
        self.keys = keys

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                mom = data_dict[k]
                mom = np.where(mom > 0, np.log10(np.clip(mom, 1e-6, None)), mom)
                data_dict[k] = mom
        return data_dict

@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    """Randomly auto-contrast the per-point RGB color.

    With probability ``p`` and when ``"color"`` is present, rescales each RGB
    channel to span ``[0, 255]`` (per-channel min/max stretch) and blends the
    result with the original color by ``blend_factor``; modifies
    ``data_dict["color"][:, :3]`` in place. Registered as
    ``ChromaticAutoContrast`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        p (float): Probability of applying the transform. Defaults to ``0.2``.
        blend_factor (float, optional): Blend weight toward the contrasted
            color; if ``None`` a fresh ``U(0, 1)`` value is drawn per call.
            Defaults to ``None``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"color": np.array([[50., 100., 150.],
            ...                            [100., 150., 200.]], dtype="f4")}
            >>> ChromaticAutoContrast(p=1.0)(data)["color"].round(2)
            array([[ 14.24,  28.48,  42.72],
                   [210.85, 225.09, 239.34]], dtype=float32)  # stretched toward [0, 255]
    """

    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict

@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    """Randomly shift all RGB channels by a shared offset.

    With probability ``p`` and when ``"color"`` is present, adds the same random
    per-channel offset (drawn in ``+/- 255 * ratio``) to every point, then clips
    to ``[0, 255]``; modifies ``data_dict["color"][:, :3]`` in place. Registered
    as ``ChromaticTranslation`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        p (float): Probability of applying the transform. Defaults to ``0.95``.
        ratio (float): Fraction of the full ``255`` range bounding the random
            offset magnitude. Defaults to ``0.05``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"color": np.array([[100., 100., 100.]], dtype="f4")}
            >>> ChromaticTranslation(p=1.0, ratio=0.1)(data)["color"].round(2)
            array([[110.97, 105.24, 102.29]], dtype=float32)  # shared per-channel offset added
    """

    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict

@TRANSFORMS.register_module()
class EnergeticTranslation(object):
    """Randomly shift per-point energy by a shared scalar offset.

    Energy analogue of :class:`ChromaticTranslation`. With probability ``p`` and
    when ``"energy"`` is present, adds one random offset (drawn in
    ``+/- ratio``) to every point's energy, then clips to ``[-1, 1]`` (so it
    expects already log/linear-normalized energy); modifies
    ``data_dict["energy"]`` in place. Registered as ``EnergeticTranslation`` —
    use this string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        p (float): Probability of applying the transform. Defaults to ``0.95``.
        ratio (float): Half-width of the uniform offset added to energy.
            Defaults to ``0.05``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"energy": np.array([[0.0], [0.5]], dtype="f4")}
            >>> EnergeticTranslation(p=1.0, ratio=0.1)(data)["energy"].round(4)
            array([[0.043 ],
                   [0.543 ]], dtype=float32)  # one shared offset added to all points
    """

    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "energy" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1) - 0.5) * 2 * self.ratio
            data_dict["energy"] = np.clip(tr + data_dict["energy"], -1, 1)
        return data_dict

@TRANSFORMS.register_module()
class ChromaticJitter(object):
    """Add independent per-point Gaussian noise to RGB channels.

    With probability ``p`` and when ``"color"`` is present, adds zero-mean
    Gaussian noise with standard deviation ``std * 255`` independently per point
    and per channel, then clips to ``[0, 255]``; modifies
    ``data_dict["color"][:, :3]`` in place. Registered as ``ChromaticJitter`` —
    use this string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        p (float): Probability of applying the transform. Defaults to ``0.95``.
        std (float): Noise standard deviation as a fraction of the ``255``
            range. Defaults to ``0.005``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"color": np.array([[100., 100., 100.]], dtype="f4")}
            >>> ChromaticJitter(p=1.0, std=0.05)(data)["color"].round(2)
            array([[109.46, 119.8 ,  71.08]], dtype=float32)  # per-point per-channel Gaussian noise
    """

    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict

@TRANSFORMS.register_module()
class EnergyJitter(object):
    """Apply multiplicative per-point jitter to energy.

    With probability ``p`` and when ``"energy"`` is present, scales each point's
    energy by ``1 + U(-jitter_ratio, jitter_ratio)`` then lower-clips to
    ``min_val``; modifies ``data_dict["energy"]`` in place. Registered as
    ``EnergyJitter`` — use this string as the ``type`` in a ``transform=[...]``
    config list.

    Args:
        p (float): Probability of applying the transform. Defaults to ``0.5``.
        jitter_ratio (float): Half-width of the multiplicative jitter applied to
            energy. Defaults to ``0.005``.
        min_val (float): Lower clip applied after jittering. Defaults to
            ``0.0``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"energy": np.array([[1.0], [2.0]], dtype="f4")}
            >>> EnergyJitter(p=1.0, jitter_ratio=0.1)(data)["energy"].round(4)
            array([[1.043 ],
                   [2.0411]], dtype=float32)  # each point scaled by 1 +/- jitter
    """

    def __init__(self, p=0.5, jitter_ratio=0.005, min_val=0.0):
        self.p = p
        self.jitter_ratio = jitter_ratio
        self.min_val = min_val

    def __call__(self, data_dict):
        if "energy" in data_dict.keys() and np.random.rand() < self.p:
            # jitter by +/- 0.5%
            jitter = (
                np.random.rand(*data_dict["energy"].shape) * 2 - 1
            ) * self.jitter_ratio
            data_dict["energy"] = np.clip(
                data_dict["energy"] * (1 + jitter), self.min_val, None
            )
        return data_dict

@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    """Randomly convert per-point RGB color to grayscale.

    With probability ``p``, replaces ``data_dict["color"]`` with its luminance
    (ITU-R ``0.2989 R + 0.587 G + 0.114 B``) broadcast back to 3 channels.
    Requires ``"color"`` with at least 3 channels. Registered as
    ``RandomColorGrayScale`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        p (float): Probability of applying the grayscale conversion.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> data = {"color": np.array([[255., 0., 0.]], dtype="f4")}
            >>> RandomColorGrayScale(p=1.0)(data)["color"].round(2)
            array([[76.22, 76.22, 76.22]], dtype=float32)  # 0.2989*R luminance, 3 channels
    """

    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict

@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """Randomly jitter brightness, contrast, saturation, and hue of point colors.

    A torchvision-style ``ColorJitter`` for 3D point-cloud ``"color"``. Each
    call draws a random order over the four adjustments and, for each enabled
    one, applies it with probability ``p`` using a factor sampled from the range
    implied by the constructor argument; modifies ``data_dict["color"]`` in
    place. Registered as ``RandomColorJitter`` — use this string as the ``type``
    in a ``transform=[...]`` config list.

    Args:
        brightness (float or tuple): Either a non-negative magnitude ``b``
            giving the factor range ``[max(0, 1 - b), 1 + b]`` or an explicit
            ``(min, max)`` pair. ``0`` disables. Defaults to ``0``.
        contrast (float or tuple): Same convention as ``brightness`` for
            contrast. Defaults to ``0``.
        saturation (float or tuple): Same convention as ``brightness`` for
            saturation. Defaults to ``0``.
        hue (float or tuple): Either a magnitude ``h`` (``0 <= h <= 0.5``)
            giving the range ``[-h, h]`` or an explicit ``(min, max)`` within
            ``[-0.5, 0.5]``. ``0`` disables. Defaults to ``0``.
        p (float): Per-adjustment probability of application. Defaults to
            ``0.95``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"color": np.array([[100., 150., 200.]], dtype="f4")}
            >>> RandomColorJitter(brightness=0.4, p=1.0)(data)["color"].round(2)
            array([[103.91, 155.86, 207.81]], dtype=float32)  # brightness factor ~1.04 applied
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict

@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    """Randomly translate hue and scale saturation of point colors.

    When ``"color"`` is present, converts ``data_dict["color"][:, :3]`` to HSV,
    adds a random hue offset in ``+/- hue_max`` (wrapped modulo 1) and multiplies
    saturation by ``1 + U(-saturation_max, saturation_max)`` (clipped to
    ``[0, 1]``), then converts back to RGB and clips to ``[0, 255]``; modifies
    the color in place. Registered as ``HueSaturationTranslation`` — use this
    string as the ``type`` in a ``transform=[...]`` config list.

    Args:
        hue_max (float): Half-width of the random hue offset (HSV hue units in
            ``[0, 1]``). Defaults to ``0.5``.
        saturation_max (float): Half-width of the random saturation scaling
            factor about 1. Defaults to ``0.2``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> data = {"color": np.array([[200., 100., 50.]], dtype="f4")}
            >>> HueSaturationTranslation(hue_max=0.5, saturation_max=0.2)(data)["color"]
            array([[200., 139.,  37.]], dtype=float32)  # hue rotated, saturation rescaled
    """

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict

@TRANSFORMS.register_module()
class RandomColorDrop(object):
    """Randomly drop (zero out or attenuate) per-point color.

    With probability ``p`` and when ``"color"`` is present, multiplies
    ``data_dict["color"]`` by ``color_augment`` in place (``0.0`` drops color
    entirely), forcing the model not to rely solely on color. Registered as
    ``RandomColorDrop`` — use this string as the ``type`` in a
    ``transform=[...]`` config list.

    Args:
        p (float): Probability of dropping color. Defaults to ``0.2``.
        color_augment (float): Multiplier applied to color when dropped (``0.0``
            zeroes it). Defaults to ``0.0``.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> data = {"color": np.array([[100., 150., 200.]], dtype="f4")}
            >>> RandomColorDrop(p=1.0, color_augment=0.0)(data)["color"]
            array([[0., 0., 0.]], dtype=float32)  # color zeroed out (dropped)
    """

    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )
