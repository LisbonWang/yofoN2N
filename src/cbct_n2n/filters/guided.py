"""3D guided filter and guided-filter sharpening for CBCT volumes.

He, Sun & Tang (2010), *Guided Image Filters*, generalised from 2D images to 3D
volumes.  Everything here is a pure function of the data -- no training, no GPU.

The filter
----------
Model the volume locally as a *linear* function of the guide ``I`` inside a
window of radius ``r`` (a ``(2r+1)^3`` box), then average the per-window
coefficients over overlapping windows::

    a  = cov(I, p) / (var(I) + eps)          b  = mean(p) - a * mean(I)
    A  = box_r(a)                            B  = box_r(b)
    q  = A * I + B

``eps`` is the regularisation: it is the variance above which a local
fluctuation is treated as signal rather than noise.  With ``p == I`` (self
guidance) ``a = var/(var+eps)`` is exactly the **local Wiener gain** -- the
optimal fraction of a fluctuation that is signal -- so ``a -> 0`` in a flat,
noise-dominated region and ``a -> 1`` on an edge.

Free of charge, this buys an O(1)-per-voxel edge-preserving smoother: the cost
does not depend on ``r`` because every window statistic is a separable box mean
(six passes of ``scipy.ndimage.uniform_filter``), not a per-window regression.

Turning it into a sharpener
---------------------------
Let ``m = box_r(p)`` be the local mean.  Two ways to boost detail:

``detail`` (textbook guided detail enhancement, ``q + s*(p - q)``)
    The boost acts on the **whole** residual ``p - q = (1-a)*(p - m)``.  In a
    flat region ``a -> 0``, so the residual there *is* the noise and gets
    amplified by ``s``.  Sharpening and noise amplification are inseparable.

``wiener`` (this module's default)
    Boost the **structure estimate** instead::

        out = p + s * (q - m)

    Because ``q - m = A*(p - m)`` whenever ``p`` is locally slowly varying,
    this multiplies the local deviation by ``A`` *before* amplifying it: flat
    regions (``A -> 0``) are left untouched and only the edge/high-SNR part of
    the signal is boosted.  ``s = 0`` gives back the input exactly, and a pure
    noise field has ``A -> 0`` everywhere, so the gain stays bounded.

Both modes are linear in ``s``, so a strength sweep only costs the metrics --
the filter itself is computed once per ``(radius, eps)``; see
:func:`guided_structure_field`.

Memory
------
A 360x640x640 float32 volume is 590 MB, and the filter needs ~10 volumes of
scratch, i.e. ~6 GB if processed in one go.  :func:`guided_filter_3d` therefore
streams along ``z`` in slabs; a slab is read together with a halo of ``2*r``
slices, which is exactly the support the two chained box means need, so the
slabbed result is *identical* to the whole-volume result (verified in
``tests/filters/test_guided_filter.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = [
    "MODES",
    "GuidedFilterResult",
    "box_mean_3d",
    "cap_darkening",
    "eps_from_noise",
    "guided_filter_3d",
    "guided_sharpen_3d",
    "guided_structure_field",
    "iter_z_slabs",
]

#: Sharpening modes accepted by :func:`guided_structure_field`.
MODES = ("wiener", "detail")


def eps_from_noise(sigma: float, k: float = 2.0) -> float:
    """Regularisation ``eps = (k * sigma)^2`` from a noise-std estimate.

    ``k`` is the "how many sigma is a fluctuation before it counts as signal"
    knob: ``k = 1`` keeps anything above the noise floor, ``k = 3`` only keeps
    what clearly stands out.  For a volume with noise std ``sigma`` the local
    Wiener gain at a truly flat voxel is ``a = sigma^2/(sigma^2+eps) =
    1/(1+k^2)``, e.g. ``k = 2`` -> 0.2, ``k = 4`` -> 0.06.
    """
    sigma = float(sigma)
    k = float(k)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"k must be positive and finite, got {k!r}")
    return (k * sigma) ** 2


def box_mean_3d(volume: np.ndarray, radius: int) -> np.ndarray:
    """Separable ``(2r+1)^3`` box mean, ``mode="nearest"``, float32 in/out.

    Cost is independent of ``radius`` (three 1D running-sum passes).
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius!r}")
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"volume must be 3D (z, y, x), got shape {data.shape}")
    return uniform_filter(data, size=2 * radius + 1, mode="nearest")


def iter_z_slabs(nz: int, slab: int, halo: int) -> Iterator[tuple[int, int, int, int]]:
    """Split ``[0, nz)`` into slabs, each with its halo clipped to the volume.

    Yields ``(z0, z1, gz0, gz1)``: the slab ``[z0, z1)`` is computed from
    ``[gz0, gz1)`` and kept at local offsets ``[z0 - gz0, z1 - gz0)``.  At the
    volume ends the halo is simply shorter -- ``mode="nearest"`` replicates the
    edge slice in both the slabbed and the whole-volume computation, so the
    clipped halo is exact, not an approximation.
    """
    if nz < 1:
        raise ValueError(f"nz must be >= 1, got {nz!r}")
    if slab < 1:
        raise ValueError(f"slab must be >= 1, got {slab!r}")
    if halo < 0:
        raise ValueError(f"halo must be >= 0, got {halo!r}")
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        yield z0, z1, max(0, z0 - halo), min(nz, z1 + halo)


@dataclass
class GuidedFilterResult:
    """Output of :func:`guided_filter_3d` (all float32, volume-shaped)."""

    base: np.ndarray
    """``q`` -- the filtered volume (``A * I + B``)."""

    mean_src: np.ndarray
    """``m`` -- the box mean of the *source* volume (the local DC level).

    Fixed by the source and the radius only: it is the zero point against which
    every structure estimate is measured, so it is worth returning instead of
    recomputing.
    """

    a: np.ndarray | None
    """Local Wiener gain ``cov(I,p)/(var(I)+eps)`` *before* the second box mean.

    Only returned when ``keep_a=True``; it is the map that says where the
    filter considers a fluctuation to be signal (``a -> 1``) or noise
    (``a -> 0``), which is what makes the sharpening auditable.
    """

    radius: int
    eps: float

    @property
    def structure(self) -> np.ndarray:
        """``q - m``: the guided estimate of the local structure component."""
        return self.base - self.mean_src


def _guided_filter_block(
    guide: np.ndarray, src: np.ndarray, radius: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-block guided filter (see :func:`guided_filter_3d` for the slabbed one)."""
    size = 2 * radius + 1
    mean_g = uniform_filter(guide, size=size, mode="nearest")
    mean_s = uniform_filter(src, size=size, mode="nearest")

    # Exactly He et al.: var = E[I^2] - E[I]^2, cov = E[I*p] - E[I]*E[p].
    # The obvious "stable" rewrite, E[(I - mean_I)^2], is *not* this quantity:
    # each sample gets centred by its own local mean, which inflates the result
    # by the variation of the local mean across the window (measured on r4b:
    # 3.5x too large).  Centring by a coarse level instead is worse still, as the
    # level is not locally constant (median 32 % error).  The plain form in
    # float32 is accurate here: against an exact float64 sliding-window variance
    # on r4b it lands at 3.6e-5 median relative error, 2.6e-6 in flat tissue.
    corr_gg = uniform_filter(guide * guide, size=size, mode="nearest")
    corr_gs = uniform_filter(guide * src, size=size, mode="nearest")
    var_g = corr_gg - mean_g * mean_g
    del corr_gg
    cov = corr_gs - mean_g * mean_s
    del corr_gs

    a = cov / (var_g + eps)
    del cov, var_g
    b = mean_s - a * mean_g

    A = uniform_filter(a, size=size, mode="nearest")
    B = uniform_filter(b, size=size, mode="nearest")
    base = A * guide + B
    return base, mean_s, a


def guided_filter_3d(
    guide: np.ndarray,
    src: np.ndarray | None = None,
    *,
    radius: int = 4,
    eps: float,
    slab: int | None = None,
    keep_a: bool = False,
    on_slab: Callable[[int, int], None] | None = None,
) -> GuidedFilterResult:
    """3D guided filter: filter ``src`` using the edge structure of ``guide``.

    ``src=None`` means self-guidance (``p = I``), the sharpening case.  ``eps``
    is mandatory: there is no defensible default for "what variance is noise" --
    use :func:`eps_from_noise` with a measured noise std.

    ``slab`` bounds the peak memory by streaming along ``z`` (see module
    docstring); ``None`` processes the volume in one block.  ``on_slab`` is an
    optional progress callback ``(z0, z1)``.
    """
    guide_array = np.asarray(guide, dtype=np.float32)
    if guide_array.ndim != 3:
        raise ValueError(f"guide must be 3D (z, y, x), got shape {guide_array.shape}")
    if src is None:
        src_array = guide_array
    else:
        src_array = np.asarray(src, dtype=np.float32)
        if src_array.shape != guide_array.shape:
            raise ValueError(
                f"guide shape {guide_array.shape} != src shape {src_array.shape}"
            )
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius!r}")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")

    base = np.empty(guide_array.shape, dtype=np.float32)
    mean_src = np.empty(guide_array.shape, dtype=np.float32)
    a_map = np.empty(guide_array.shape, dtype=np.float32) if keep_a else None

    nz = guide_array.shape[0]
    slab_len = nz if slab is None else int(slab)
    halo = 2 * radius  # a,b must be valid one more radius out: 2r, not r
    for z0, z1, gz0, gz1 in iter_z_slabs(nz, slab_len, halo):
        block_base, block_mean, block_a = _guided_filter_block(
            guide_array[gz0:gz1], src_array[gz0:gz1], radius, eps
        )
        lo = z0 - gz0
        hi = lo + (z1 - z0)
        base[z0:z1] = block_base[lo:hi]
        mean_src[z0:z1] = block_mean[lo:hi]
        if a_map is not None:
            a_map[z0:z1] = block_a[lo:hi]
        del block_base, block_mean, block_a
        if on_slab is not None:
            on_slab(z0, z1)

    return GuidedFilterResult(base=base, mean_src=mean_src, a=a_map, radius=radius, eps=eps)


def guided_structure_field(
    volume: np.ndarray,
    *,
    radius: int = 4,
    eps: float,
    mode: str = "wiener",
    guide: np.ndarray | None = None,
    slab: int | None = None,
) -> np.ndarray:
    """The per-voxel field ``d`` such that ``out = volume + strength * d``.

    Isolating the field makes a strength sweep almost free: the guided filter is
    the expensive part and it does not depend on the strength, because both
    modes are linear in it.

    ``mode="wiener"`` -> ``d = q - m`` (structure estimate, noise-safe)
    ``mode="detail"`` -> ``d = volume - q`` (guided residual, noise included)
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (choose {'|'.join(MODES)})")
    if guide is not None and mode != "wiener":
        raise ValueError(f"mode={mode!r} is self-guided only; use mode='wiener' with --guide")
    data = np.asarray(volume, dtype=np.float32)
    result = guided_filter_3d(
        data if guide is None else guide, data, radius=radius, eps=eps, slab=slab
    )
    if mode == "wiener":
        return result.structure
    return np.subtract(data, result.base, dtype=np.float32)


def cap_darkening(
    sharpened: np.ndarray,
    reference: np.ndarray,
    max_darken: float,
) -> np.ndarray:
    """Bound how much darker ``sharpened`` may become than ``reference``.

    Why this exists
    ---------------
    The field is ``q - m ~= A*(p - m)``, so where ``A -> 1`` (a strong edge) the
    operator degenerates into unsharp masking with a window of radius ``r``.  At
    a *low-density gap narrower than that window* the local mean ``m`` is pulled
    up by the dense walls, so the gap floor is driven far below its true value --
    the dark line ("黑边") that reads as ringing.  Measured on the project volume:
    the floor of a 4-voxel gap went ``0.35 -> -0.18`` at ``r=2, s=1``.

    ``eps``/``k`` cannot touch this (at a strong edge ``var >> eps``, so ``a`` is
    ~1 whatever ``k`` is); only ``radius``, ``strength`` or this bound can.

    The bound
    ---------
    With ``dark = max(reference - sharpened, 0)``:

        dark' = D * (1 - exp(-dark / D))

    so small darkenings pass untouched (``f'(0) = 1``: no kink is introduced),
    large ones saturate at ``D``, and the *correction* ``dark - dark'`` is
    monotone in the overshoot.  Because the bound follows ``reference``, which is
    smooth, the result stays smooth -- unlike a local-min bound, which flattens
    the offending valley into a plateau and inflates the high-frequency band.

    Only the dark side is bounded, and deliberately so: at trabecular bone the
    gain comes from brightening the trabeculae *and* darkening the marrow, so a
    two-sided bound throws away about half the contrast gain (measured: bone ROI
    sigma +93.6% -> +0.0% at D=0.10, i.e. it disabled the sharpening).

    ``max_darken`` is in the volume's own units (the RAW is float32 attenuation,
    tissue ~0.2-0.4).  ``None`` disables the bound.
    """
    if max_darken is None:
        return sharpened
    depth = float(max_darken)
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError(f"max_darken must be positive and finite, got {max_darken!r}")
    dark = np.subtract(np.asarray(reference, dtype=np.float32),
                       np.asarray(sharpened, dtype=np.float32))
    np.maximum(dark, 0.0, out=dark)
    # `recovered` is 0 where nothing darkened, so the bright side is untouched.
    recovered = dark - depth * (1.0 - np.exp(-dark / np.float32(depth)))
    return np.add(sharpened, recovered, dtype=np.float32)


def guided_sharpen_3d(
    volume: np.ndarray,
    *,
    radius: int = 4,
    eps: float,
    strength: float = 1.0,
    mode: str = "wiener",
    guide: np.ndarray | None = None,
    slab: int | None = None,
    out: np.ndarray | None = None,
    max_darken: float | None = None,
) -> np.ndarray:
    """Sharpen ``volume``: ``out = volume + strength * guided_structure_field(...)``.

    ``strength = 0`` returns the input bit-for-bit (in float32), which is the
    identity check used in the tests.  Values are neither clipped nor otherwise
    modified, so out-of-range overshoot stays visible to the caller -- except
    that ``max_darken`` (see :func:`cap_darkening`) can bound the dark side.
    """
    strength = float(strength)
    field = guided_structure_field(
        volume, radius=radius, eps=eps, mode=mode, guide=guide, slab=slab
    )
    if out is None:
        out = np.empty(field.shape, dtype=np.float32)
    elif out.shape != field.shape:
        raise ValueError(f"out shape {out.shape} != volume shape {field.shape}")
    if strength == 1.0:
        np.add(field, np.asarray(volume, dtype=np.float32), out=out)
    else:
        # Scale the field in place (it is freshly allocated here), then add.
        np.multiply(field, np.float32(strength), out=field)
        np.add(field, np.asarray(volume, dtype=np.float32), out=out)
    if max_darken is not None:
        out = cap_darkening(out, volume, max_darken)
    return out
