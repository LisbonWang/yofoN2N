"""Density-adaptive blending of two denoised full volumes (recursive r3 target).

The recursive scheme's rounds 1 and 2 each push the supervision target
globally cleaner (``full.dn_3d_128_r1``, ``full.dn_3d_128_r2``).  Round 3 is a
*lateral* refinement instead of another global push: it mixes the two
already-written targets voxel-wise, keyed on the r2 (aggressive) intensity --
the denoised CT value in the mu domain -- so that low-density tissue keeps the
conservative r1 result and high-density (bone/enamel) takes the aggressive r2
result, with a smooth crossfade in between.

Blend rule (per voxel, ``v`` = r2 value, ``lo=0.2``, ``hi=0.4``)::

    w(v) = 0                        v <= lo    -> pure r1 (conservative)
    w(v) = smoothstep((v-lo)/(hi-lo)) lo < v < hi -> smooth crossfade
    w(v) = 1                        v >= hi    -> pure r2 (aggressive)

    target = r1 + w * (r2 - r1)

The watershed ``v = (lo+hi)/2 = 0.3`` gives ``w = 0.5``.  ``smoothstep`` is
C1-continuous with zero slope at both ends, so the crossfade introduces no
gradient discontinuity at the band edges.

.. note::
    This is the **target layer only**.  Under half->full training the student's
    output is *cleaner than its supervision target* (Noise2Noise averages out
    the target's residual noise), so the final effect is one recursion step
    deeper than this blend: soft tissue ends up ~r2 (NOT r1), bone exceeds r2.
    See ``docs/plans/r3-density-blend.md`` sec. 1-2.
"""
from __future__ import annotations

import numpy as np

__all__ = ["smoothstep", "cosine_ramp", "density_blend_weight", "blend_volumes"]


def smoothstep(t: np.ndarray) -> np.ndarray:
    """C1 smooth ramp 0->1 over [0,1]; zero derivative at both ends."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cosine_ramp(t: np.ndarray) -> np.ndarray:
    """C1 cosine ramp 0->1 over [0,1] (alternative smooth crossfade)."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


_RAMPS = {
    "smoothstep": smoothstep,
    "cosine": cosine_ramp,
    "linear": lambda t: np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0),
}


def density_blend_weight(values: np.ndarray, lo: float = 0.2, hi: float = 0.4,
                         kind: str = "smoothstep") -> np.ndarray:
    """Per-voxel blend weight ``w`` in [0,1] selecting the ``hi``-side volume.

    ``values <= lo`` -> ``w = 0`` (low-density / conservative side); ``values
    >= hi`` -> ``w = 1`` (high-density / aggressive side); otherwise a smooth
    ramp whose midpoint (``w = 0.5``) sits at ``(lo+hi)/2``.
    """
    if hi <= lo:
        raise ValueError(f"hi ({hi}) must be > lo ({lo})")
    try:
        ramp = _RAMPS[kind]
    except KeyError:
        raise ValueError(f"unknown ramp kind {kind!r} (choose smoothstep|cosine|linear)") from None
    values = np.asarray(values, dtype=np.float32)
    return ramp((values - lo) / (hi - lo))


def blend_volumes(low_vol: np.ndarray, high_vol: np.ndarray,
                  lo: float = 0.2, hi: float = 0.4, kind: str = "smoothstep",
                  mask_source: str = "high") -> np.ndarray:
    """Voxel-wise blend of ``low_vol`` (r1) and ``high_vol`` (r2).

    The mask key defaults to ``high`` (the r2 result): it is already denoised,
    so its intensity -- unlike the noisy ``full`` -- does not jitter the blend
    boundary.  ``low_vol`` and ``high_vol`` must be same-shape; they are the
    same ``full`` reconstruction denoised at two aggressions, so no
    registration is needed.

    Returns ``low_vol + w * (high_vol - low_vol)`` as float32.
    """
    low = np.asarray(low_vol, dtype=np.float32)
    high = np.asarray(high_vol, dtype=np.float32)
    if low.shape != high.shape:
        raise ValueError(f"shape mismatch: r1 {low.shape} vs r2 {high.shape}")
    key = high if mask_source == "high" else low
    w = density_blend_weight(key, lo, hi, kind)
    return low + w * (high - low)
