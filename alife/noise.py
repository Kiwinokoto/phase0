from __future__ import annotations

import numpy as np


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def value_noise_2d(
    rng: np.random.Generator,
    width: int,
    height: int,
    grid_w: int,
    grid_h: int,
) -> np.ndarray:
    """Generate tileable-ish 2D value noise with bilinear interpolation.

    It is not intended to be physically realistic. It is only a compact,
    dependency-free way to get coherent continents and climate patches.
    """
    grid_w = max(2, int(grid_w))
    grid_h = max(2, int(grid_h))

    # +1 so interpolation has a right/bottom edge to sample.
    lattice = rng.random((grid_h + 1, grid_w + 1), dtype=np.float64)

    xs = np.linspace(0.0, grid_w, width, endpoint=False)
    ys = np.linspace(0.0, grid_h, height, endpoint=False)

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    tx = _smoothstep(xs - x0)
    ty = _smoothstep(ys - y0)

    v00 = lattice[y0[:, None], x0[None, :]]
    v10 = lattice[y0[:, None], x1[None, :]]
    v01 = lattice[y1[:, None], x0[None, :]]
    v11 = lattice[y1[:, None], x1[None, :]]

    top = v00 * (1.0 - tx[None, :]) + v10 * tx[None, :]
    bottom = v01 * (1.0 - tx[None, :]) + v11 * tx[None, :]
    return top * (1.0 - ty[:, None]) + bottom * ty[:, None]


def fractal_noise_2d(
    rng: np.random.Generator,
    width: int,
    height: int,
    base_grid: int = 4,
    octaves: int = 5,
    gain: float = 0.5,
) -> np.ndarray:
    """Layer several value-noise octaves and normalize to [0, 1]."""
    total = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    amplitude_sum = 0.0

    for octave in range(octaves):
        scale = base_grid * (2**octave)
        # Keep horizontal detail roughly proportional to map aspect ratio.
        grid_w = scale * 2
        grid_h = scale
        total += amplitude * value_noise_2d(rng, width, height, grid_w, grid_h)
        amplitude_sum += amplitude
        amplitude *= gain

    total /= max(amplitude_sum, 1e-9)
    min_v = float(total.min())
    max_v = float(total.max())
    return (total - min_v) / max(max_v - min_v, 1e-9)
