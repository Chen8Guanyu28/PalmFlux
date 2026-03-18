# PalmFlux Sphere

A simplified version of the earlier prototype.

## What changed

- Removed all discrete gesture switching.
- Removed shape switching entirely.
- The particle system is now always a sphere.
- One hand continuously controls only one parameter: sphere radius.
- More open hand -> larger sphere.
- More closed hand -> smaller sphere.
- Added stronger smoothing on both landmarks and control signal to reduce jitter.

## Why this should feel more stable

Instead of classifying several gesture categories every frame, this version uses one continuous measurement:

`average fingertip distance to palm center / palm width`

That signal is then smoothed in two stages:
1. landmark smoothing
2. openness smoothing + deadband

This substantially reduces flicker and random jumping compared with category-based gesture switching.

## Run

Serve the folder statically.

```bash
python -m http.server 5500
```

Open `http://127.0.0.1:5500`.
