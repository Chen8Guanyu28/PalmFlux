# PalmFlux

PalmFlux is a lightweight browser-based hand-tracked particle interaction demo built for lower latency and more reliable gesture behavior than the earlier prototype.

## What changed

- Switched from canned gesture recognition to `HandLandmarker` plus custom gesture logic for faster, more controllable interaction.
- Pinch uses normalized thumb-index distance with hysteresis, which is more stable across different camera distances.
- Follow behavior uses frame-rate independent smoothing and velocity clamping, so the particle field tracks the hand without jumpy overshoot.
- The project is plain static frontend code, so it can run directly in VS Code Live Server or any static hosting setup.

## Gesture map

- `pinch`: grab and drag the particle cluster
- `fist`: switch particle shape
- `open palm`: trigger burst
- `victory`: inject swirl motion
- `index point`: attract particles toward the finger direction

## Run locally

Open the folder with VS Code and run Live Server on `index.html`, or serve it with any static server.

```bash
python -m http.server 5500
```

Then open `http://127.0.0.1:5500`.

## Notes

Recognition quality still depends on camera quality, frame rate, light, and hand visibility. This version is optimized for responsiveness and robustness, but no browser-based webcam tracker can guarantee perfect accuracy on every device.
