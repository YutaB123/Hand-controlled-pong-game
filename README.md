# Hand Tennis

A two-player browser Pong where your hands are the paddles. Uses your webcam and MediaPipe Hands to track each hand in real time — left hand drives the left paddle, right hand drives the right.

## Run it locally

Pure static site. Serve over HTTP (webcam access needs a secure context):

```bash
npx serve .
# or
python -m http.server 8000
```

Open `http://localhost:8000` and allow camera access. Click **Start Game**.

## Files

- `index.html` — page structure
- `style.css` — visuals
- `game.js` — game loop, paddle and ball physics, hand-tracking glue
- Hand tracking via `@mediapipe/tasks-vision` loaded from a CDN — no build step

## Controls

- **Left hand** → left paddle
- **Right hand** → right paddle