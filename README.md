# Over/Under — VEX Competition Match Analysis

Computer vision pipeline for analyzing VEX Over/Under competition matches. Extracts and labels triball positions from match video for performance analysis and strategy development.

## Features

- **Frame Extraction** — Decomposes match recordings into individual frames (100k+ labeled frames)
- **Object Detection** — Triball identification and tracking across frames
- **Label Generation** — Structured JSON dataset of triball positions per frame
- **Training Pipeline** — ML model training on extracted frame data

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3 |
| ML / CV | PyTorch, YOLO (inferred) |
| Data | JSON-based label format |
| Storage | Git LFS for large assets |

## Project Structure

```
data.py             — Dataset loading & preprocessing
train.py            — Model training entrypoint
v0.3.py             — Pipeline version 0.3 (LFS)
triball_frames/     — Extracted match frames (LFS)
triball_labels.json — Frame-level triball annotations (LFS)
m.mp4               — Source match recording (LFS)
```

## Usage

```bash
# Labels and frames are stored with Git LFS
git lfs pull

# Train detection model
python train.py
```

## Notes

Dataset is large (~100k frames, ~850 MB labels) and stored via Git LFS. Ensure `git lfs` is installed before cloning.
