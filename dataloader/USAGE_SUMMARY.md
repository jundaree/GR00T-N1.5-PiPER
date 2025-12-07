# Quick Usage Guide

## What Changed

The dataloader now treats **each datasetX folder as one episode** and combines all episodes from a task into a single multi-episode LeRobot dataset.

### Previous Behavior
- Each dataset was converted separately
- Output: `datasets_preprocessed/<task>/<dataset_name>/`

### New Behavior
- All datasets in a task are combined into one dataset
- Each datasetX becomes episode_XXXXXX
- Output: `datasets_preprocessed/<task>/`

## Quick Commands

### Convert All Tasks
```bash
python dataloader/dataloader.py
```
This creates:
- `datasets_preprocessed/bin_pushing/` (29 episodes)
- `datasets_preprocessed/pick_and_place/` (30 episodes)

### Convert One Task (Recommended)
```bash
# Bin pushing only (without depth)
python dataloader/dataloader.py --task bin_pushing

# Pick and place with depth images
python dataloader/dataloader.py --task pick_and_place --include-depth
```

### Example Output Structure
```
datasets_preprocessed/bin_pushing/
├── data/chunk-000/
│   ├── episode_000000.parquet  (dataset1 → 765 frames)
│   ├── episode_000001.parquet  (dataset2 → 922 frames)
│   ├── episode_000002.parquet  (dataset3 → 672 frames)
│   └── ...                     (29 episodes total, 19,427 frames)
├── videos/chunk-000/
│   ├── observation.images.logi_rgb/
│   │   ├── episode_000000.mp4
│   │   ├── episode_000001.mp4
│   │   └── ...
│   └── observation.images.rs_rgb/
│       └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl          (29 lines, one per episode)
    ├── tasks.jsonl
    ├── modality.json
    └── stats.json
```

## Key Features

✅ **Multi-episode datasets**: All datasetX folders combined into one LeRobot dataset  
✅ **Skip invalid datasets**: Automatically skips datasets missing joint_positions.npy  
✅ **Optional depth**: Use `--include-depth` to include rs_depth videos  
✅ **Progress tracking**: Shows progress bar during conversion  
✅ **Error handling**: Continues processing if individual episodes fail  

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | None | Task to process (`bin_pushing` or `pick_and_place`) |
| `--include-depth` | False | Include depth images in videos |
| `--input-root` | `datasets/training_dataset` | Input directory |
| `--output-root` | `datasets_preprocessed` | Output directory |
| `--fps` | 20.0 | Video frame rate |

