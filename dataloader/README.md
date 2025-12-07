# Robot Dataset to LeRobot Format Converter

This dataloader converts raw robot datasets into LeRobot v2.0 compatible format.

## Overview

The dataloader reads data from the `datasets/training_dataset/` folder and converts it to LeRobot format, outputting to `datasets_preprocessed/`. **Each datasetX folder is treated as one episode**, and all episodes from a task are combined into a single LeRobot dataset.

### Input Format

```
datasets/training_dataset/<task>/
    dataset1/
        joint_positions.npy     # Shape: (num_frames, 7) - joint positions
        logi_rgb/               # RGB images from logitech camera
            000000.png
            000001.png
            ...
        rs_rgb/                 # RGB images from realsense camera
            000000.png
            000001.png
            ...
        rs_depth/               # Depth images from realsense (optional)
            000000.png
            000001.png
            ...
    dataset2/
        ...
    dataset3/
        ...
```

### Output Format (LeRobot v2.0)

```
datasets_preprocessed/<task>/
    data/
        chunk-000/
            episode_000000.parquet    # dataset1
            episode_000001.parquet    # dataset2
            episode_000002.parquet    # dataset3
            ...
    videos/
        chunk-000/
            observation.images.logi_rgb/
                episode_000000.mp4
                episode_000001.mp4
                ...
            observation.images.rs_rgb/
                episode_000000.mp4
                episode_000001.mp4
                ...
    meta/
        info.json           # Dataset metadata
        episodes.jsonl      # Episode information (one line per episode)
        tasks.jsonl         # Task descriptions
        modality.json       # Modality configuration
        stats.json          # Statistics for normalization
```

## Usage

### Convert All Datasets

To convert all datasets in the `datasets/training_dataset/` folder:

```bash
python dataloader/dataloader.py
```

### Convert Specific Task

To convert all datasets for a specific task (e.g., `bin_pushing` or `pick_and_place`) into a single multi-episode dataset:

```bash
python dataloader/dataloader.py --task bin_pushing
```

This will combine all `dataset1`, `dataset2`, `dataset3`, etc. into a single LeRobot dataset with multiple episodes.

### Command Line Options

```
--input-root      Root directory of input datasets (default: datasets/training_dataset)
--output-root     Root directory for output datasets (default: datasets_preprocessed)
--fps             Frames per second for video encoding (default: 20.0)
--chunk-size      Number of episodes per chunk (default: 1000)
--task            Specific task to process (e.g., 'bin_pushing', 'pick_and_place')
--include-depth   Include depth images in the conversion (default: False)
```

## Examples

### Example 1: Convert all bin_pushing datasets into one multi-episode dataset
```bash
python dataloader/dataloader.py --task bin_pushing
```

### Example 2: Convert pick_and_place datasets with depth images
```bash
python dataloader/dataloader.py --task pick_and_place --include-depth
```

### Example 3: Convert all tasks with custom FPS
```bash
python dataloader/dataloader.py --fps 30.0
```

### Example 4: Convert to a different output directory
```bash
python dataloader/dataloader.py --output-root my_preprocessed_data --task bin_pushing
```

### Example 5: Convert all tasks at once
```bash
python dataloader/dataloader.py
```

## Data Format Details

### Parquet Files
Each episode is stored as a parquet file containing:
- `observation.state`: Joint positions (7-DOF)
- `action`: Action values (same as state for this dataset)
- `timestamp`: Frame timestamps
- `annotation.human.action.task_description`: Task index
- `task_index`: Task index
- `annotation.human.validity`: Validity flag (1 = valid)
- `episode_index`: Episode index
- `index`: Frame index
- `next.reward`: Reward (0.0 for all frames)
- `next.done`: Episode completion flag (True for last frame)

### Videos
Images are encoded as MP4 videos with:
- Codec: mp4v
- FPS: 20.0 (configurable)
- Format: YUV420P

### Metadata Files

#### info.json
Contains dataset metadata including:
- Robot type
- Total episodes, frames, tasks, videos
- FPS and chunk configuration
- Feature definitions

#### modality.json
Defines the structure of state and action arrays:
- `state.arm`: indices 0-7 (joint positions)
- `action.arm`: indices 0-7 (joint positions)
- `video`: mapping of video modalities
- `annotation`: annotation fields

#### episodes.jsonl
One line per episode with:
- Episode index
- Task descriptions
- Episode length (number of frames)

#### tasks.jsonl
One line per task with:
- Task index
- Task description

#### stats.json
Statistics for normalization:
- Mean, std, min, max for observation.state
- Mean, std, min, max for action

## Dependencies

Required Python packages:
- numpy
- pandas
- opencv-python (cv2)
- Pillow
- tqdm

Install with:
```bash
pip install numpy pandas opencv-python Pillow tqdm pyarrow
```

## Notes

- **Each datasetX folder is treated as one episode** in the combined LeRobot dataset
- All episodes from the same task are combined into a single LeRobot dataset
- Joint positions are used for both state and action
- All frames are marked as valid
- Task descriptions are automatically generated based on the task name:
  - `bin_pushing`: "push the bin to the target location"
  - `pick_and_place`: "pick the object and place it at the target location"
- Video encoding may take some time depending on the number of frames and image resolution
- Depth images are excluded by default. Use `--include-depth` to include them
- Statistics (mean, std, min, max) are computed across all episodes for normalization

## Troubleshooting

### Missing images
If some image folders are missing, the dataloader will skip them and only process available modalities.

### Image count mismatch
If the number of images doesn't match the number of frames in joint_positions.npy, a warning will be displayed.

### Video encoding issues
Ensure OpenCV is properly installed with video codec support:
```bash
pip install opencv-python-headless
```
