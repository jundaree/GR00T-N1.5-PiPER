#!/usr/bin/env python3
"""
Dataloader to convert raw robot datasets to LeRobot format.

Input structure (datasets folder):
    datasets/training_dataset/<task>/
        dataset1/
            joint_positions.npy  # Shape: (num_frames, 7) - joint positions
            logi_rgb/            # RGB images from logitech camera
                000000.png
                000001.png
                ...
            rs_rgb/              # RGB images from realsense camera
            rs_depth/            # Depth images from realsense (optional)
        dataset2/
            ...
        dataset3/
            ...

Output structure (LeRobot format):
    datasets_preprocessed/<task>/
        data/
            chunk-000/
                episode_000000.parquet  # dataset1
                episode_000001.parquet  # dataset2
                episode_000002.parquet  # dataset3
                ...
        videos/
            chunk-000/
                observation.images.logi_rgb/
                    episode_000000.mp4
                    episode_000001.mp4
                    ...
                observation.images.rs_rgb/
                    episode_000000.mp4
                    ...
        meta/
            info.json
            episodes.jsonl
            tasks.jsonl
            modality.json
            stats.json

Each datasetX folder is treated as one episode in the combined dataset.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import ffmpeg


class RobotDataConverter:
    """Convert raw robot datasets to LeRobot format."""

    # Task descriptions defined once for the entire class
    TASK_DESCRIPTIONS = {
        "bin_pushing": "push the red bowl to the target location between two yellow lines",
        "pick_and_place": "pick the cube and place it in the red bowl",
    }

    def __init__(
        self,
        input_root: str = "datasets/training_dataset",
        output_root: str = "datasets_preprocessed/training_dataset",
        fps: float = 20.0,
        include_depth: bool = False,
        downsample_factor: int = 3,
        lookahead: int = 0,
    ):
        """
        Initialize the converter.

        Args:
            input_root: Root directory of input datasets
            output_root: Root directory for output datasets
            fps: Frames per second for video encoding
            include_depth: Whether to include depth images
            downsample_factor: Spatial downsample factor for video scaling
            lookahead: Frame gap between observation state and action.
                       action[t] = observation.state[t + lookahead]
                       For the last `lookahead` frames, actions are set to the last observation.
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.fps = fps
        self.chunk_size = 1000  # Fixed chunk size
        self.include_depth = include_depth
        self.downsample_factor = downsample_factor
        self.video_codec_used = None  # Track which codec was successfully used
        self.lookahead = max(0, int(lookahead))

    def convert_all(self):
        """Convert all datasets in the input directory."""
        # Find all tasks (bin_pushing, pick_and_place)
        tasks = [d for d in self.input_root.iterdir() if d.is_dir()]
        
        if not tasks:
            print("No tasks found in input directory")
            return

        # Collect datasets from all tasks, splitting each task's last 5 for validation
        train_datasets = []
        val_datasets = []
        
        for task_dir in tasks:
            task_name = task_dir.name
            datasets = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("dataset")]
            datasets = sorted(datasets, key=lambda x: int(x.name.replace("dataset", "")))
            
            if not datasets:
                continue
            
            # Split this task's datasets into train and val
            if len(datasets) > 5:
                task_train = datasets[:-5]
                task_val = datasets[-5:]
            else:
                task_train = datasets
                task_val = []
            
            # Add with task name
            for dataset in task_train:
                train_datasets.append((dataset, task_name))
            for dataset in task_val:
                val_datasets.append((dataset, task_name))
            
            print(f"Found {len(datasets)} episodes in {task_name} (train: {len(task_train)}, val: {len(task_val)})")

        if not train_datasets and not val_datasets:
            print("No datasets found in any task")
            return

        print(f"\n{'='*60}")
        print(f"Merging all tasks with {len(train_datasets) + len(val_datasets)} total episodes")
        print(f"Train: {len(train_datasets)} episodes, Val: {len(val_datasets)} episodes")
        print(f"{'='*60}")

        # Process training set
        if train_datasets:
            train_output_dir = Path(str(self.output_root).replace("training_dataset", "train_dataset"))
            try:
                self.convert_merged_task(train_datasets, train_output_dir, "merged")
                print(f"✓ Successfully converted training set with {len(train_datasets)} episodes")
            except Exception as e:
                print(f"✗ Error converting training set: {e}")
                import traceback
                traceback.print_exc()

        # Process validation set
        if val_datasets:
            val_output_dir = Path(str(self.output_root).replace("training_dataset", "val_dataset"))
            try:
                self.convert_merged_task(val_datasets, val_output_dir, "merged")
                print(f"✓ Successfully converted validation set with {len(val_datasets)} episodes")
            except Exception as e:
                print(f"✗ Error converting validation set: {e}")
                import traceback
                traceback.print_exc()

    def convert_merged_task(self, dataset_task_pairs: List[tuple], output_dir: Path, merged_name: str):
        """
        Convert multiple datasets from different tasks into a single merged LeRobot format dataset.

        Args:
            dataset_task_pairs: List of (dataset_dir, task_name) tuples
            output_dir: Output directory for combined dataset
            merged_name: Name for the merged dataset (e.g., 'merged')
        """
        # Create output directory structure
        data_dir = output_dir / "data" / "chunk-000"
        video_dir = output_dir / "videos" / "chunk-000"
        meta_dir = output_dir / "meta"

        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        all_joint_positions = []
        episode_lengths = []
        episode_tasks = []  # Track which task each episode belongs to
        image_modalities_available = set()
        episode_counter = 0

        print(f"\nProcessing {len(dataset_task_pairs)} episodes for merged dataset...")

        # Process each dataset as an episode
        for dataset_dir, task_name in tqdm(dataset_task_pairs, desc="Converting episodes"):
            dataset_name = dataset_dir.name
            
            # Check if joint_positions.npy exists
            joint_positions_path = dataset_dir / "joint_positions.npy"
            if not joint_positions_path.exists():
                print(f"\n  Warning: Skipping {dataset_name} - joint_positions.npy not found")
                continue
            
            # Load joint positions (actions/states)
            try:
                joint_positions = np.load(joint_positions_path)
            except Exception as e:
                print(f"\n  Warning: Skipping {dataset_name} - error loading joint_positions.npy: {e}")
                continue
                
            num_frames = len(joint_positions)

            # Apply temporal downsampling to joints and images by keeping every Nth frame
            down_factor = max(1, self.downsample_factor)
            if down_factor > 1:
                indices = np.arange(0, num_frames, down_factor)
                joint_positions = joint_positions[indices]
                num_frames = len(indices)

            # Find available image modalities dynamically
            image_modalities = {}
            for item in dataset_dir.iterdir():
                if not item.is_dir():
                    continue
                    
                folder_name = item.name.lower()
                
                # Check if folder contains 'rgb' or 'depth'
                if 'rgb' in folder_name or (self.include_depth and 'depth' in folder_name):
                    images = sorted(list(item.glob("*.png")))
                    if images and len(images) > 0:
                        # Adjust num_frames if images are fewer
                        if len(images) < num_frames:
                            print(f"\n  Note: {dataset_name}/{item.name} has only {len(images)} images, trimming from {num_frames} frames")
                            num_frames = len(images)
                            joint_positions = joint_positions[:num_frames]
                            # Update indices accordingly when trimming
                            if down_factor > 1:
                                indices = np.arange(0, num_frames)
                        # Use only downsampled images matching joints length
                        if down_factor > 1:
                            image_modalities[item.name] = [images[i] for i in range(0, len(images[:num_frames]), down_factor)][:num_frames]
                        else:
                            image_modalities[item.name] = images[:num_frames]
                        image_modalities_available.add(item.name)
                        if len(images) > num_frames:
                            print(f"\n  Note: {dataset_name}/{item.name} has {len(images)} images, using first {num_frames}")

            if not image_modalities:
                print(f"\n  Warning: No valid image modalities found for {dataset_name}, skipping video creation")

            # Update episode tracking with potentially trimmed data
            all_joint_positions.append(joint_positions)
            episode_lengths.append(num_frames)
            episode_tasks.append(task_name)

            # Create episode data with the specific task
            episode_data = self._create_episode_data(
                joint_positions, episode_counter, task_name
            )

            # Save parquet file
            parquet_path = data_dir / f"episode_{episode_counter:06d}.parquet"
            episode_df = pd.DataFrame(episode_data)
            episode_df.to_parquet(parquet_path, index=False)

            # Create videos for each image modality
            for modality_name, images in image_modalities.items():
                video_modality_dir = video_dir / f"observation.images.{modality_name}"
                video_modality_dir.mkdir(parents=True, exist_ok=True)

                video_path = video_modality_dir / f"episode_{episode_counter:06d}.mp4"
                if not self._create_video(images, video_path):
                    print(f"\n  Warning: Video creation failed for {modality_name} in episode {episode_counter}")
                    break  # Skip remaining modalities if one fails
            
            episode_counter += 1

        if episode_counter == 0:
            raise ValueError(f"No valid episodes found")

        # Count episodes per task
        task_counts = {}
        for task in episode_tasks:
            task_counts[task] = task_counts.get(task, 0) + 1

        print(f"\n✓ Processed {episode_counter} episodes")
        print(f"  Task breakdown: {task_counts}")
        print(f"  Total frames: {sum(episode_lengths)}")
        print(f"  Image modalities: {sorted(image_modalities_available)}")

        # Combine all joint positions for statistics
        all_joint_positions_combined = np.vstack(all_joint_positions)

        # Create metadata files with merged tasks
        self._create_merged_metadata(
            meta_dir,
            num_episodes=episode_counter,
            num_frames=sum(episode_lengths),
            episode_tasks=episode_tasks,
            image_modalities=sorted(image_modalities_available),
            joint_positions=all_joint_positions_combined,
            episode_lengths=episode_lengths,
        )
        print(f"  Created metadata in {meta_dir}")

    def convert_task(self, dataset_dirs: List[Path], output_dir: Path, task_name: str):
        """
        Convert multiple datasets (episodes) into a single LeRobot format dataset.

        Args:
            dataset_dirs: List of dataset directories (each becomes an episode)
            output_dir: Output directory for combined dataset
            task_name: Name of the task (e.g., 'bin_pushing', 'pick_and_place')
        """
        # Create output directory structure
        data_dir = output_dir / "data" / "chunk-000"
        video_dir = output_dir / "videos" / "chunk-000"
        meta_dir = output_dir / "meta"

        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        all_joint_positions = []
        episode_lengths = []
        image_modalities_available = set()
        episode_counter = 0

        print(f"\nProcessing {len(dataset_dirs)} episodes for {task_name}...")

        # Process each dataset as an episode
        for dataset_dir in tqdm(dataset_dirs, desc="Converting episodes"):
            dataset_name = dataset_dir.name
            
            # Check if joint_positions.npy exists
            joint_positions_path = dataset_dir / "joint_positions.npy"
            if not joint_positions_path.exists():
                print(f"\n  Warning: Skipping {dataset_name} - joint_positions.npy not found")
                continue
            
            # Load joint positions (actions/states)
            try:
                joint_positions = np.load(joint_positions_path)
            except Exception as e:
                print(f"\n  Warning: Skipping {dataset_name} - error loading joint_positions.npy: {e}")
                continue
                
            num_frames = len(joint_positions)

            # Find available image modalities dynamically
            image_modalities = {}
            for item in dataset_dir.iterdir():
                if not item.is_dir():
                    continue
                    
                folder_name = item.name.lower()
                
                # Check if folder contains 'rgb' or 'depth'
                if 'rgb' in folder_name or (self.include_depth and 'depth' in folder_name):
                    images = sorted(list(item.glob("*.png")))
                    if images and len(images) > 0:
                        # Adjust num_frames if images are fewer
                        if len(images) < num_frames:
                            print(f"\n  Note: {dataset_name}/{item.name} has only {len(images)} images, trimming from {num_frames} frames")
                            num_frames = len(images)
                            joint_positions = joint_positions[:num_frames]
                        # Use only the first num_frames images (handle case where there's an extra frame)
                        image_modalities[item.name] = images[:num_frames]
                        image_modalities_available.add(item.name)
                        if len(images) > num_frames:
                            print(f"\n  Note: {dataset_name}/{item.name} has {len(images)} images, using first {num_frames}")

            if not image_modalities:
                print(f"\n  Warning: No valid image modalities found for {dataset_name}, skipping video creation")

            # Update episode tracking with potentially trimmed data
            all_joint_positions.append(joint_positions)
            episode_lengths.append(num_frames)

            # Create episode data
            episode_data = self._create_episode_data(
                joint_positions, episode_counter, task_name
            )

            # Save parquet file
            parquet_path = data_dir / f"episode_{episode_counter:06d}.parquet"
            episode_df = pd.DataFrame(episode_data)
            episode_df.to_parquet(parquet_path, index=False)

            # Create videos for each image modality
            for modality_name, images in image_modalities.items():
                video_modality_dir = video_dir / f"observation.images.{modality_name}"
                video_modality_dir.mkdir(parents=True, exist_ok=True)

                video_path = video_modality_dir / f"episode_{episode_counter:06d}.mp4"
                print(f"\n  Processing video for {modality_name}:")
                if not self._create_video(images, video_path):
                    print(f"\n  Warning: Video creation failed for {modality_name} in episode {episode_counter}")
                    break  # Skip remaining modalities if one fails
            
            episode_counter += 1

        if episode_counter == 0:
            raise ValueError(f"No valid episodes found in {task_name}")

        print(f"\n✓ Processed {episode_counter} episodes")
        print(f"  Total frames: {sum(episode_lengths)}")
        print(f"  Image modalities: {sorted(image_modalities_available)}")

        # Combine all joint positions for statistics
        all_joint_positions_combined = np.vstack(all_joint_positions)

        # Create metadata files
        self._create_metadata(
            meta_dir,
            num_episodes=episode_counter,
            num_frames=sum(episode_lengths),
            task_name=task_name,
            image_modalities=sorted(image_modalities_available),
            joint_positions=all_joint_positions_combined,
            episode_lengths=episode_lengths,
        )
        print(f"  Created metadata in {meta_dir}")

    def convert_dataset(self, input_dir: Path, output_dir: Path, task_name: str):
        """
        Convert a single dataset to LeRobot format (legacy method).
        Use convert_task for combining multiple episodes.

        Args:
            input_dir: Input dataset directory
            output_dir: Output dataset directory
            task_name: Name of the task (e.g., 'bin_pushing', 'pick_and_place')
        """
        self.convert_task([input_dir], output_dir, task_name)

    def _create_episode_data(
        self, joint_positions: np.ndarray, episode_index: int, task_name: str
    ) -> Dict:
        """
        Create episode data in LeRobot format.

        Args:
            joint_positions: Joint position array (num_frames, 7)
            episode_index: Episode index
            task_name: Task name

        Returns:
            Dictionary with episode data
        """
        num_frames = len(joint_positions)

        # Get task description from class attribute
        task_description = self.TASK_DESCRIPTIONS.get(task_name, task_name)

        # Prepare action with lookahead shift: action[t] = obs[t + lookahead],
        # and fill the last `lookahead` entries with the last observation.
        if self.lookahead <= 0:
            actions = joint_positions
        else:
            if num_frames == 0:
                actions = joint_positions
            else:
                shifted = joint_positions[self.lookahead:]  # obs starting at t+lookahead
                tail_fill = np.repeat(joint_positions[-1][None, :], self.lookahead, axis=0)
                actions = np.vstack([shifted, tail_fill])

        episode_data = {
            "observation.state": joint_positions.tolist(),
            "action": actions.tolist(),
            "timestamp": (np.arange(num_frames) / self.fps).tolist(),
            "annotation.human.action.task_description": [0] * num_frames,
            "task_index": [0] * num_frames,
            "annotation.human.validity": [1] * num_frames,
            "episode_index": [episode_index] * num_frames,
            "index": list(range(num_frames)),
            "next.reward": [0.0] * num_frames,
            "next.done": [False] * (num_frames - 1) + [True],
        }

        return episode_data

    def _create_video(self, image_paths: List[Path], output_path: Path) -> bool:
        """
        Create MP4 video from image sequence using ffmpeg-python.
        
        Args:
            image_paths: List of image file paths
            output_path: Output video path
            
        Returns:
            bool: True if video was created successfully, False otherwise
        """
        if not image_paths:
            print(f"  Warning: No images provided for {output_path.name}")
            return False

        # Ensure directory exists before encoding
        output_path.parent.mkdir(parents=True, exist_ok=True)
        

        # Get the directory of the first image
        input_dir = image_paths[0].parent
        
        # Check if output file already exists (e.g., from a failed run)
        if output_path.exists():
            print(f"  Removing existing video: {output_path.name}")
            os.remove(output_path)

        # The image pattern for ffmpeg input (assumes 6-digit zero-padding)
        input_pattern = str(input_dir / "%06d.png")
        

        try:
            # Construct the FFmpeg command using ffmpeg-python
            (
                ffmpeg
                .input(input_pattern, framerate = self.fps * self.downsample_factor,start_number=0)
                .filter(
                    'select',
                    f"not(mod(n,{max(1, self.downsample_factor)}))",
                )  # drop frames by factor (keep every Nth)
                .filter('setpts', f'N/{self.fps}/TB')
                .output(
                    str(output_path),
                    vcodec='libx264',      # Use the installed libx264 encoder
                    pix_fmt='yuv420p',     # Standard pixel format for H.264
                    movflags='faststart',  # Optimization for web streaming
                    r=self.fps,            # Set output frame rate
                    loglevel='error'       # Only show errors
                )
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            
            # Update the codec used for metadata tracking
            if self.video_codec_used is None:
                self.video_codec_used = "libx264"
            
            return True
                
        except ffmpeg.Error as e:
            # If FFmpeg fails, print the error output from the subprocess
            print(f"  ✗ FFmpeg Error for {output_path.name}:")
            # Decode the error stream for debugging
            if e.stderr:
                print(e.stderr.decode('utf8'))
            
            # Clean up partial output file if one was created
            if output_path.exists():
                os.remove(output_path)
            
            # Return False to indicate failure instead of raising
            return False
        except FileNotFoundError:
            # This handles if the 'ffmpeg' binary itself is not in the PATH
            print("  ✗ FFmpeg executable not found. Ensure it is installed and in the system PATH.")
            return False

    def _create_metadata(
        self,
        meta_dir: Path,
        num_episodes: int,
        num_frames: int,
        task_name: str,
        image_modalities: List[str],
        joint_positions: np.ndarray,
        episode_lengths: Optional[List[int]] = None,
    ):
        """
        Create metadata files for LeRobot format.

        Args:
            meta_dir: Metadata directory
            num_episodes: Number of episodes
            num_frames: Total number of frames
            task_name: Task name
            image_modalities: List of image modality names
            joint_positions: Joint positions for computing statistics
            episode_lengths: List of episode lengths
        """
        if episode_lengths is None:
            episode_lengths = [num_frames]
        
        # Get task description from class attribute
        task_description = self.TASK_DESCRIPTIONS.get(task_name, task_name)

        # Prepare tasks.jsonl
        tasks_data = []
        for idx, task in enumerate(set(task_name)):
            description = self.TASK_DESCRIPTIONS.get(task, task)
            tasks_data.append({"task_index": idx, "task": description})
        
        # Write tasks.jsonl
        tasks_path = meta_dir / "tasks.jsonl"
        with open(tasks_path, "w") as f:
            for task_info in tasks_data:
                f.write(json.dumps(task_info) + "\n")

        # Prepare episodes.jsonl
        episodes_data = []
        for ep_idx in range(num_episodes):
            episode_info = {
                "episode_index": ep_idx,
                "tasks": [task_description],
                "length": episode_lengths[ep_idx],
            }
            episodes_data.append(episode_info)
        
        # Write episodes.jsonl
        episodes_path = meta_dir / "episodes.jsonl"
        with open(episodes_path, "w") as f:
            for episode_info in episodes_data:
                f.write(json.dumps(episode_info) + "\n")

        # Create modality.json
        modality_config = {
            "state": {
                "single_arm": {"start": 0, "end": 6},
                "gripper": {"start": 6, "end": 7}
            },
            "action": {
                "single_arm": {"start": 0, "end": 6, "absolute": True},
                "gripper": {"start": 6, "end": 7, "absolute": True}
            },
        }

        # Add video modalities
        if image_modalities:
            modality_config["video"] = {}
            for modality_name in image_modalities:
                modality_config["video"][modality_name] = {
                    "original_key": f"observation.images.{modality_name}"
                }

        # Add annotation modalities
        modality_config["annotation"] = {
            "human.action.task_description": {},
            "human.validity": {},
        }

        modality_path = meta_dir / "modality.json"
        with open(modality_path, "w") as f:
            json.dump(modality_config, f, indent=4)

        # Create stats.json
        stats = self._compute_statistics(joint_positions)
        stats_path = meta_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

        # Create info.json
        info = self._create_info_json(
            num_episodes, num_frames, image_modalities, task_name
        )
        info_path = meta_dir / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    def _compute_statistics(self, joint_positions: np.ndarray) -> Dict:
        """
        Compute statistics for normalization.

        Args:
            joint_positions: Joint positions array

        Returns:
            Statistics dictionary
        """
        stats = {
            "observation.state": {
                "mean": joint_positions.mean(axis=0).tolist(),
                "std": joint_positions.std(axis=0).tolist(),
                "min": joint_positions.min(axis=0).tolist(),
                "max": joint_positions.max(axis=0).tolist(),
            },
            "action": {
                "mean": joint_positions.mean(axis=0).tolist(),
                "std": joint_positions.std(axis=0).tolist(),
                "min": joint_positions.min(axis=0).tolist(),
                "max": joint_positions.max(axis=0).tolist(),
            },
        }
        return stats

    def _create_info_json(
        self,
        num_episodes: int,
        num_frames: int,
        image_modalities: List[str],
        task_name: str,
    ) -> Dict:
        """
        Create info.json metadata.

        Args:
            num_episodes: Number of episodes
            num_frames: Total number of frames
            image_modalities: List of image modality names
            task_name: Task name

        Returns:
            Info dictionary
        """
        info = {
            "codebase_version": "v2.0",
            "robot_type": "piper",
            "total_episodes": num_episodes,
            "total_frames": num_frames,
            "total_tasks": 2,
            "total_videos": len(image_modalities) * num_episodes,
            "total_chunks": 0,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": "0:100"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
                },
            },
        }

        # Add video features for each modality
        for modality_name in image_modalities:
            # Determine if this is a depth modality
            is_depth = 'depth' in modality_name.lower()
            
            # Try to get image dimensions from first image
            info["features"][f"observation.images.{modality_name}"] = {
                "dtype": "video",
                "shape": [480, 640, 3],  # Default, will be updated if we can read image
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": self.fps,
                    "video.codec": self.video_codec_used if self.video_codec_used else "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": is_depth,
                    "has_audio": False,
                },
            }

        return info

    def _create_merged_metadata(
        self,
        meta_dir: Path,
        num_episodes: int,
        num_frames: int,
        episode_tasks: List[str],
        image_modalities: List[str],
        joint_positions: np.ndarray,
        episode_lengths: Optional[List[int]] = None,
    ):
        """
        Create metadata files for merged LeRobot format dataset.

        Args:
            meta_dir: Metadata directory
            num_episodes: Number of episodes
            num_frames: Total number of frames
            episode_tasks: List of task names for each episode
            image_modalities: List of image modality names
            joint_positions: Joint positions for computing statistics
            episode_lengths: List of episode lengths
        """
        if episode_lengths is None:
            episode_lengths = [num_frames]
        
        # Get unique tasks and create task index mapping
        unique_tasks = sorted(set(episode_tasks))
        task_to_index = {task: idx for idx, task in enumerate(unique_tasks)}

        # Create tasks.jsonl with all unique tasks
        tasks_path = meta_dir / "tasks.jsonl"
        with open(tasks_path, "w") as f:
            for task_name in unique_tasks:
                task_description = self.TASK_DESCRIPTIONS.get(task_name, task_name)
                f.write(json.dumps({"task_index": task_to_index[task_name], "task": task_description}) + "\n")

        # Create episodes.jsonl
        episodes_path = meta_dir / "episodes.jsonl"
        with open(episodes_path, "w") as f:
            for ep_idx in range(num_episodes):
                task_name = episode_tasks[ep_idx]
                task_description = self.TASK_DESCRIPTIONS.get(task_name, task_name)
                episode_info = {
                    "episode_index": ep_idx,
                    "tasks": [task_description],
                    "length": episode_lengths[ep_idx],
                }
                f.write(json.dumps(episode_info) + "\n")

        # Create modality.json
        modality_config = {
            "state": {
                "single_arm": {"start": 0, "end": 6},
                "gripper": {"start": 6, "end": 7}
            },
            "action": {
                "single_arm": {"start": 0, "end": 6, "absolute": True},
                "gripper": {"start": 6, "end": 7, "absolute": True}
            },
        }

        # Add video modalities
        if image_modalities:
            modality_config["video"] = {}
            for modality_name in image_modalities:
                modality_config["video"][modality_name] = {
                    "original_key": f"observation.images.{modality_name}"
                }

        # Add annotation modalities
        modality_config["annotation"] = {
            "human.action.task_description": {},
            "human.validity": {},
        }

        modality_path = meta_dir / "modality.json"
        with open(modality_path, "w") as f:
            json.dump(modality_config, f, indent=4)

        # Create stats.json
        stats = self._compute_statistics(joint_positions)
        stats_path = meta_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

        # Create info.json
        info = self._create_info_json(
            num_episodes, num_frames, image_modalities, "merged"
        )
        info["total_tasks"] = len(unique_tasks)
        info_path = meta_dir / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

def main():
    """Main entry point for the dataloader."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert robot datasets to LeRobot format"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="datasets/training_dataset",
        help="Root directory of input datasets",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="datasets_preprocessed",
        help="Root directory for output datasets",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0, help="Frames per second for video encoding"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to process (e.g., 'bin_pushing', 'pick_and_place'). If not specified, merges all tasks.",
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Include depth images in the conversion",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=3,
        help="Spatial downsample factor for video scaling (e.g., 3 -> width/3, height/3).",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=0,
        help="Frame gap between observation state and action. For the last 'lookahead' frames, actions are set to the last observation.",
    )

    args = parser.parse_args()

    converter = RobotDataConverter(
        input_root=args.input_root,
        output_root=args.output_root,
        fps=args.fps,
        include_depth=args.include_depth,
        downsample_factor=args.downsample_factor,
        lookahead=args.lookahead,
    )

    if args.task:
        # Process a specific task only (not merged)
        task_dir = Path(args.input_root) / args.task
        datasets = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("dataset")]
        if not datasets:
            print(f"No datasets found in {args.task}")
            return
        datasets = sorted(datasets, key=lambda x: int(x.name.replace("dataset", "")))
        
        # Split into train and val
        if len(datasets) > 5:
            train_datasets = datasets[:-5]
            val_datasets = datasets[-5:]
        else:
            train_datasets = datasets
            val_datasets = []
        
        # Process training set
        if train_datasets:
            train_output_dir = Path(str(args.output_root).replace("training_dataset", "train_dataset")) / args.task
            print(f"Processing task '{args.task}' training set with {len(train_datasets)} episodes")
            converter.convert_task(train_datasets, train_output_dir, args.task)
        
        # Process validation set
        if val_datasets:
            val_output_dir = Path(str(args.output_root).replace("training_dataset", "val_dataset")) / args.task
            print(f"Processing task '{args.task}' validation set with {len(val_datasets)} episodes")
            converter.convert_task(val_datasets, val_output_dir, args.task)
    else:
        # Process all tasks (merged)
        converter.convert_all()

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)


if __name__ == "__main__":
    main()
