# GR00T-N1.5-PiPER

This project preprocesses MuJoCo and real-robot datasets and trains a policy following the **GR00T-N1.5** architecture. The system takes language instructions, two RGB image observations and robot state as input and outputs an **action chunk** comprising robot actions over the next 16 time frames.

![System Diagram](/media/system_overview.png)

To run this project, you need to set up two docker containers: **server container** and **client container**. The server container can be constructed from a Dockerfile or by pulling a docker image from **Docker Hub** ([jundaree/gr00t-piper](https://hub.docker.com/r/jundaree/gr00t-piper)). This server container takes policy input from the client container, runs the GR00T-N1.5 policy, and returns action chunks to the client container. The client container passes language input from the user to the server container along with vision input. The client container receives action chunks from the server container and actuates **the AgileX PiPER robot arm in MuJoCo or real environments**. Separate client containers are needed for MuJoCo and real robot settings, while the server container can be shared. 

Unfortunately, this work is focused on building the **server container**, not the client container, since building the client container is highly dependent on the robot setup and personal usage. However, inference codes are uploaded in the client_container_scripts/ folder for reference.

## Setup

### Server Container

#### Option 1: Pull from Docker Hub (Recommended)

Pull the pre-built Docker image from **Docker Hub**:

```bash
docker pull jundaree/gr00t-piper
```

Run the container with GPU support and dataset volume mounted:

```bash
docker run -v /path/to/your/datasets:/workspace/datasets \
--name gr00t-piper \
--gpus all \
--shm-size="16g" \
-it jundaree/gr00t-piper
```

#### Option 2: Build from Dockerfile

Build the Docker image locally:

```bash
docker build -t gr00t-piper .
```

Run the container:

```bash
docker run -v /path/to/your/datasets:/workspace/datasets \
--name gr00t-piper \
--gpus all \
--shm-size="16g" \
-it gr00t-piper
```

### Client Container

For the client container (MuJoCo or real robot), install required dependencies:

```bash
pip install zmq msgpack
```

## Usage

### 1. Data Preprocessing

Convert your raw dataset to **LeRobot format** with downsampling and lookahead:

**Preprocessing Parameters:**

- `--downsample-factor`: Temporally downsamples video frames by keeping every Nth frame (e.g., `--downsample-factor 6` keeps every 6th frame). This **reduces the total number of frames** in the dataset without changing the spatial resolution (width, height) of individual frames. Higher values result in fewer frames and faster training but may lose temporal detail.

- `--lookahead`: **Shifts action labels forward** by N timesteps to account for action-observation delays. This is needed when your dataset records action values that match the observed state at the same timestep. Use this parameter if your dataset has synchronization between actions and observations. **Exclude this** if there are already discrepancies between action values and state values in your dataset (i.e., actions are already properly offset from observations).

- `--fps`: Sets the frame rate (frames per second) of the encoded output videos. This determines the temporal resolution of the preprocessed video files.

**For MuJoCo dataset:**

```bash
python dataloader/dataloader_lookahead.py \
--input-root datasets/training_dataset_mujoco \
--output-root datasets_preprocessed/training_dataset_mujoco_downsampled_6_lookahead \
--downsample-factor 6 --lookahead 3 --fps 10.0
```

**For Real Robot dataset:**

```bash
python dataloader/dataloader_lookahead.py \
--input-root datasets/training_dataset \
--output-root datasets_preprocessed/training_dataset_realrobot_downsampled_6_lookahead \
--downsample-factor 6 --lookahead 3 --fps 10.0
```

### 2. Training

**Train on MuJoCo dataset:**

```bash
python scripts/gr00t_finetune.py \
--dataset-path ./datasets_preprocessed/training_dataset_mujoco_downsampled_6_lookahead \
--output_dir outputs/gr00t_finetune_mujoco_lora_downsampled_6_lookahead \
--max_steps 5000 \
--save_steps 500 \
--data_config piper_mujoco \
--lora_rank 64 \
--lora_alpha 64 \
--batch_size 32 \
--num-gpus 1
```

**Train on Real Robot dataset:**

```bash
python scripts/gr00t_finetune.py \
--dataset-path ./datasets_preprocessed/training_dataset_realrobot_downsampled_6_lookahead \
--output_dir outputs/gr00t_finetune_realrobot_lora_downsampled_6_lookahead \
--max_steps 4000 \
--save_steps 500 \
--data_config piper_real \
--lora_rank 64 \
--lora_alpha 64 \
--batch_size 32 \
--num-gpus 1
```

### 3. Inference Service

Start the inference server with a trained checkpoint:

```bash
python scripts/inference_service.py --server \
--model_path outputs/gr00t_finetune_mujoco_lora_downsampled_6_lookahead/checkpoint-5000 \
--embodiment-tag new_embodiment \
--data-config piper_mujoco \
--denoising-steps 4
```

### 4. Evaluation

Evaluate the trained policy:

```bash
python scripts/eval_policy.py \
--embodiment_tag new_embodiment \
--model_path outputs/gr00t_finetune_mujoco_lora_downsampled_6_lookahead/checkpoint-5000 \
--data_config piper_mujoco \
--dataset_path datasets_preprocessed/val_dataset_mujoco \
--modality_keys single_arm gripper
```

Plot learning curves:

**For MuJoCo:**

```bash
python plot_learning_curve.py \
--checkpoint-dir outputs/gr00t_finetune_mujoco_lora_downsampled_6_lookahead \
--mse-results mse_results.txt \
--output /workspace/eval_results/learning_curve_mujoco.png
```

**For Real Robot:**

```bash
python plot_learning_curve.py \
--checkpoint-dir outputs/gr00t_finetune_realrobot_lora_downsampled_6_lookahead \
--mse-results mse_results_realrobot.txt \
--output /workspace/eval_results/learning_curve_realrobot.png
```

### 5. Client-Side Inference

**MuJoCo Simulation:**

```bash
python inference_mujoco.py -l "pick the cube and place it in the red bowl"
```

**Real Robot:**

```bash
python inference_piper_real.py -l "push the red bowl to the target location between two yellow lines"
```

## Inputs
Since the original GR00T-N1.5 model has been pretrained without depth images, depth images are excluded in this project.
- Vision
  - MuJoCo
    - side_cam_rgb: 640x480
    - top_cam_rgb: 640x480
  - Real PiPER
    - logi_rgb: 640x480
    - rs_rgb: 640x480
- Robot State
  - observation.state: 6-DoF joint positions + 1-DoF gripper
- Language
  - task instruction text (e.g., "pick the cube and place it in the red bowl" or "push the red bowl to the target location between two yellow lines")


## Model Architecture 
![Model Architecture Diagram](/media/architecture_overview.png)

- **Vision Encoder**: encodes image streams into visual tokens.
- **Text Tokenizer + VLM** (Eagle-2, frozen): converts the instruction into language tokens and provides fused visual-language context.
- **State Encoder**: embeds robot state q_t.
- **Action Encoder**: embeds a noised action trajectory (a_t … a_{t+H-1}) which are later denoised in DiT Blocks.
- **DiT Blocks**: alternating self-attention and cross-attention over state/action tokens, attending to VLM latent features.
- **Action Decoder**: predicts motor actions over a horizon H=16; iteration over 4 steps at inference.

## Objective and Loss
- **Flow Matching Loss**: DiT blocks predicts added noise on the input noised actions. These noises are learned via flow matching loss.
- **Teacher Forcing Loss**: Unlike rollout loss, this model forces the robot to take ground-truth actions while training. This stabilizes action sequence prediction and speeds up convergence.

## Data Format
- Episodes are preprocessed to **LeRobot format** since it is supported by GR00T-N1.5:
  - data: parquet per episode with observation.state, action, and timestamps.
  - videos: mp4 per modality.
  - meta: info.json, episodes.jsonl, tasks.jsonl, modality.json, stats.json.
- Temporal downsampling and lookahead are configurable in the dataloader.


