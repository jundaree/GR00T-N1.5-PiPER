# MP4 Overview (Junhyun Kim)

This project preprocesses MuJoCo and real-robot data and trains a policy following the GR00T-N1.5 architecture. The system takes language instructions, two RGB images observation and robot state as input and outputs an action chunk comprising robot actions over the next 16 time frames.

## Inputs
Since the original GR00T-N1.5 model has been pretrained without depth images, depth images are excluded in this project.
- Vision
  - MuJoCo
    - side_cam_rgb: 640x480
    - top_cam_rgb: 640x480
  - Real PiPER
    - logi_rgb: RGB frames from a Logitech camera
    - rs_rgb: RGB frames from a RealSense camera
- Robot State
  - observation.state: 7-DoF joint positions
- Language
  - task instruction text (e.g., “pick the cube and place it in the red bowl” or “pick the cube and place it in the red bowl”)


## Model Architecture 
![Model Architecture Diagram](/media/diagram.png)

- Vision Encoder: encodes image streams into visual tokens.
- Text Tokenizer + VLM (Eagle-2, frozen): converts the instruction into language tokens and provides fused visual-language context.
- State Encoder: embeds robot state q_t.
- Action Encoder: embeds a noised action trajectory (a_t … a_{t+H-1}) which are later denoised in DiT Blocks.
- DiT Blocks (repeated N times): alternating self-attention and cross-attention over state/action tokens, attending to VLM latent features.
- Action Decoder: predicts motor actions over a horizon H=16; iteration over 4 steps at inference.

## Objective and Loss
- Flow Matching Loss: DiT blocks predicts added noise on the input noised actions. These noises are learned via flow matching loss.
- Teacher Forcing Loss: Unlike rollout loss, this model is trained with ground-truth actions that are fed to the action encoder. This stabilizes action sequence prediction and speeds up convergence.

## Data Format
- Episodes are preprocessed to LeRobot format since it is supported by GR00T-N1.5:
  - data: parquet per episode with observation.state, action, and timestamps.
  - videos: mp4 per modality.
  - meta: info.json, episodes.jsonl, tasks.jsonl, modality.json, stats.json.
- Downsampling and lookahead are configurable in the dataloader.
