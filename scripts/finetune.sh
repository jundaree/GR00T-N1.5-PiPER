

python scripts/gr00t_finetune.py --dataset-path ./datasets_preprocessed/train_dataset_mujoco_downsampled_6_lookahead \
--output_dir outputs/gr00t_finetune_mujoco_lora_downsampled_6_lookahead_alpha_128 \
--max_steps 5000 \
--save_steps 500 \
--data_config piper_mujoco \
--lora_rank 64 \
--lora_alpha 128 \
--batch_size 32 \
--num-gpus 1

python scripts/gr00t_finetune.py --dataset-path ./datasets_preprocessed/train_dataset_mujoco_downsampled_6_lookahead \
--output_dir outputs/gr00t_finetune_mujoco_lora_downsampled_6_lookahead_alpha_256 \
--max_steps 5000 \
--save_steps 500 \
--data_config piper_mujoco \
--lora_rank 64 \
--lora_alpha 256 \
--batch_size 32 \
--num-gpus 1