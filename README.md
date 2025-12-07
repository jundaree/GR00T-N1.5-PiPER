Server Container

docker build -t arm-gr00t-train .
docker run -v /home/junhyun/arm/datasets:/workspace/datasets \
--name arm-gr00t-train \
--gpus all \
--shm-size="8g" \
-it arm-gr00t-train


python dataloader/dataloader.py --input-root datasets/training_dataset_mujoco --output-root datasets_preprocessed/training_dataset_mujoco

<!-- python scripts/gr00t_finetune.py --dataset-path ./datasets_preprocessed/train_dataset_mujoco \
--output_dir outputs/gr00t_finetune_mujoco_full \
--data_config piper \
--batch_size 1 \
--num-gpus 1 -->


python scripts/gr00t_finetune.py --dataset-path ./datasets_preprocessed/train_dataset_mujoco \
--output_dir outputs/gr00t_finetune_mujoco_lora \
--data_config piper \
--lora_rank 64 \
--batch_size 1 \
--num-gpus 1

python scripts/gr00t_finetune.py --dataset-path ./datasets_preprocessed/train_dataset_mujoco \
--output_dir outputs/gr00t_finetune_mujoco_lora \
--resume \
--max_steps 50000 \
--data_config piper \
--lora_rank 64 \
--batch_size 1 \
--num-gpus 1

python scripts/inference_service.py --server \
  --model_path outputs/gr00t_finetune_mujoco_lora/checkpoint-10000 \
  --embodiment-tag new_embodiment \
  --data-config piper \
  --denoising-steps 4

python scripts/eval_policy.py --plot \
   --embodiment_tag new_embodiment \
   --model_path outputs/gr00t_finetune_mujoco_lora/checkpoint-10000 \
   --data_config piper \
  --dataset_path datasets_preprocessed/val_dataset_mujoco \
   --modality_keys single_arm gripper  

python scripts/eval_policy.py --trajs 10\
   --embodiment_tag new_embodiment \
   --model_path outputs/gr00t_finetune_mujoco_lora/checkpoint-10000 \
   --data_config piper \
  --dataset_path datasets_preprocessed/val_dataset_mujoco \
   --modality_keys single_arm gripper  


todo list
1. dataloader : done
2. choose http vs zmq : http
3. divide train & val dataset : done
4. finetune the model with lora and sim data : done
5. simulation_service.py vs inference_service.py ??? : infernce_service.py
6. eval_policy.py
7. eval_simulation.py with integrating mujoco
8. finetune the model with lora and real world data
10. eval_real_piperx.py with integrating piper_sdk

Client(Mujoco or Piper) Container

pip install requests msgpack