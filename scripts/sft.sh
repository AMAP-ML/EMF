#!/bin/bash
#SBATCH --job-name=blip3o    # Job name
#SBATCH --nodes=4                         # Number of nodes
#SBATCH --gres=gpu:8                         # Number of GPUs per node
#SBATCH --time=96:00:00                      # Time limit (hh:mm:ss)

# !/usr/bin/env bash

if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
else
    export PATH="/opt/conda/bin:$PATH"
fi

conda activate Onestep

cd ../

# export WANDB_API_KEY='your wandb key'
# export HF_HOME=/your/hf/home/
export CUDA_LAUNCH_BLOCKING=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=1200000
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64 


export HF_ENDPOINT=https://hf-mirror.com   # 例如清华镜像
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1


AR_BACKBONE=../BLIP3o/BLIP3o-NEXT-SFT-3B

DIFFUSION="../Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"

LR=5e-5
RUN_NAME="train"

echo "AR_BACKBONE: ${AR_BACKBONE}"
echo "DIFFUSION: ${DIFFUSION}"
echo "RUN_NAME: ${RUN_NAME}"

# LOCAL_DIR="models/${RUN_NAME}"

LOCAL_DIR="../models_local/${RUN_NAME}"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun \
--nproc_per_node=8 \
--nnodes=1 \
--master_port=29508 \
blip3o/train/train.py \
--deepspeed scripts/zero2.json \
--num_image_tokens 65536 \
--num_scale_tokens 3 \
--load_embeddings_from_vision True \
--model_name_or_path $AR_BACKBONE \
--diffusion_name_or_path   ${DIFFUSION} \
--version "qwen_1_5" \
--dataset_cls 'mix' \
--dispatch_batches False \
--mm_vision_select_layer -2 \
--mm_use_im_start_end True \
--group_by_modality_length True \
--image_aspect_ratio square \
--mm_patch_merge_type flat \
--num_train_epochs 30 \
--bf16 True \
--run_name $RUN_NAME \
--output_dir ${LOCAL_DIR} \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 10000 \
--learning_rate ${LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 5 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing False \
--dataloader_num_workers 1 \
--lazy_preprocess True \
--report_to tensorboard \
--torch_compile False \
--dataloader_drop_last True 