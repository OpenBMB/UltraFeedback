
export NCCL_IGNORE_DISABLED_P2P=1

pip install transformers -U
pip install tokenizers -U
pip install deepspeed -U
pip install accelerate -U
pip install vllm -U


echo $1

export NCCL_IGNORE_DISABLED_P2P=1
export RAY_memory_monitor_refresh_ms=0
CUDA_LAUNCH_BLOCKING=1 python main_vllm_batch.py --model_type ${1}

