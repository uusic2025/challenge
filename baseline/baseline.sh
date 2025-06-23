python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=2 \
    --master_port=12345 \
    omni_train.py \
    --output_dir=exp_out/trial_1 \
    --prompt \~
    --base_lr=0.003 \
export CUDA_VISIBLE_DEVICES=0,1



python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12345 \
    omni_test.py \
    --output_dir=exp_out/trial_1 \
    --prompt \
    --base_lr=0.003 \
export CUDA_VISIBLE_DEVICES=0