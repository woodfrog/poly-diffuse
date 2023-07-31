# Multiple GPUs training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=8 --master_port 12365 \
    train.py --outdir=training-runs/nuscenes/denoise \
    --exp_name_suffix="maptr_r50_samples_5M" \
    --config_path="projects/configs/maptr/maptr_tiny_r50.py" \
    --pretrained_model_ckpt="ckpts/maptr_tiny_r50_110e.pth" \
    --sig_data 1.0  --p_mean=-0.5  --p_std=1.5 --lambda_dir=2e-3 \
    --lr=6e-4 --batch=48 --batch-gpu=6 \
    --tick=1 --snap=10 --dump=500 \
    --workers=4 --duration=5 \
    --guide_ckpt="training-runs/nuscenes_pretrained_ckpts/guide/polydiffuse-nuscenes-guide/network-snapshot.pth" \


# Single GPU debugging
#CUDA_VISIBLE_DEVICES=0 python train.py --outdir=training-runs/nuscenes/denoise \
#    --exp_name_suffix="dev" \
#    --config_path="projects/configs/maptr/maptr_tiny_r50.py" \
#    --pretrained_model_ckpt="ckpts/maptr_tiny_r50_110e.pth" \
#    --batch=2  --lr=6e-4 --tick=1 --snap=1 --workers=0 \
#    --sig_data 1.0  --p_mean=-0.5  --p_std=1.5  \
#    --guide_ckpt="training-runs/nuscenes_pretrained_ckpts/guide/polydiffuse-nuscenes-guide/network-snapshot.pth" \
