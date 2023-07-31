CUDA_VISIBLE_DEVICES=0 python train.py --outdir=training-runs/nuscenes/guide \
    --train_mode=guide --config_path="projects/configs/maptr/maptr_tiny_r50.py" \
    --batch=32  --lr=2e-4 --tick=1 --snap=5 --workers=8 \
    --p_mean=1.0 --p_std=4.0 --sig_data=1.0 \
    --exp_name_suffix="reg_0.1" --duration 0.1
