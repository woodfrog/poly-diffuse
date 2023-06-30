CUDA_VISIBLE_DEVICES=0 python train.py --outdir=training-runs/s3d/guide \
   --train_mode=guide \
   --data_dir=data/processed_s3d --init_dir="" \
   --batch=32  --lr=2e-4 --tick=1 --snap=5 --workers=4 \
   --p_mean=1.0 --p_std=4.0 --sig_data=1.0 \
   --exp_name_suffix="reg_0.5" --duration 0.1 \
