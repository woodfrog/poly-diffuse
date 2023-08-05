python generate.py --seeds=0-1 --batch=1 --port=12347 \
--data_dir=./data/processed_s3d --init_dir=init_results/npy_roomformer_s3d_256 \
--workers 8 \
--steps=10 --sigma_max=5.0 --sigma_min=0.01 --S_churn 0.0 --rho 7.0 \
--second_order=False --compute_likelihood=False \
--proposal_type=roomformer \
--guide_ckpt="training-runs/s3d_pretrained_ckpts/guide/polydiffuse-s3d-guide/network-snapshot.pth" \
--network="./training-runs/s3d_pretrained_ckpts/denoise/polydiffuse-s3d-denoise/network-snapshot.pth" \
--outdir=outputs/s3d_polydiffuse_step10 \
--viz_results=False
