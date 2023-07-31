python generate.py --seeds=0-1 --batch=1 --port=12370 \
--config_path="projects/configs/maptr/maptr_tiny_r50.py" \
--workers=12 \
--second_order=False --proposal_type=maptr \
--init_path="init_results/maptr_test.json" \
--outdir=outputs/nuscenes_polydiffuse_step10 \
--steps 10 --sigma_max=5.0 --sigma_min=0.1 --S_churn 0 --rho 7.0 \
--viz_results=False --viz_gif=False \
--guide_ckpt="training-runs/nuscenes_pretrained_ckpts/guide/polydiffuse-nuscenes-guide/network-snapshot.pth" \
--network="./training-runs/nuscenes_pretrained_ckpts/denoise/polydiffuse-nuscenes-denoise/network-snapshot.pth"

