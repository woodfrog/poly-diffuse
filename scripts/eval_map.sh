#!/usr/bin/env bash
PORT=${PORT:-29505}

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    eval.py "projects/configs/maptr/maptr_tiny_r50.py"  --launcher pytorch --eval chamfer \
    --results_path "outputs/nuscenes_polydiffuse_step10/npy/deformed_results.json" \
    --consider_angle

