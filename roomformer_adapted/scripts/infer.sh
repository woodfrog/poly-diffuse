python infer.py  \
--data_path ../data/processed_s3d --image_size 256 \
--checkpoint_path ./checkpoints/roomformer_adapted/checkpoint.pth \
--viz_base ./results/roomformer_adapted/viz --save_base ./results/roomformer_adapted/npy
