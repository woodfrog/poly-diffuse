CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12345 \
train.py \
--data_path ../data/processed_s3d \
--epochs 200 --lr_drop 160  --batch_size 24 --output_dir ./checkpoints/roomformer_adapted \
--image_size 256 --lr 2e-4 --num_workers 8  --hidden_dim 256 \
--distributed --run_validation  --print_freq 60 \
--repeat_train 10

## Single-GPU debugging
#CUDA_VISIBLE_DEVICES=0 python train.py  --data_path ../data/processed_s3d \
#--epochs 200 --lr_drop 160  --batch_size 8  --hidden_dim 256 \
#--output_dir ./checkpoints/s3d_debug  \
#--image_size 256 --lr 2e-4 --num_workers 8
