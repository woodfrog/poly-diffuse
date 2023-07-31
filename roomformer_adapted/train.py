import torch
import torch.nn as nn
import os
import time
import datetime
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from arguments import get_args_parser
from datasets.s3d_floorplan_polygons import PolygonDataset
from models.room_models import RoomModel
from models.resnet import ResNetBackbone
from models.loss import RoomCriterion
import utils.misc as utils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from einops import rearrange


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(image_size, backbone, room_model, room_criterion, data_loader,
                    optimizer, epoch, max_norm, args):
    backbone.train()
    room_model.train()
    room_criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=100, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    # get the positional encodings for all pixels
    for data in metric_logger.log_every(data_loader, print_freq, header):
        pred_logits, pred_coords, matches, poly_loss = run_model(
            data,
            backbone,
            room_model,
            room_criterion,
            epoch,
            args)

        loss = poly_loss['loss']

        if torch.isnan(loss):
            import pdb; pdb.set_trace()

        # FPN has some ununsed BN parameters, 
        # This is a workaround for DDP's error regarding those unused parameters
        pseudo_losses = [p.sum() * 0 for p in backbone.parameters()]
        pseudo_losses += [p.sum() * 0 for p in room_model.parameters()]
        pseudo_losses = torch.stack(pseudo_losses).mean()
        loss += pseudo_losses

        loss_dict = {
            'exs_loss': poly_loss['exs_loss'],
            'coord_loss': poly_loss['coord_loss'],
        }
        loss_value = loss.item()

        optimizer.zero_grad()
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(room_model.parameters(), max_norm)

        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def run_model(data, backbone, room_model, room_criterion, epoch, args):
    image = data['image'].to(args.gpu)
    gt_polys = data['polygon_verts'].to(args.gpu)
    num_polys = data['num_polygon']

    image_feats, feat_mask, all_image_feats = backbone(image)

    # run the edge model
    pred_polys = room_model(image_feats, feat_mask)
    pred_logits = rearrange(pred_polys['pred_logits'], 'b (n1 n2) c -> b n1 n2 c', n1=gt_polys.shape[1])
    pred_coords = rearrange(pred_polys['pred_coords'], 'b (n1 n2) c -> b n1 n2 c', n1=gt_polys.shape[1])
    pred_scores = pred_logits.softmax(dim=-1)[:, :, :, 1:2]
    merged_pred_polys = torch.cat([pred_coords, pred_scores], dim=-1)

    # set-to-set polygon loss
    matches, poly_loss = room_criterion(gt_polys, merged_pred_polys, pred_coords, pred_logits, num_polys)

    return pred_logits, pred_coords, matches, poly_loss


@torch.no_grad()
def evaluate(image_size, backbone, room_model, room_criterion, data_loader, epoch, args):
    backbone.eval()
    room_model.eval()
    room_criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for data in metric_logger.log_every(data_loader, 10, header):
        pred_logits, pred_coords, matches, poly_loss = run_model(
            data,
            backbone,
            room_model,
            room_criterion,
            epoch,
            args)

        loss_dict = {
            'exs_loss': poly_loss['exs_loss'],
            'coord_loss': poly_loss['coord_loss'],
        }

        loss = poly_loss['loss']
        loss_value = loss.item()
        metric_logger.update(loss=loss_value, **loss_dict)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )
        args.world_size = dist.get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    image_size = args.image_size
    data_path = args.data_path
    train_dataset = PolygonDataset(data_path, split='train', rand_aug=True, repeat_train=args.repeat_train)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=True, drop_last=True)

    test_dataset = PolygonDataset(data_path, split='val', rand_aug=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=False, drop_last=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  num_workers=args.num_workers, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=args.num_workers, 
                                 sampler=test_sampler)

    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels

    backbone = backbone.to(args.gpu)
    if args.distributed:
        backbone = DDP(backbone, device_ids=[args.gpu], output_device=args.local_rank)

    room_model = RoomModel(num_poly=20, num_vert=40, 
                          input_dim=128, hidden_dim=args.hidden_dim, num_feature_levels=4, 
                          backbone_strides=strides, backbone_num_channels=num_channels)
    room_model = room_model.to(args.gpu)
    if args.distributed:
        room_model = DDP(room_model, device_ids=[args.gpu], output_device=args.local_rank)

    room_criterion = RoomCriterion()

    backbone_params = [p for p in backbone.parameters()]
    room_params = [p for p in room_model.parameters()]

    all_params = room_params + backbone_params
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': room_params, 'lr': args.lr},
    ] , lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = args.start_epoch

    if args.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        ckpt = torch.load(args.resume, map_location=map_location)
        backbone.load_state_dict(ckpt['backbone'])
        room_model.load_state_dict(ckpt['room_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        lr_scheduler.step_size = args.lr_drop

        print('Resume from ckpt file {}, starting from epoch {}'.format(args.resume, ckpt['epoch']))
        start_epoch = ckpt['epoch'] + 1

    n_backbone_parameters = sum(p.numel() for p in backbone_params if p.requires_grad)
    n_room_parameters = sum(p.numel() for p in room_params if p.requires_grad)
    n_all_parameters = sum(p.numel() for p in all_params if p.requires_grad)
    print('number of trainable backbone params:', n_backbone_parameters)
    print('number of trainable room params:', n_room_parameters)
    print('number of all trainable params:', n_all_parameters)

    print("Start training")
    start_time = time.time()

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    #TODO: change to some val acc later, now use the validation loss
    best_val_loss = 1e4
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            image_size, backbone, room_model, room_criterion, train_dataloader,
            optimizer,
            epoch, args.clip_max_norm, args)
        lr_scheduler.step()

        if args.run_validation and epoch % 20 == 0:
            val_stats = evaluate(
                image_size, backbone, room_model, room_criterion, test_dataloader,
                epoch, args
            )

            val_loss = val_stats['loss']
            if val_loss < best_val_loss:
                is_best = True
                best_val_loss = val_loss
            else:
                is_best = False
        else:
            best_val_loss = 1e4
            is_best = False

        # only save the ckpt at the rank0
        if args.local_rank == 0 and args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if is_best:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')

            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'backbone': backbone.state_dict(),
                    'room_model': room_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_loss': best_val_loss,
                }, checkpoint_path)
            
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.distributed:
        cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
