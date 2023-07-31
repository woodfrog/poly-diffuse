import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.s3d_floorplan_polygons import PolygonDataset
from models.resnet import ResNetBackbone
from models.room_models import RoomModel
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse
from einops import rearrange
import tqdm

import imageio
import matplotlib.pyplot as plt


def visualize_per_iter_results(pred_polygons, gt_polygons, image, save_path):
    fig = plt.figure(figsize=(20, 8))
    fig_rows = 2
    fig_cols=6
    fig.add_subplot(fig_rows, fig_cols, 1)
    plt.imshow(image)

    gt_polygon_image = plot_polygons(gt_polygons, image.shape[0])
    fig.add_subplot(fig_rows, fig_cols, 2)
    plt.imshow(gt_polygon_image)

    for iter_i, iter_polygons in enumerate(pred_polygons):
        pred_polygon_image = plot_polygons(iter_polygons, image.shape[0])
        fig.add_subplot(fig_rows, fig_cols, iter_i+7)
        plt.imshow(pred_polygon_image)

    plt.savefig(save_path)


def plot_polygons(polygons, image_size):
    image = np.zeros([image_size, image_size, 3]).astype(np.uint8)
    for polygon_verts in polygons:
        prev = None
        for idx, vert in enumerate(polygon_verts):
            c = vert
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 255), 1)
            if idx > 0:
                cv2.line(image, (int(prev[0]), int(prev[1])), (int(c[0]), int(c[1])), (255, 0, 0), 2)
            prev = c
    return image


def visualize_ordered_polygons(polygons, image_size, out_path):
    num_verts = [len(poly) for poly in polygons]
    all_verts = sum(num_verts)
    num_cols = 8
    num_rows = all_verts // num_cols + 1
    fig_size = (20, 4 * num_rows)

    fig = plt.figure(figsize=fig_size)
    fig_idx = 1
    for polygon in polygons:
        image = np.zeros([image_size, image_size, 3]).astype(np.uint8)
        prev = None
        for idx, vert in enumerate(polygon):
            c = vert
            if idx == 0:
                junc_color = (0, 0, 255)
            elif idx == len(polygon) - 2:
                junc_color = (0, 255, 0)
            else:
                junc_color = (0, 255, 255)
            if idx != len(polygon) - 1:
                cv2.circle(image, (int(c[0]), int(c[1])), 4, junc_color, 1)
            if idx > 0:
                cv2.line(image, (int(prev[0]), int(prev[1])), (int(c[0]), int(c[1])), (255, 0, 0), 2)
            prev = c
            fig.add_subplot(num_rows, num_cols, fig_idx)
            plt.imshow(image)
            fig_idx += 1
    plt.savefig(out_path)

        
def main(data_path, ckpt_path, image_size, viz_base, save_base):
    ckpt = torch.load(ckpt_path)
    print('Load from ckpts of epoch {}'.format(ckpt['epoch']))
    ckpt_args = ckpt['args']

    test_dataset = PolygonDataset(data_path, split='test', rand_aug=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels
    backbone = nn.DataParallel(backbone)
    backbone = backbone.cuda()
    backbone.eval()

    room_model = RoomModel(num_poly=20, num_vert=40, 
                          input_dim=128, hidden_dim=ckpt_args.hidden_dim, num_feature_levels=4, backbone_strides=strides,
                          backbone_num_channels=num_channels)
    room_model = nn.DataParallel(room_model)
    room_model = room_model.cuda()
    room_model.eval()

    backbone.load_state_dict(ckpt['backbone'])
    room_model.load_state_dict(ckpt['room_model'])
    print('Loaded saved model from {}'.format(ckpt_path))

    if not os.path.exists(viz_base):
        os.makedirs(viz_base)
    if not os.path.exists(save_base):
        os.makedirs(save_base)

    data_i = 0
    for data in tqdm.tqdm(test_dataloader):
        with torch.no_grad():
            pred_polygons = get_results(data, backbone, room_model, ckpt_args, image_size=image_size)

        image_path = data['image_path'][0]
        image = imageio.imread(image_path)
        viz_image = np.stack([image] * 3, axis=-1)

        num_gt_polygons = data['num_polygon'][0]
        gt_polygons = []
        for idx in range(num_gt_polygons.item()):
            gt_verts = data['polygon_verts'][0, idx]
            coords = gt_verts[:, :2]
            exs = gt_verts[:, -1]
            eff_coords = coords[exs==1]
            eff_coords = torch.cat([eff_coords, eff_coords[:1, :]], dim=0)
            eff_coords = (eff_coords * 255).cpu().numpy().astype(np.uint8)
            gt_polygons.append(eff_coords)

        out_path = os.path.join(viz_base, '{}_pred.png'.format(data_i))
        visualize_per_iter_results(pred_polygons, gt_polygons, viz_image, out_path)

        ## Uncomment to visualize detailed order polygon results
        #out_path_ordered = os.path.join(viz_base, '{}_pred_order.png'.format(data_i))
        #visualize_ordered_polygons(pred_polygons[-1], image_size, out_path_ordered)

        scene=data['image_path'][0].split('/')[-2].split('_')[-1]
        polygons = np.array([p for p in pred_polygons[-1]], dtype=object)

        np.save(os.path.join(save_base, '{}.npy'.format(scene)), polygons)
        data_i += 1


def get_results(data, backbone, room_model, args, image_size=256):
    image = data['image']
    gt_polys = data['polygon_verts'].cuda()
    num_polys = data['num_polygon']

    image_feats, feat_mask, all_image_feats = backbone(image)
    pred_polys = room_model(image_feats, feat_mask,)

    all_iter_logits = rearrange(pred_polys['all_logits'], 'n3 b (n1 n2) c -> b n3 n1 n2 c', n1=gt_polys.shape[1])
    all_iter_coords = rearrange(pred_polys['all_coords'], 'n3 b (n1 n2) c -> b n3 n1 n2 c', n1=gt_polys.shape[1])
    all_iter_results = []

    for iter_i in range(all_iter_coords.shape[1]):
        pred_scores = all_iter_logits[0, iter_i].softmax(dim=-1)[:, :, 1]
        eff_poly_verts = (pred_scores > 0.5).sum(-1)
        eff_poly_ids = torch.where(eff_poly_verts > 3)[0]
        pred_polygons = []
        
        for eff_poly_id in eff_poly_ids:
            poly_coords = all_iter_coords[0, iter_i, eff_poly_id, pred_scores[eff_poly_id] > 0.5]
            poly_coords = torch.cat([poly_coords, poly_coords[:1, :]], axis=0)
            poly_coords = np.round(poly_coords.cpu().numpy() * 255).astype(np.uint8)
            pred_polygons.append(poly_coords)
        all_iter_results.append(pred_polygons)

    return all_iter_results


def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer adapted', add_help=False)
    parser.add_argument('--data_path', default='',
                        help='root path of the data')
    parser.add_argument('--checkpoint_path', default='',
                        help='path to the checkpoints of the model')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--viz_base', default='./results/viz',
                        help='path to save the intermediate visualizations')
    parser.add_argument('--save_base', default='./results/npy',
                        help='path to save the prediction results in npy files')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args.data_path, args.checkpoint_path, args.image_size, args.viz_base, args.save_base)
