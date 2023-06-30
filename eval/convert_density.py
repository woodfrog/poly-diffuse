import os
import numpy as np
import cv2
from planar_graph_utils import plot_floorplan_with_regions, process_floorplan

source = '../data/processed_s3d/test/'
dst_input = './visualizations/density'
dst_gt = './visualizations/gt'

if not os.path.exists(dst_input):
    os.makedirs(dst_input)
if not os.path.exists(dst_gt):
    os.makedirs(dst_gt)


for dirname in sorted(os.listdir(source)):
    if 'scene' not in dirname:
        continue
    density_path = os.path.join(source, dirname, 'density.png')
    density_img = cv2.imread(density_path)
    density = 255 - density_img
    out_path = os.path.join(dst_input, dirname + '.png')
    out_alpha_path = os.path.join(dst_input, dirname + '_alpha.png')
    alphas = np.zeros([density.shape[0], density.shape[1], 1], dtype=np.int32)
    alphas[density_img.sum(axis=-1) > 0] = 255
    density_alpha = np.concatenate([density, alphas], axis=-1)
    cv2.imwrite(out_path, density)
    cv2.imwrite(out_alpha_path, density_alpha)
    
    floorplan_path = os.path.join(source, dirname, 'processed_floorplan.npy')
    
    gt = np.load(floorplan_path, allow_pickle=True).tolist()
    floorplan = [item['polygon'] for item in gt if item['label'] not in ['door', 'window', 'outwall']]
    corners, edges, regions = process_floorplan(floorplan)

    floorplan_image = plot_floorplan_with_regions(regions, corners, edges, scale=1000)
    cv2.imwrite(os.path.join(dst_gt, '{}.png'.format(dirname[-5:])), floorplan_image)
