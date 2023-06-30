import cv2
import matplotlib.pyplot as plt
import torch as th
import numpy as np

from shapely.geometry import Polygon
import os
import imageio


def visualize_results(pred_polygons, pred_poly_ids, gt_polygons, image, save_path):
    fig_cols = 4
    fig_rows = len(pred_polygons) // fig_cols + 2
    fig_size = (10, 3*fig_rows)
    fig = plt.figure(figsize=fig_size)

    # Add Input, G.T. and full prediction 
    fig.add_subplot(fig_rows, fig_cols, 1)

    plt.imshow(image)

    gt_polygon_image = plot_polygons(gt_polygons, image.shape[0])
    fig.add_subplot(fig_rows, fig_cols, 2)
    plt.imshow(gt_polygon_image.astype(np.uint8))

    pred_polygon_image = plot_polygons(pred_polygons, image.shape[0])
    fig.add_subplot(fig_rows, fig_cols, 3)
    plt.imshow(pred_polygon_image.astype(np.uint8))

    # Add per polygon results
    fig_idx = 5
    for polygon, poly_id in zip(pred_polygons, pred_poly_ids):
        viz_image = np.zeros(image.shape)
        prev = None
        for idx, vert in enumerate(polygon):
            c = vert
            if idx == 0:
                junc_color = (255, 69, 0)
            elif idx == len(polygon) - 1:
                junc_color = (0, 255, 0)
            else:
                junc_color = (0, 255, 255)
            cv2.circle(viz_image, (int(c[0]), int(c[1])), 5, junc_color, 2)
            if idx > 0:
                cv2.line(viz_image, (int(prev[0]), int(prev[1])), (int(c[0]), int(c[1])), (255, 0, 0), 3)
            prev = c
        cv2.putText(viz_image, f'Poly idx:{poly_id}', (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (204, 229, 255), 2, cv2.LINE_AA)
        fig.add_subplot(fig_rows, fig_cols, fig_idx)
        plt.imshow(viz_image.astype(np.uint8))
        fig_idx += 1

    plt.savefig(save_path)
    plt.close()


def visualize_ordered_polygons(polygons, image_size, out_path):
    num_verts = [len(poly) for poly in polygons]
    all_verts = sum(num_verts)
    num_cols = 8
    num_rows = all_verts // num_cols + 1
    fig_size = (20, 4 * num_rows)

    fig = plt.figure(figsize=fig_size)
    fig_idx = 1
    for polygon in polygons:
        image = np.zeros([image_size, image_size, 3])
        prev = None
        for idx, vert in enumerate(polygon):
            c = vert
            if idx == 0:
                junc_color = (0, 0, 255)
            elif idx == len(polygon) - 1:
                junc_color = (0, 255, 0)
            else:
                junc_color = (0, 255, 255)
            cv2.circle(image, (int(c[0]), int(c[1])), 4, junc_color, 1)
            if idx > 0:
                cv2.line(image, (int(prev[0]), int(prev[1])), (int(c[0]), int(c[1])), (255, 0, 0), 2)
            prev = c
            fig.add_subplot(num_rows, num_cols, fig_idx)
            plt.imshow(image)
            fig_idx += 1
    plt.savefig(out_path)


def plot_polygons(polygons, image_size):
    image = np.zeros([image_size, image_size, 3])
    for polygon_verts in polygons:
        prev = None
        for idx, vert in enumerate(polygon_verts):
            c = vert
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 255), 1)
            if idx > 0:
                cv2.line(image, (int(prev[0]), int(prev[1])), (int(c[0]), int(c[1])), (255, 0, 0), 2)
            prev = c
    image = image.astype(np.uint8)
    return image


def process_polygons(polygons, mask, omit_small=True):
    pred_scores = ~mask
    eff_poly_verts = (pred_scores > 0).sum(-1)
    eff_poly_ids = th.where(eff_poly_verts > 3)[0]
    pred_polygons = []
    final_eff_poly_ids = []
    
    for eff_poly_id in eff_poly_ids:
        end_idx = th.where(pred_scores[eff_poly_id] <= 0.5)[0]
        end_idx = end_idx[0] if len(end_idx) > 0 else pred_scores.shape[1] + 1
        poly_coords = polygons[eff_poly_id, :end_idx, :2]
        if len(poly_coords) < 3:  # remove failed polygons
            continue
        poly_coords = (poly_coords + 1) / 2
        # If enabled, skip too small polygons (likely are noisy outputs)
        coords_std = poly_coords.std(dim=0).mean()
        if omit_small and coords_std < 0.02:
            continue
        poly_coords = th.cat([poly_coords, poly_coords[:1, :]], axis=0)
        poly_coords = np.round(poly_coords.cpu().numpy() * 255)
        pred_polygons.append(poly_coords)
        final_eff_poly_ids.append(eff_poly_id)

    return pred_polygons, final_eff_poly_ids


def visualize_inter_results(results, mask, sample_idx, base_dir):
    images = []
    for idx, result in enumerate(results):
        inter_poly = th.clip(result[0], -1, 1)
        inter_polygons, inter_poly_ids = process_polygons(inter_poly.detach(), mask, omit_small=False)
        img = plot_polygons(inter_polygons, 256)
        images.append(img)
    save_path = os.path.join(base_dir, f'scene_{sample_idx}_inter_steps.gif')
    imageio.mimsave(save_path, images)


def circular_init_polygons(gt_polygons):
    bs = gt_polygons.shape[0]
    results = []
    shifted = []
    for b_i in range(bs):
        circ_init_i, shifted_gt_i = _circ_init_polygons(gt_polygons[b_i])
        results.append(circ_init_i)
        shifted.append(shifted_gt_i)
    results = th.stack(results, dim=0)
    shifted_gt = th.stack(shifted, dim=0)
    return results, shifted_gt

        
def _circ_init_polygons(polygons):
    polygons_init = polygons.clone()
    shifted_polygons = polygons.clone()
    for poly_idx, poly in enumerate(polygons):
        num_vert = (poly[:, -1] == 1).sum().item()
        if num_vert == 0:
            continue
        coords = (poly[poly[:, -1] == 1][:, :2] + 1) / 2
        coords = coords.cpu().numpy()
        shapely_poly = Polygon(coords)
        centroid = shapely_poly.centroid
        centroid = np.array([centroid.x, centroid.y])[None, :]
        dists = np.sqrt(((coords - centroid) ** 2).sum(-1))
        radius = dists.mean()
        deg_interval = 360 / num_vert
        init_degs = np.arange(0, -360, -deg_interval) / 180 * np.pi

        offs = coords - centroid
        thetas = np.arctan(offs[:, 1] / (offs[:, 0]+1e-6))
        thetas += np.pi * (offs[:, 0] < 0)
        thetas[thetas<0] = 10
        start_idx = thetas.argmin()
        shifted_polygons[poly_idx, :(num_vert-start_idx), :2] = poly[start_idx:num_vert, :2]
        shifted_polygons[poly_idx, (num_vert-start_idx):num_vert, :2] = poly[:start_idx, :2]
        init_offsets = np.stack([np.cos(init_degs), np.sin(init_degs)], axis=-1) * radius
        init_coords = centroid + init_offsets
        init_coords = th.tensor(init_coords * 2 - 1).double().to(polygons.device)
        polygons_init[poly_idx, :num_vert, :2] = init_coords
    return polygons_init, shifted_polygons