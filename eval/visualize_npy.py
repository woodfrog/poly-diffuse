import os
import numpy as np
import cv2
import imageio
from planar_graph_utils import plot_floorplan_with_regions, process_floorplan


def main():
    results_base = './outputs/s3d_polydiffuse_step10/npy'
    viz_base = './eval/visualizations/s3d_polydiffuse_step10'
    viz_img_base = os.path.join(viz_base, 'images')
    viz_gif_base = os.path.join(viz_base, 'gifs')

    if not os.path.exists(viz_base):
        os.makedirs(viz_base)
    if not os.path.exists(viz_img_base):
        os.makedirs(viz_img_base)
    if not os.path.exists(viz_gif_base):
        os.makedirs(viz_gif_base)

    for filename in sorted(os.listdir(results_base)):
        if not 'inter' in filename:
            continue
        pg_path = os.path.join(results_base, filename)
        pred_floorplan_all_steps = np.load(pg_path, allow_pickle=True).tolist()

        print('Processing file: {}'.format(filename))

        all_images = []
        for step in range(len(pred_floorplan_all_steps)):
            corners, edges, regions = process_floorplan(pred_floorplan_all_steps[step])
            floorplan_image = plot_floorplan_with_regions(regions, corners, edges, scale=1000, sort_regions=False)
            cv2.imwrite(os.path.join(viz_img_base, '{}_step_{}.png'.format(filename[:-4], step)), floorplan_image)
            image = floorplan_image[:, :, :3][:, :, ::-1]
            alpha = floorplan_image[:, :, -1]
            image[alpha==0] = 255
            all_images.append(image)
        
        gif_out_path = os.path.join(viz_gif_base, '{}_inter_steps.gif'.format(filename[:-4]))
        imageio.mimsave(gif_out_path, all_images)


if __name__ == '__main__':
    main()
