import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.nn_utils import positional_encoding_1d

import skimage
from einops import repeat

import cv2
import imageio
import itertools


class PolygonDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        rand_aug,
        repeat_train=1,
        max_polygon=20,
        max_vert=40,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_polygon = max_polygon
        self.max_vert = max_vert
        self.rand_aug = rand_aug
        all_scenes = sorted(os.listdir(os.path.join(data_dir, split)))
        all_scenes = [x for x in all_scenes if 'scene' in x]
        assert split in ['train', 'val', 'test'], 'Invalid split {}'.format(split)

        if split == 'train' and repeat_train > 1:
            copied_scenes = [all_scenes for _ in range(repeat_train)]
            all_scenes = list(itertools.chain.from_iterable(copied_scenes))

        self.local_scenes = all_scenes

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        
    def __len__(self):
        return len(self.local_scenes)

    def __getitem__(self, idx):
        scene = self.local_scenes[idx]
        scene_dir = os.path.join(self.data_dir, self.split, scene)
        polygon_annot_path = os.path.join(scene_dir, 'processed_floorplan.npy')
        all_polygon_annot = np.load(polygon_annot_path, allow_pickle=True).tolist()
        room_polygons = [item['polygon'] for item in all_polygon_annot if item['label'] not in ['door', 'window', 'outwall']]

        if len(room_polygons) > self.max_polygon:
            new_idx = np.random.randint(0, len(self.local_scenes))
            return self.__getitem__(new_idx)

        density_path = os.path.join(scene_dir, 'density.png')
        normal_path = os.path.join(scene_dir, 'normals.png')
        density = cv2.imread(density_path)
        normal = cv2.imread(normal_path)
        img = np.maximum(density, normal)

        if self.rand_aug:
            img, room_polygons = self.random_aug_data(img, room_polygons)

        # create with the maximum size, as the model does not know the number of polygons and corners beforehand
        polygon_verts = np.zeros([self.max_polygon, self.max_vert, 2])
        verts_indicator = np.zeros([self.max_polygon, self.max_vert])
        for poly_idx, poly_item in enumerate(room_polygons):
            # pre-process the polygon, make sure that the order is fixed
            processed_polygon = self.preprocess_polygon(poly_item)            

            polygon_verts[poly_idx, :len(poly_item), :] = processed_polygon
            verts_indicator[poly_idx, :len(poly_item)] = 1
            
        # rescale the coordinate range to [0, 1]
        polygon_verts = polygon_verts / 255
        verts_indicator_tensor = verts_indicator[..., None]
        polygon_verts = np.concatenate([polygon_verts, verts_indicator_tensor], axis=-1)

        # Visualize all the polygons, for verifying the data quality only
        #self._visualize_polygons(scene, img, polygon_verts, verts_indicator, len(room_polygons))
        
        img = self.preprocess_image(img)
        #TODO: add the image information later
        data = {
                'image_path': density_path,
                'image': img,
                'polygon_verts': polygon_verts, 
                'verts_indicator': verts_indicator,
                'num_polygon': len(room_polygons),
        }
        return data
    
    def preprocess_image(self, img):
        # preprocess image
        img = skimage.img_as_float(img).transpose((2, 0, 1))
        img = (img - np.array(self.img_mean)[:, np.newaxis, np.newaxis]) / np.array(self.img_std)[:, np.newaxis, np.newaxis]
        img = img.astype(np.float32)
        return img
    
    def preprocess_polygon(self, polygon_verts):
        sort_inds = np.lexsort((polygon_verts[:, 1], polygon_verts[:, 0]))
        start_idx = sort_inds[0]
        sorted_verts = np.concatenate([polygon_verts[start_idx:, :], polygon_verts[:start_idx, :]], axis=0)
        winding = self.compute_winding(sorted_verts)
        # Note that the algorithm is flipped since the y-axis of image coordinate frame is flipped...
        if winding < 0: # meaning the current polygon is clock-wise, reverse the order
            sorted_verts = np.concatenate([sorted_verts[0:1, :], sorted_verts[-1:0:-1, :]], axis=0)
        return sorted_verts
    
    @staticmethod
    def compute_winding(polygon_verts):
        winding_sum = 0
        for idx, vert in enumerate(polygon_verts):
            next_idx = idx + 1 if idx < len(polygon_verts) - 1 else 0
            next_vert = polygon_verts[next_idx]
            winding_sum += (next_vert[0] - vert[0]) * (next_vert[1] + vert[1])
        return winding_sum
    
    def random_aug_data(self, img, polygons):
        img, polygons = self.random_flip(img, polygons)
        img, polygons = self.random_rotate(img, polygons)
        return img, polygons

    def random_rotate(self, img, polygons):
        ## Arbitary rotation angle...
        theta = np.random.randint(0, 360) / 360 * np.pi * 2
        ## Mahanttan rotation only
        #theta = np.random.choice([0, 90, 180, 270])
        #theta = theta / 360 * np.pi * 2

        image_size = img.shape[0]
        r = image_size / 256
        origin = [127 * r, 127 * r]
        p1_new = [127 * r + 100 * np.sin(theta) * r, 127 * r - 100 * np.cos(theta) * r]
        p2_new = [127 * r + 100 * np.cos(theta) * r, 127 * r + 100 * np.sin(theta) * r]
        p1_old = [127 * r, 127 * r - 100 * r]  # y_axis
        p2_old = [127 * r + 100 * r, 127 * r]  # x_axis
        pts1 = np.array([origin, p1_old, p2_old]).astype(np.float32)
        pts2 = np.array([origin, p1_new, p2_new]).astype(np.float32)
        M_rot = cv2.getAffineTransform(pts1, pts2)

        num_verts = [len(poly) for poly in polygons]
        all_corners = np.concatenate(polygons, axis=0)
        ones = np.ones([all_corners.shape[0], 1])
        all_corners = np.concatenate([all_corners, ones], axis=-1)
        all_corners = np.matmul(M_rot, all_corners.T).T

        # if the augmentation is too aggressive, give up and do not use it
        if all_corners.min() <= 0 or all_corners.max() >= (image_size - 1):
            return img, polygons
        
        split_indices = np.cumsum(num_verts)[:-1]
        new_polygons = np.split(all_corners, split_indices, axis=0)
        
        rows, cols, ch = img.shape
        new_img = cv2.warpAffine(img, M_rot, (cols, rows), borderValue=(255, 255, 255))
        y_start = (new_img.shape[0] - image_size) // 2
        x_start = (new_img.shape[1] - image_size) // 2
        new_img = new_img[y_start:y_start + image_size, x_start:x_start + image_size, :]

        return new_img, new_polygons
    
    def random_flip(self, img, polygons):
        height, width, _ = img.shape
        rand_int = np.random.randint(0, 4)
        if rand_int == 0:
            return img, polygons

        num_verts = [len(poly) for poly in polygons]
        all_corners = np.concatenate(polygons, axis=0)
        new_corners = np.array(all_corners)

        if rand_int == 1:
            img = img[:, ::-1, :]
            new_corners[:, 0] = (width - 1) - new_corners[:, 0]
        elif rand_int == 2:
            img = img[::-1, :, :]
            new_corners[:, 1] = (height - 1) - new_corners[:, 1]
        else:
            img = img[::-1, ::-1, :]
            new_corners[:, 0] = (width - 1) - new_corners[:, 0]
            new_corners[:, 1] = (height - 1) - new_corners[:, 1]
        
        split_indices = np.cumsum(num_verts)[:-1]
        new_polygons = np.split(new_corners, split_indices, axis=0)
        return img, new_polygons
    
    @staticmethod
    def _visualize_polygons(scene, img, all_verts, all_verts_indicator, num_polygons):
        viz_dir = os.path.join('./viz_polygon', scene)
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        prev = None
        img = np.ascontiguousarray(img, dtype=np.uint8)
        for poly_i in range(num_polygons):
            verts = all_verts[poly_i]
            verts_indicator = all_verts_indicator[poly_i]
            for idx, vert in enumerate(verts):
                if verts_indicator[idx] == 0:
                    continue
                c = vert * 255
                cv2.circle(img, (int(c[0]), int(c[1])), 3, (0, 0, 255), 1)
                if idx == 0:
                    last_idx = np.where(verts_indicator ==1)[0][-1]
                    prev = verts[last_idx] * 255
                cv2.line(img, (int(prev[0]), int(prev[1])), (int(c[0]), int(c[1])), (255, 255, 0), 2)
                prev = c
                save_path = os.path.join(viz_dir, 'poly_{}_vert_{}.png'.format(poly_i, idx))
                imageio.imsave(save_path, img)
