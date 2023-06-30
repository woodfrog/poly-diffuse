import os
import cv2

import numpy as np
from torch.utils.data import Dataset
import skimage

from shapely.geometry import Polygon


class S3DPolygonDataset(Dataset):
    def __init__(
        self,
        data_dir,
        init_dir,
        split,
        rand_aug=False,
        max_polygon=20,
        max_vert=40,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.init_dir = init_dir
        self.rand_aug = rand_aug
        self.split = split
        self.max_polygon = max_polygon
        self.max_vert = max_vert
        all_scenes = sorted(os.listdir(os.path.join(data_dir, split)))
        all_scenes = [x for x in all_scenes if 'scene' in x]
        self.all_scenes = all_scenes

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        
    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, idx):
        scene = self.all_scenes[idx]
        scene_dir = os.path.join(self.data_dir, self.split, scene)
        polygon_annot_path = os.path.join(scene_dir, 'processed_floorplan.npy')
        all_polygon_annot = np.load(polygon_annot_path, allow_pickle=True).tolist()
        room_polygons = [item['polygon'] for item in all_polygon_annot if item['label'] not in ['door', 'window', 'outwall']]

        if len(room_polygons) > self.max_polygon:
            new_idx = np.random.randint(0, len(self.all_scenes))
            return self.__getitem__(new_idx)
        
        density_path = os.path.join(scene_dir, 'density.png')
        normal_path = os.path.join(scene_dir, 'normals.png')
        density = cv2.imread(density_path)
        normal = cv2.imread(normal_path)
        img = np.maximum(density, normal)

        if self.rand_aug:
            img, room_polygons = self.random_aug_data(img, room_polygons)

        # create with the maximum size, as the model does not know the number of polygons and corners beforehand
        polygon_verts = np.zeros([self.max_polygon, self.max_vert, 2], dtype=np.float32)
        verts_indicator = np.zeros([self.max_polygon, self.max_vert], dtype=np.float32)
        gt_num_verts = np.zeros([self.max_polygon]).astype(np.int32)

        # 1 means provide the polygon with G.T., which serves as the condiiton (polygon-level masked reconstruction)
        polygon_mask = self.generate_polygon_mask(room_polygons)
        # NOTE: turn off the polygon masking (not used in the final version)
        polygon_mask[:] = 0

        for poly_idx, poly_item in enumerate(room_polygons):
            # pre-process the polygon, make sure that the order is fixed
            processed_polygon = self.preprocess_polygon(poly_item)            
            gt_num_verts[poly_idx] = len(poly_item)

            polygon_verts[poly_idx, :len(processed_polygon), :] = processed_polygon
            verts_indicator[poly_idx, :len(processed_polygon)] = 1
        
        # rescale the coordinate range to [0, 1]
        polygon_verts = polygon_verts / 127.5 - 1
        verts_indicator_tensor = verts_indicator[..., None] * 2 - 1
        polygon_verts = np.concatenate([polygon_verts, verts_indicator_tensor], axis=-1)

        # Init Proposal derived from the G.T.
        mimic_proposal_results = self.get_polygons_circ_init(polygon_verts, gt_num_verts)

        # Proposal model for init (only used for inference)
        if 'train' not in self.split:
            roomformer_results = self.get_results_from_proposal_model(scene)

        img = self.preprocess_image(img)
        data = {
            'image': img,
            'density_path': density_path,
            'normal_path': normal_path,
            'polygon_verts': polygon_verts, 
            'verts_indicator': verts_indicator,
            'num_polygon': len(room_polygons),
            'num_verts': gt_num_verts,
            'polygon_mask': polygon_mask,
            'mimic_proposal_results': mimic_proposal_results,
        }
        if 'train' not in self.split:
            data['roomformer_results'] = roomformer_results
        return data
    
    def generate_polygon_mask(self, polygons):
        num_verts = np.array([x.shape[0] for x in polygons])
        full_mask = np.zeros([self.max_polygon]).astype(np.int32)
        # at test time, the polygon mask is always 0
        if 'train' in self.split:
            rand_mask = np.random.random([len(polygons)]) > 0.5
            rand_mask[num_verts >= 8] = 0
            # re-generate if all effective polygons are masked...
            while rand_mask.sum() == len(polygons):
                rand_mask = np.random.random([len(polygons)]) > 0.5
                rand_mask[num_verts > 8] = 0

            full_mask[:len(polygons)] = rand_mask
        return full_mask
    
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
    
    def get_polygons_circ_init(self, polygons, gt_num_verts):
        polygons_init = polygons.copy()
        for poly_idx, poly in enumerate(polygons):
            gt_num_vert = gt_num_verts[poly_idx]
            num_vert = (poly[:, -1] == 1).sum().item()
            if num_vert == 0:
                continue
            poly = polygons[poly_idx]
            coords = (poly[poly[:, -1] == 1][:, :2] + 1) / 2
            # Use G.T. coords to compute the centroid
            if gt_num_vert != num_vert:
                coords = coords[:gt_num_vert]
            shapely_poly = Polygon(coords)
            centroid = shapely_poly.centroid
            centroid = np.array([centroid.x, centroid.y])[None, :]

            deg_interval = 360 / num_vert
            init_degs = (np.arange(0, -360, -deg_interval) - 180) / 180 * np.pi
            radius = 0.1 # set up a fixed small radius

            init_offsets = np.stack([np.cos(init_degs), np.sin(init_degs)], axis=-1) * radius
            init_coords = (centroid + init_offsets) * 2 - 1
            polygons_init[poly_idx, :num_vert, :2] = init_coords
        return polygons_init

    def get_results_from_proposal_model(self, scene):
        proposal_path = os.path.join(self.init_dir, scene[6:]+'.npy')
        proposal_data = np.load(proposal_path, allow_pickle=True)
        
        polygon_results = np.zeros([self.max_polygon, self.max_vert, 3])
        for poly_idx, poly_item in enumerate(proposal_data):
            processed_polygon = self.preprocess_polygon(poly_item[:-1, :].astype(np.float32))  
            polygon_results[poly_idx, :len(processed_polygon), :2] = processed_polygon
            polygon_results[poly_idx, :len(processed_polygon), -1] = 1
        polygon_results[:, :, :2] = polygon_results[:, :, :2] / 127.5 - 1
        polygon_results[:, :, -1] = polygon_results[:, :, -1] * 2 - 1
        return polygon_results
