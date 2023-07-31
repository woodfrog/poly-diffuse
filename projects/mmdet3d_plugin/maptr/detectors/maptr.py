import copy
import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16

import numpy as np


@DETECTORS.register_module()
class MapTR(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(MapTR,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # ----------------------------
        # Caching image information for poly-diffuse inference
        self.cached = False # whether the image features have been cached 
        self.img_metas = None
        self.img_feats = None
        self.prev_bev = None
        # ----------------------------

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            
            #TODO: understand the purpose of the image grid mask
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          query_coords,
                          timestep,
                          poly_class,
                          poly_mask,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          use_cached_feat=False,
                          cache_image_feat=False,):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(
            query_coords, timestep, poly_class, poly_mask, pts_feats, img_metas, prev_bev, 
            use_cached_feat=use_cached_feat, cache_image_feat=cache_image_feat)
        return outs

    def forward(self, x_t, timestep, **kwargs):
        return self._forward(x_t, timestep, **kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def _forward(self,
                query_coords,
                timestep,  
                use_cached_feat=False,
                cache_image_feat=False,
                poly_class=None,
                poly_mask=None,
                points=None,
                img_metas=None,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                gt_labels=None,
                gt_bboxes=None,
                img=None,
                proposals=None,
                gt_bboxes_ignore=None,
                img_depth=None,
                img_mask=None,
        ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if not use_cached_feat:
            len_queue = img.size(1)
            prev_img = img[:, :-1, ...]
            img = img[:, -1, ...]

            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None

            img_metas = [each[len_queue-1] for each in img_metas]
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            if cache_image_feat:
                self.set_cache(img_metas, img_feats, prev_bev)
        else:
            assert self.cached
            img_metas, img_feats, prev_bev = self.img_metas, self.img_feats, self.prev_bev
        
        outputs = self.forward_pts_train(query_coords, timestep, poly_class, poly_mask,
                                        img_feats, gt_bboxes_3d,
                                        gt_labels_3d, img_metas,
                                        gt_bboxes_ignore, prev_bev, 
                                        use_cached_feat=use_cached_feat, 
                                        cache_image_feat=cache_image_feat)
        return outputs
    
    def set_cache(self, img_metas, img_feats, prev_bev):
        self.img_metas = img_metas
        self.img_feats = img_feats
        self.prev_bev = prev_bev
        self.cached = True
    
    def clear_cache(self):
        self.cached = False
        self.img_metas, self.img_feats, self.prev_bev = None, None, None
        self.pts_bbox_head.clear_cache()
    
    def load_from_ckpt(self, state_dict, keys_to_load):
        own_state = self.state_dict()
        loaded = 0
        loaded_names = []
        for name, param in state_dict.items():
            included = np.array([item in name for item in keys_to_load], dtype=np.bool).any()
            if included:
                if name in own_state:
                    own_state[name].copy_(param)
                    loaded += 1
                    loaded_names.append(name)
        print('Loaded {} / {} from the pre-trained ckpt'.format(loaded, len(own_state)))
        return loaded_names
 