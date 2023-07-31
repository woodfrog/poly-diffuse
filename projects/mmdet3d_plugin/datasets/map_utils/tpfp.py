import numpy as np
from .tpfp_chamfer import custom_polyline_score


def custom_tpfp_gen(gen_lines,
             gt_lines,
             cls_name,
             consider_angle=False,
             threshold=0.5,
             threshold_angle=5,
             metric='chamfer'):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if metric == 'chamfer':
        if threshold >0:
            threshold= -threshold
    else:
        raise NotImplementedError

    th_angle = threshold_angle * 2 if cls_name == 'ped_crossing' else threshold_angle

    # import pdb;pdb.set_trace()
    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp
    
    if num_gens == 0:
        return tp, fp
    
    gen_scores = gen_lines[:,-1] # n
    # distance matrix: n x m

    pred_pts = gen_lines[:,:-1].reshape(num_gens,-1,2)
    gt_pts = gt_lines.reshape(num_gts,-1,2)

    matrix = custom_polyline_score(
            gen_lines[:,:-1].reshape(num_gens,-1,2), 
            gt_lines.reshape(num_gts,-1,2),linewidth=2., metric=metric)
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_maxids = np.argsort(-matrix, axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-gen_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            cand_gt_ids = [x for x in matrix_maxids[i] if matrix[i][x] >= threshold]
            matched_gt = None
            for gt_id in cand_gt_ids:
                if consider_angle:
                    # find the angle-matched g.t. from all location-matched g.t.candidates
                    avg_angle_diff = compute_angle_diff(pred_pts[i], gt_pts[gt_id], cls_name, num_fixed_pts=100)
                    if avg_angle_diff <= th_angle:
                        matched_gt = gt_id
                        break
                else:
                    matched_gt = gt_id
                    break
            if matched_gt is not None and not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def compute_angle_diff(pred_pts, gt_pts, class_name, num_fixed_pts=20):
    assert pred_pts.shape[0] == num_fixed_pts
    assert gt_pts.shape[0] == num_fixed_pts
    if class_name == 'ped_crossing': # is polygon (ped crossing)
        winding_gt = compute_winding(gt_pts)
        winding_pred = compute_winding(pred_pts)
        if winding_pred*winding_gt < 0:
            pred_pts = np.concatenate([pred_pts[0:1, :], pred_pts[-1:0:-1, :]], axis=0)
        rotated_targets = _rotate_target(gt_pts, num_fixed_pts)
    else:
        rotated_targets = _rotate_target_line(gt_pts, num_fixed_pts)

    dists = np.abs(rotated_targets - pred_pts[None, :, :]).sum(-1).mean(-1)
    min_gt_idx = dists.argmin(0)
    min_gt_pts = rotated_targets[min_gt_idx]
    edges_pred = (pred_pts[1:] - pred_pts[:-1]).astype(np.float32)
    edges_gt = min_gt_pts[1:] - min_gt_pts[:-1]
    edges_pred_norm = np.maximum(np.linalg.norm(edges_pred, 2, axis=-1), 1e-6)
    edges_gt_norm = np.maximum(np.linalg.norm(edges_gt, 2, axis=-1), 1e-6)
    cos_angle = (edges_pred * edges_gt).sum(-1) / (edges_pred_norm * edges_gt_norm)
    cos_angle = np.clip(cos_angle, -1, 1)
    angles = np.arccos(cos_angle) / np.pi * 180

    if np.isnan(angles).sum() != 0:
        import pdb; pdb.set_trace()

    avg_angle = angles.mean()
    return avg_angle


def _rotate_target(target, num_valid_vert):
    all_rotated_targets = []
    for start_idx in range(num_valid_vert-1):
        rotated_target = np.roll(target[:num_valid_vert-1], -start_idx, 0)
        all_rotated_targets.append(rotated_target)
    rotated_targets = np.stack(all_rotated_targets, axis=0)
    repeat_pt = rotated_targets[:, :1, :]
    rotated_targets = np.concatenate([rotated_targets, repeat_pt], axis=1)
    zero_paddings = np.zeros([rotated_targets.shape[0], target.shape[0] - num_valid_vert, 
                                rotated_targets.shape[2]])
    rotated_targets = np.concatenate([rotated_targets, zero_paddings], axis=1)
    return rotated_targets
    

def _rotate_target_line(target, num_valid_vert):
    reversed_target = np.flip(target, axis=0)
    rotated_targets = np.stack([target, reversed_target], axis=0)
    return rotated_targets


def compute_winding(polygon_verts):
    winding_sum = 0
    for idx, vert in enumerate(polygon_verts):
        next_idx = idx + 1 if idx < len(polygon_verts) - 1 else 0
        next_vert = polygon_verts[next_idx]
        winding_sum += (next_vert[0] - vert[0]) * (next_vert[1] + vert[1])
    return winding_sum