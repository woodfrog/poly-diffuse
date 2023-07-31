import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
from PIL import Image
from shapely.geometry import LineString
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from scipy.spatial import distance


#----------------------------------------------------------------------------
# Visualization utils

def plot_map(viz_pts, labels, out_path, pc_range, colors_plt, car_img, matching=None,
             matching_scores=None, matching_angles=None, prec=None, recall=None, viz_end=True, viz_score=True):
    plt.figure(figsize=(2, 4))
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')
    viz_pts = viz_pts.copy()
    for item_idx, (gt_bbox_3d, gt_label_3d) in enumerate(zip(viz_pts, labels)):
        pts = gt_bbox_3d
        pts[:, 0] *= pc_range[3]
        pts[:, 1] *= pc_range[4]
        x = np.array([pt[0] for pt in pts])
        y = np.array([pt[1] for pt in pts])

        # show the hit pred with a different color
        if viz_score and (matching is not None and item_idx in matching):
            plt.plot(x, y, color='red',linewidth=1,alpha=0.8,zorder=-1)
        else:
            plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)

        if viz_score and (matching is not None and item_idx in matching):
            plt.scatter(x[1:-1], y[1:-1], color='red',s=2,alpha=0.8,zorder=-1)
        else:
            plt.scatter(x[1:-1], y[1:-1], color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
        
        if viz_end:
            plt.scatter(x[0:1], y[0:1], color='purple',s=3,alpha=0.8,zorder=-1)
            plt.scatter(x[-1:], y[-1:], color='black',s=3,alpha=0.8,zorder=-1)
        else:
            plt.scatter(x[0:1], y[0:1], color=colors_plt[gt_label_3d],s=3,alpha=0.8,zorder=-1)
            plt.scatter(x[-1:], y[-1:], color=colors_plt[gt_label_3d],s=3,alpha=0.8,zorder=-1)

        # show the matching pos error and angle error for the instance (draw on the middle of the poly)
        if viz_score and (matching_scores is not None):
            if matching_scores[item_idx] < 10:
                text_x = np.clip(x[9]-1, pc_range[0]+1, pc_range[3]-1) 
                text_y = np.clip(y[9], pc_range[1]+1, pc_range[4]-1) 
                plt.text(text_x, text_y, '{:.2f}'.format(matching_scores[item_idx]), fontsize=5)
                plt.text(text_x-1, text_y-1, 'ang:{:.1f}'.format(matching_angles[item_idx]), fontsize=5)

    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    if viz_score and (prec is not None and recall is not None):
        viz_text = 'prec:{:.2f}, recall:{:.2f}'.format(prec, recall)
        plt.text(0.5, 0.5, viz_text, fontsize=8, horizontalalignment='center',
        verticalalignment='center')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches='tight', format='png', dpi=200)
    plt.close()   

    pil_image = Image.open(img_buf)
    pil_image = pil_image.convert('RGBA')
    pil_image = _remote_white_bg(pil_image)
    pil_image.save(out_path)


def visualize_all_iter(all_pts, labels, out_path, pc_range, colors_plt, car_img):
    all_results = []
    for iter_pts in all_pts:
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        for gt_bbox_3d, gt_label_3d in zip(iter_pts, labels):
            pts = np.clip(gt_bbox_3d, -1, 1)
            pts[:, 0] *= pc_range[3]
            pts[:, 1] *= pc_range[4]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])

            plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(x, y, color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
            #plt.scatter(x[1:-1], y[1:-1], color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
            #plt.scatter(x[0:1], y[0:1], color='red',s=2,alpha=0.8,zorder=-1)
            #plt.scatter(x[-1:], y[-1:], color='black',s=2,alpha=0.8,zorder=-1)
        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        img_buf = io.BytesIO()
        plt.savefig(img_buf, bbox_inches='tight', format='png', dpi=200)
        plt.close()
        viz_image = np.array(Image.open(img_buf))
        all_results.append(viz_image)

    imageio.mimsave(out_path, all_results)


def _remote_white_bg(img):
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img


def preprocess_init_result(result, pc_range, conf_thresh=0.1):
    filtered_items = [item for item in result if item['confidence_level'] >= conf_thresh]
    pred_pts = np.array([item['pts'] for item in filtered_items])
    pred_labels = np.array([item['type'] for item in filtered_items])
    pred_confs = np.array([item['confidence_level'] for item in filtered_items])
    processed_pred_pts = pred_pts.copy()

    for poly_idx, poly_pts in enumerate(pred_pts):
        start_pts = poly_pts[0]
        end_pts = poly_pts[-1]
        is_poly = np.equal(start_pts, end_pts)
        is_poly = is_poly.all()
        pts_num, coords_num = poly_pts.shape
        if is_poly or pred_labels[poly_idx] == 1:
            # TODO: determine the direction (CC or C), then pick the first vertex
            no_dup_points = poly_pts[:-1, :].copy()
            sort_inds = np.lexsort((no_dup_points[:, 0], no_dup_points[:, 1]))
            start_idx = sort_inds[0]
            sorted_verts = np.concatenate([no_dup_points[start_idx:, :], no_dup_points[:start_idx, :]], axis=0)
            winding = compute_winding(sorted_verts)
            # Note that the algorithm is flipped since the y-axis of image coordinate frame is flipped...
            if winding < 0: # meaning the current polygon is clock-wise, reverse the order
                sorted_verts = np.concatenate([sorted_verts[0:1, :], sorted_verts[-1:0:-1, :]], axis=0)
            sorted_verts = np.concatenate([sorted_verts, sorted_verts[0:1, :]], axis=0)
        else:
            #determine the fisrt vertex
            sampled_points = poly_pts.copy()
            vert_diff = np.abs(sampled_points[0, 1] - sampled_points[-1, 1])
            if vert_diff > 1:
                sort_inds = np.lexsort((sampled_points[:, 1],))
            else:
                sort_inds = np.lexsort((sampled_points[:, 0],))
            first_vert_ind = np.where(sort_inds == 0)[0]
            last_vert_ind = np.where(sort_inds == poly_pts.shape[0] - 1)[0]
            if first_vert_ind > last_vert_ind:
                sorted_verts = sampled_points[::-1, :].copy()
            else:
                sorted_verts = sampled_points
        processed_pred_pts[poly_idx] = sorted_verts

    if len(pred_pts) == 0:
        return None, None, None, None, None
    else:
        processed_pred_pts[:, :, 0] /= pc_range[3]
        processed_pred_pts[:, :, 1] /= pc_range[4]

        remaining_items = [item for item in result if item['confidence_level'] < conf_thresh]
        return processed_pred_pts, pred_labels, pred_confs, filtered_items, remaining_items


def compute_winding(polygon_verts):
    winding_sum = 0
    for idx, vert in enumerate(polygon_verts):
        next_idx = idx + 1 if idx < len(polygon_verts) - 1 else 0
        next_vert = polygon_verts[next_idx]
        winding_sum += (next_vert[0] - vert[0]) * (next_vert[1] + vert[1])
    return winding_sum



#----------------------------------------------------------------------------
# Evaluation utils


cls_id_to_name = {
    0: 'divider',
    1: 'ped_crossing',
    2: 'boundary',
}


def compute_samplewise_metric(pred_pts, pred_labels, pred_confs, gt_pts, gt_labels, pc_range, threshold):
    if isinstance(pred_pts, np.ndarray):
        pred_pts = pred_pts.copy()
        pred_labels = pred_labels.copy()
    else:
        pred_pts = pred_pts.cpu().numpy().copy()
        pred_labels = pred_labels.cpu().numpy().copy()
    gt_pts = gt_pts.cpu().numpy().copy()
    gt_labels = gt_labels.cpu().numpy().copy()

    pred_pts[:, :, 0] *= pc_range[3]
    pred_pts[:, :, 1] *= pc_range[4]
    gt_pts[:, :, 0] *= pc_range[3]
    gt_pts[:, :, 1] *= pc_range[4]

    gt_covered = np.zeros(gt_labels.shape[0], dtype=np.bool)
    pred_hits = np.zeros(pred_labels.shape[0], dtype=np.bool)
    pred_matching_scores = np.zeros(pred_labels.shape[0])
    pred_matching_angles = np.zeros(pred_labels.shape[0])
    pred_to_gt_matchings = dict()

    cls_results = {}
    for class_id in cls_id_to_name:
        pred_ids = np.where(pred_labels == class_id)[0]
        gt_ids = np.where(gt_labels == class_id)[0]
        cls_pred_pts = pred_pts[pred_ids]
        cls_gt_pts = gt_pts[gt_ids]
        cls_pred_confs = pred_confs[pred_ids]

        # keep record of matched prediction and G.T.
        if len(gt_ids) == 0 or len(pred_ids) == 0:
            cls_results[class_id] = {
                'prec': 0, 'recall': 0, 'num_hit':0,
                'num_gt': len(gt_ids), 'num_pred': len(pred_ids),
            }
            continue

        cls_gt_covers, cls_pred_hits, cls_matchings, cls_matching_scores, cls_matching_angles \
            = evaluate_per_class(pred_ids, class_id, cls_pred_pts, cls_gt_pts, cls_pred_confs, threshold)

        cls_prec = cls_pred_hits.sum() / len(pred_ids)
        cls_recall = cls_gt_covers.sum() / len(gt_ids)
        cls_entry = {
            'prec': cls_prec, 'recall': cls_recall,
            'num_gt': len(gt_ids), 'num_pred': len(pred_ids),
            'num_hit': len(cls_matchings),
        }
        cls_results[class_id] = cls_entry

        gt_covered[gt_ids] = cls_gt_covers
        pred_hits[pred_ids] = cls_pred_hits
        pred_matching_scores[pred_ids] = cls_matching_scores
        pred_matching_angles[pred_ids] = cls_matching_angles
        for pred_id, gt_id in cls_matchings.items():
            pred_to_gt_matchings[pred_ids[pred_id]] = gt_ids[gt_id]
    
    prec = pred_hits.sum() / pred_hits.shape[0]
    recall = gt_covered.sum() / gt_covered.shape[0]
    return prec, recall, cls_results, pred_to_gt_matchings, pred_matching_scores, pred_matching_angles


def evaluate_per_class(pred_ids, class_id, pred_pts, gt_pts, pred_confs, threshold):
    num_preds = pred_pts.shape[0]
    num_gts = gt_pts.shape[0]
    hits = np.zeros(num_preds).astype(np.bool)
    gt_covered = np.zeros(num_gts).astype(np.bool)
    if threshold >0:
        threshold= -threshold
    
    #angle_thresh = 10 if class_id != 1 else 20
    
    if num_gts == 0 or num_preds == 0:
        return hits, gt_covered, {}

    sampled_pred_pts = resample_poly(pred_pts)
    sampled_gt_pts = resample_poly(gt_pts)

    matrix = custom_polyline_score(sampled_pred_pts, sampled_gt_pts, linewidth=2, class_id=class_id)

    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-pred_confs)

    pred_to_gt_matching = {}
    #pred_matching_scores = np.ones([num_preds]) * 100
    pred_matching_angles = np.ones([num_preds]) * 100
    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            # TODO: further compute the angle diff here 
            matched_gt = matrix_argmax[i]
            avg_angle_diff = compute_angle_diff(sampled_pred_pts[i], sampled_gt_pts[matched_gt], class_id, num_fixed_pts=100)
            if not gt_covered[matched_gt]:
            #if not gt_covered[matched_gt] and avg_angle_diff < angle_thresh:
                gt_covered[matched_gt] = True
                hits[i] = True
                pred_to_gt_matching[i.tolist()] = matched_gt
                #pred_matching_scores[i] = -matrix_max[i]
                pred_matching_angles[i] = avg_angle_diff
    pred_matching_scores = -matrix_max
    return gt_covered, hits, pred_to_gt_matching, pred_matching_scores, pred_matching_angles


def compute_angle_diff(pred_pts, gt_pts, class_id, num_fixed_pts=20):
    assert pred_pts.shape[0] == num_fixed_pts
    assert gt_pts.shape[0] == num_fixed_pts
    if class_id == 1: # is polygon (ped crossing)
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
    


def resample_poly(poly_pts, num_sample=100):
    results = []
    for pts in poly_pts:
        line = LineString(pts)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                                for distance in distances]).reshape(-1, 2)
        results.append(sampled_points)
    results = np.stack(results, axis=0)
    return results


def custom_polyline_score(pred_lines, gt_lines, linewidth=1., metric='chamfer', class_id=None):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''
    assert metric == 'chamfer'
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)


    pred_lines_shapely = \
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                          for i in pred_lines]
    gt_lines_shapely =\
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                        for i in gt_lines]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    iou_matrix = np.full((num_preds, num_gts), -100.)

    for i, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                dist_mat = distance.cdist(
                    pred_lines[pred_id], gt_lines[i], 'euclidean')
                valid_ab = dist_mat.min(-1).mean()
                valid_ba = dist_mat.min(-2).mean()

                iou_matrix[pred_id, i] = -(valid_ba+valid_ab)/2


    return iou_matrix

