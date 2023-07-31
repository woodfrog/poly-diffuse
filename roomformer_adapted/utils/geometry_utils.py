import torch
import numpy as np
import cv2


def building_metric(logits, label):
    preds = torch.argmax(logits, dim=-1)
    true_ids = torch.where(label==1)
    num_true = true_ids[0].shape[0]
    tp = (preds[true_ids] == 1).sum().double()
    recall = tp / num_true
    prec = tp / (preds == 1).sum()
    fscore = 2 * recall * prec / (prec + recall)
    return recall, prec, fscore


def edge_acc(logits, label, lengths, gt_values):
    """
        edge f1-score for training/validation logging
    """
    all_acc = list()
    for i in range(logits.shape[0]):
        length = lengths[i]
        gt_value = gt_values[i, :length]
        pred_idx = torch.where(gt_value == 2)
        if len(pred_idx[0]) == 0:
            continue
        else:
            preds = torch.argmax(logits[i, :, :length][:, pred_idx[0]], dim=0)
            gts = label[i, :length][pred_idx[0]]
            pos_ids = torch.where(gts == 1)
            correct = (preds[pos_ids] == gts[pos_ids]).sum().float()
            num_pos_gt = len(pos_ids[0])
            recall = correct / num_pos_gt if num_pos_gt > 0 else torch.tensor(0)
            num_pos_pred = (preds == 1).sum().float()
            prec = correct / num_pos_pred if num_pos_pred > 0 else torch.tensor(0)
            f_score = 2.0 * prec * recall / (recall + prec + 1e-8)
            f_score = f_score.cpu()
            all_acc.append(f_score)
    if len(all_acc) > 1:
        all_acc = torch.stack(all_acc, 0)
        avg_acc = all_acc.mean()
    else:
        avg_acc = all_acc[0]
    return avg_acc


def corner_eval(targets, outputs):
    assert isinstance(targets, np.ndarray)
    assert isinstance(outputs, np.ndarray)
    output_to_gt = dict()
    gt_to_output = dict()
    for target_i, target in enumerate(targets):
        dist = (outputs - target) ** 2
        dist = np.sqrt(dist.sum(axis=-1))
        min_dist = dist.min()
        min_idx = dist.argmin()
        if min_dist < 5 and min_idx not in output_to_gt:  # a positive match
            output_to_gt[min_idx] = target_i
            gt_to_output[target_i] = min_idx
    tp = len(output_to_gt)
    prec = tp / len(outputs)
    recall = tp / len(targets)
    return prec, recall


def rectify_data(image, annot):
    rows, cols, ch = image.shape
    bins = [0 for _ in range(180)]  # 5 degree per bin
    # edges vote for directions

    gauss_weights = [0.1, 0.2, 0.5, 1, 0.5, 0.2, 0.1]

    for src, connections in annot.items():
        for end in connections:
            edge = [(end[0] - src[0]), -(end[1] - src[1])]
            edge_len = np.sqrt(edge[0] ** 2 + edge[1] ** 2)
            if edge_len <= 10:  # skip too short edges
                continue
            if edge[0] == 0:
                bin_id = 90
            else:
                theta = np.arctan(edge[1] / edge[0]) / np.pi * 180
                if edge[0] * edge[1] < 0:
                    theta += 180
                bin_id = int(theta.round())
                if bin_id == 180:
                    bin_id = 0
            for offset in range(-3, 4):
                bin_idx = bin_id + offset
                if bin_idx >= 180:
                    bin_idx -= 180
                bins[bin_idx] += np.sqrt(edge[1] ** 2 + edge[0] ** 2) * gauss_weights[offset + 2]

    bins = np.array(bins)
    sorted_ids = np.argsort(bins)[::-1]
    bin_1 = sorted_ids[0]
    remained_ids = [idx for idx in sorted_ids if angle_dist(bin_1, idx) >= 30]
    bin_2 = remained_ids[0]
    if bin_1 < bin_2:
        bin_1, bin_2 = bin_2, bin_1

    dir_1, dir_2 = bin_1, bin_2
    # compute the affine parameters, and apply affine transform to the image
    origin = [127, 127]
    p1_old = [127 + 100 * np.cos(dir_1 / 180 * np.pi), 127 - 100 * np.sin(dir_1 / 180 * np.pi)]
    p2_old = [127 + 100 * np.cos(dir_2 / 180 * np.pi), 127 - 100 * np.sin(dir_2 / 180 * np.pi)]
    pts1 = np.array([origin, p1_old, p2_old]).astype(np.float32)
    p1_new = [127, 27]  # y_axis
    p2_new = [227, 127]  # x_axis
    pts2 = np.array([origin, p1_new, p2_new]).astype(np.float32)

    M1 = cv2.getAffineTransform(pts1, pts2)

    all_corners = list(annot.keys())
    all_corners_ = np.array(all_corners)
    ones = np.ones([all_corners_.shape[0], 1])
    all_corners_ = np.concatenate([all_corners_, ones], axis=-1)
    new_corners = np.matmul(M1, all_corners_.T).T

    M = np.concatenate([M1, np.array([[0, 0, 1]])], axis=0)

    x_max = new_corners[:, 0].max()
    x_min = new_corners[:, 0].min()
    y_max = new_corners[:, 1].max()
    y_min = new_corners[:, 1].min()

    side_x = (x_max - x_min) * 0.1
    side_y = (y_max - y_min) * 0.1
    right_border = x_max + side_x
    left_border = x_min - side_x
    bot_border = y_max + side_y
    top_border = y_min - side_y
    pts1 = np.array([[left_border, top_border], [right_border, top_border], [right_border, bot_border]]).astype(
        np.float32)
    pts2 = np.array([[5, 5], [250, 5], [250, 250]]).astype(np.float32)
    M_scale = cv2.getAffineTransform(pts1, pts2)

    M = np.matmul(np.concatenate([M_scale, np.array([[0, 0, 1]])], axis=0), M)

    new_image = cv2.warpAffine(image, M[:2, :], (cols, rows), borderValue=(255, 255, 255))
    all_corners_ = np.concatenate([all_corners, ones], axis=-1)
    new_corners = np.matmul(M[:2, :], all_corners_.T).T

    corner_mapping = dict()
    for idx, corner in enumerate(all_corners):
        corner_mapping[corner] = new_corners[idx]

    new_annot = dict()
    for corner, connections in annot.items():
        new_corner = corner_mapping[corner]
        tuple_new_corner = tuple(new_corner)
        new_annot[tuple_new_corner] = list()
        for to_corner in connections:
            new_annot[tuple_new_corner].append(corner_mapping[tuple(to_corner)])

    # do the affine transform
    return new_image, new_annot, M


def angle_dist(a1, a2):
    if a1 > a2:
        a1, a2 = a2, a1
    d1 = a2 - a1
    d2 = a1 + 180 - a2
    dist = min(d1, d2)
    return dist
