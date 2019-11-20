# -*- coding: utf-8 -*-
import numpy as np


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, gt_landmarks, corner_form_priors, iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        gt_landmarks:
        corner_form_priors (num_priors, 4): corner form priors
        iou_threshold:
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(np.expand_dims(gt_boxes, 0), np.expand_dims(corner_form_priors, 1))
    # size: num_priors
    best_target_per_prior = ious.max(1)
    best_target_per_prior_index = ious.argmax(1)
    # size: num_targets
    best_prior_per_target = ious.max(0)
    best_prior_per_target_index = ious.argmax(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior[best_prior_per_target_index] = 2
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    landmarks = gt_landmarks[best_target_per_prior_index]
    return boxes, labels, landmarks


def corner_form_to_center_form(boxes):
    return np.concatenate([(boxes[..., :2] + boxes[..., 2:]) / 2,
                           boxes[..., 2:] - boxes[..., :2]], len(boxes.shape) - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if len(center_form_priors.shape) + 1 == len(center_form_boxes.shape):
        center_form_priors = np.expand_dims(center_form_priors, 0)
    return np.concatenate([(center_form_boxes[..., :2] - center_form_priors[..., :2])
                           / center_form_priors[..., 2:] / center_variance,
                           np.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
                           ], axis=len(center_form_boxes.shape) - 1)


def convert_landmarks_to_locations(landmarks, center_form_priors, center_variance):
    if len(center_form_priors.shape) + 1 == len(landmarks.shape):
        center_form_priors = np.expand_dims(center_form_priors, 0)
    landmarks[..., 0::2] -= center_form_priors[..., 0:1]
    landmarks[..., 1::2] -= center_form_priors[..., 1:2]
    landmarks[..., 0::2] /= center_form_priors[..., 2:3]*center_variance
    landmarks[..., 1::2] /= center_form_priors[..., 3:4]*center_variance
    return landmarks


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels, gt_landmarks):
        boxes, labels, landmarks = assign_priors(gt_boxes, gt_labels, gt_landmarks,
                                                 self.corner_form_priors, self.iou_threshold)
        boxes = corner_form_to_center_form(boxes)
        locations = convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        landmarks = convert_landmarks_to_locations(landmarks, self.center_form_priors, self.center_variance)
        return locations, labels, landmarks

