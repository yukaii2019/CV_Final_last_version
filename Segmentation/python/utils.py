import os
import csv
import json
import random

import numpy as np
import torch
import cv2
def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def linVal(x, xlims, ylims, offset):
    '''
    Given xlims (x_min, x_max) and ylims (y_min, y_max), i.e, start and end,
    compute the value of y=f(x). Offset contains the x0 such that for all x<x0,
    y is clipped to y_min.
    '''
    if x < offset:
        return ylims[0]
    elif x > xlims[1]:
        return ylims[1]
    else:
        y = (np.diff(ylims)/np.diff(xlims))*(x - offset)
        return y.item()

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count

def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended


def true_negative_curve(confs: np.ndarray, labels: np.ndarray, nr_thresholds: int = 1000):
    """Compute true negative rates
    Args:
        confs: the algorithm outputs
        labels: the ground truth labels
        nr_thresholds: number of splits for sliding thresholds

    Returns:

    """
    thresholds = np.linspace(0, 1, nr_thresholds)
    tn_rates = []
    for th in thresholds:
        # thresholding
        predict_negatives = (confs < th).astype(int)
        # true negative
        tn = np.sum((predict_negatives * (1 - labels) > 0).astype(int))
        tn_rates.append(tn / np.sum(1 - labels))
    return np.array(tn_rates)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    """Calculate the IoU score between two segmentation masks
    Args:
        mask1: 1st segmentation mask
        mask2: 2nd segmentation mask
    """
    if len(mask1.shape) == 3:
        mask1 = mask1.sum(axis=-1)
    if len(mask2.shape) == 3:
        mask2 = mask2.sum(axis=-1)
    area1 = cv2.countNonZero((mask1 > 0).astype(int))
    area2 = cv2.countNonZero((mask2 > 0).astype(int))
    if area1 == 0 or area2 == 0:
        return 1
    area_union = cv2.countNonZero(((mask1 + mask2) > 0).astype(int))
    area_inter = area1 + area2 - area_union
    return area_inter / area_union


def benchmark(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    iou_meter = AverageMeter()
    iou_meter_sequence = AverageMeter()
    label_validity = []
    output_conf = []
    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            iou_meter_sequence.reset()
            label_name = os.path.join(image_folder, '0.png')
            if not os.path.exists(label_name):
                print(f'Labels are not available for {image_folder}')
                continue
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                label_name = os.path.join(image_folder, f'{idx}.png')
                image = cv2.imread(image_name)
                label = cv2.imread(label_name)
                # TODO: Modify the code below to run your method or load your results from disk
                # output, conf = my_awesome_algorithm(image)
                output = label
                conf = 1.0
                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                    iou = mask_iou(output, label)
                    iou_meter.update(conf * iou)
                    iou_meter_sequence.update(conf * iou)
                else:  # empty ground truth label
                    label_validity.append(0.0)
                output_conf.append(conf)
            # print(f'[{sequence_idx:03d}] Weighted IoU: {iou_meter_sequence.avg()}')
    tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    wiou = iou_meter.avg()
    atnr = np.mean(tn_rates)
    score = 0.7 * wiou + 0.3 * atnr
    print(f'\n\nOverall weighted IoU: {wiou:.4f}')
    print(f'Average true negative rate: {atnr:.4f}')
    print(f'Benchmark score: {score:.4f}')

    return score
