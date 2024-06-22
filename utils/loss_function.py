import torch as t
import torch.nn as nn
import numpy as np
from skimage.transform import resize


class SaliencyLoss(nn.Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, preds, labels, loss_type='cc'):
        losses = []
        if loss_type == 'cc':
            for i in range(labels.shape[0]):  # labels.shape[0] is batch size
                loss = loss_CC(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'kldiv':
            for i in range(labels.shape[0]):
                loss = loss_KLdiv(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'sim':
            for i in range(labels.shape[0]):
                loss = loss_similarity(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'nss':
            for i in range(labels.shape[0]):
                loss = loss_NSS(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'auc':
            for i in range(labels.shape[0]):
                loss = AUC_Judd(preds[i], labels[i])
                loss_tensor = t.tensor(loss, dtype=t.float64, device="cuda:0")  # Convert to tensor
                losses.append(loss_tensor)

        return t.stack(losses).mean(dim=0, keepdim=True)


def loss_KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    pred_map = pred_map / t.sum(pred_map)
    gt_map = gt_map / t.sum(gt_map)
    div = t.sum(t.mul(gt_map, t.log(eps + t.div(gt_map, pred_map + eps))))
    return div


def loss_CC(pred_map, gt_map):
    gt_map_ = (gt_map - t.mean(gt_map))
    pred_map_ = (pred_map - t.mean(pred_map))
    cc = t.sum(t.mul(gt_map_, pred_map_)) / t.sqrt(t.sum(t.mul(gt_map_, gt_map_)) * t.sum(t.mul(pred_map_, pred_map_)))
    return cc


def loss_similarity(pred_map, gt_map):
    gt_map = (gt_map - t.min(gt_map)) / (t.max(gt_map) - t.min(gt_map))
    gt_map = gt_map / t.sum(gt_map)

    pred_map = (pred_map - t.min(pred_map)) / (t.max(pred_map) - t.min(pred_map))
    pred_map = pred_map / t.sum(pred_map)

    diff = t.min(gt_map, pred_map)
    score = t.sum(diff)

    return score


def loss_NSS(pred_map, fix_map):
    '''ground truth here is a fixation map'''

    pred_map_ = (pred_map - t.mean(pred_map)) / t.std(pred_map)

    # Convert the fixation map to a binary mask
    fix_map_binary = fix_map > 0

    score = t.mean(t.masked_select(pred_map_, fix_map_binary))
    return score


def normalize(x, method='standard', axis=None):
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def AUC_Judd(saliency_map, fixation_map, jitter=True):
    saliency_map = saliency_map.cpu().numpy() if saliency_map.is_cuda else saliency_map.numpy()
    fixation_map = fixation_map.cpu().numpy() > 0.5 if fixation_map.is_cuda else (fixation_map.numpy() > 0.5)
    # If there are no fixations to predict, return NaN
    if not np.any(fixation_map):
        print('No fixations to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        # Generate random numbers in the same shape as saliency_map as float64
        random_values = np.random.rand(*saliency_map.shape).astype(np.float64)
        saliency_map = saliency_map.astype(np.float64) + random_values * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio of saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio of other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x