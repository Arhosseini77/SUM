import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def preprocess_img(img_dir, channels=3):

    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)

    image_org = img
    shape_r = 256
    shape_c = 256
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
        :] = img

    return img_padded , image_org


def postprocess_img(pred, org_dir):
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img


class TrainDataset(Dataset):
    def __init__(self, datasets_info, transform=None, num_classes=4):
        self.datasets = []
        self.num_classes = num_classes
        for dataset_info in datasets_info:
            ids = pd.read_csv(dataset_info['id_train'])
            self.datasets.append((ids, dataset_info, transform))

    def __len__(self):
        return sum(len(ids) for ids, _, _ in self.datasets)

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= len(self.datasets[dataset_idx][0]):
            idx -= len(self.datasets[dataset_idx][0])
            dataset_idx += 1
        ids, dataset_info, transform = self.datasets[dataset_idx]

        # Load image
        im_path = dataset_info['stimuli_dir'] + ids.iloc[idx, 0]
        image = Image.open(im_path).convert('RGB')
        if transform:
            image = transform(image)

        # Load saliency map
        smap_path = dataset_info['saliency_dir'] + ids.iloc[idx, 1]
        saliency = Image.open(smap_path).convert('L')
        saliency = np.array(saliency, dtype=np.float32) / 255.
        saliency = torch.from_numpy(saliency).unsqueeze(0)

        # Load fixation map
        fmap_path = dataset_info['fixation_dir'] + ids.iloc[idx, 2]
        fixation = Image.open(fmap_path).convert('L')
        fixation = np.array(fixation, dtype=np.float32) / 255.
        fixation = torch.from_numpy(fixation).unsqueeze(0)

        # Convert label to one-hot vector
        label = torch.zeros(self.num_classes)
        label[dataset_info['label']] = 1

        sample = {'image': image, 'saliency': saliency, 'fixation': fixation, 'label': label}
        return sample

class ValDataset(Dataset):
    def __init__(self, ids_path, stimuli_dir, saliency_dir, fixation_dir, label, transform=None, num_classes=4):
        self.ids = pd.read_csv(ids_path)
        self.stimuli_dir = stimuli_dir
        self.saliency_dir = saliency_dir
        self.fixation_dir = fixation_dir
        self.label = label
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Load image
        im_path = self.stimuli_dir + self.ids.iloc[idx, 0]
        image = Image.open(im_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load saliency map
        smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        saliency = Image.open(smap_path).convert('L')
        saliency = np.array(saliency, dtype=np.float32) / 255.
        saliency = torch.from_numpy(saliency).unsqueeze(0)

        # Load fixation map
        fmap_path = self.fixation_dir + self.ids.iloc[idx, 2]
        fixation = Image.open(fmap_path).convert('L')
        fixation = np.array(fixation, dtype=np.float32) / 255.
        fixation = torch.from_numpy(fixation).unsqueeze(0)

        # Convert label to one-hot vector
        label = torch.zeros(self.num_classes)
        label[self.label] = 1

        sample = {'image': image, 'saliency': saliency, 'fixation': fixation, 'label': label}
        return sample
