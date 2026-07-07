import os
import pandas as pd
import numpy as np
import torch
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import cub_root


class CustomCub2011Mask(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(
        self,
        root,
        mask_root=None,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True,
    ):

        self.root = os.path.expanduser(root)
        self.mask_root = os.path.expanduser(mask_root) if mask_root is not None else None
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def _get_mask(self, filepath):
        if self.mask_root is None:
            return np.ones((14, 14), dtype=np.float32)

        mask_path = os.path.join(self.mask_root, os.path.splitext(filepath)[0] + '.npy')
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f'Mask file not found: {mask_path}')
        return np.load(mask_path).astype(np.float32)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        # Materialize as an owned tensor to avoid DataLoader collate issues with
        # non-resizable storages created from NumPy-backed arrays.
        patch_mask = torch.tensor(self._get_mask(sample.filepath).copy(), dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx], patch_mask


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


class MergedDatasetMask(Dataset):
    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx, patch_mask = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            img, label, uq_idx, patch_mask = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        return img, label, uq_idx, np.array([labeled_or_not]), patch_mask

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


def get_cub_mask_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                          split_train_val=False, seed=0, download=False, mask_root=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCub2011Mask(
        root=cub_root,
        mask_root=mask_root,
        transform=train_transform,
        train=True,
        download=download,
    )

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCub2011Mask(
        root=cub_root,
        mask_root=mask_root,
        transform=test_transform,
        train=False,
    )

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

if __name__ == '__main__':

    x = get_cub_mask_datasets(None, None, split_train_val=False,
                              train_classes=range(100), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
