"""
This module provides access to the FER (Facial Emotion Recognition)
as encapsulated as a `torchvision.datasets.VisionDataset` class.

Notes
-----
The FER dataset [1]_ consists of `48x48` pixel grayscale images of faces.
The faces have been automatically registered so that the face is more or less
centered and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial
expression in to one of seven categories
`(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)`.

These are the overall statistics of the Dataset, per emotion.

.. list-table:: Dataset Overall Statistics
    :widths: 25 25 50
    :header-rows: 1

    * - Emotion
      - Emotion Label
      - Count
    * - 0
      - Angry
      - 4,593
    * - 1
      - Disgust
      - 547
    * - 2
      - Fear
      - 5,121
    * - 3
      - Happy
      - 8,989
    * - 4
      - Sad
      - 6,077
    * - 5
      - Surprise
      - 4,002
    * - 6
      - Neutral
      - 6,198

Samples distributions per Data partitions, and per-emotions are reported below.

.. list-table:: Data partitions statistics
    :widths: 33 33 34
    :header-rows: 1

    * - Training
      - Validation
      - Test
    * - 28,709
      - 3589
      - 3,589


The distribution of the samples per single emotion for each of the three
considered data data_partition is show in the barplot below:

.. image:: images/fer_data_partitions_distributions.png
    :width: 400
    :alt: Samples distribution per-emotion among the three data partitions


References
-----------
.. [1]  I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B. Hamner,
   W. Cukierski, Y. Tang, D. Thaler, D.-H. Lee, Y. Zhou, C. Ramaiah, F. Feng,
   R. Li, X. Wang, D. Athanasakis, J. Shawe-Taylor, M. Milakov, J. Park, R. Ionescu,
   M. Popescu, C. Grozea, J. Bergstra, J. Xie, L. Romaszko, B. Xu,
   Z. Chuang, and Y. Bengio. "Challenges in representation learning: A report on three
   machine learning contests". Neural Networks, 64:59--63, 2015.
   Special Issue on Deep Learning of Representations
"""


import os
from typing import Tuple, Optional, Callable
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image


class FER(Dataset):
    """`FER` (Facial Emotion Recognition) Dataset

    Attributes
    ----------
    root : str
        Root directory where the local copy of dataset is stored.
    split : {"train", "validation", "test"} (default: "train")
        Target data data_partition. Three data partitions are available, namely
        "training", "validation", and "test". Training data_partition is considered
        by default.
    download :  bool, optional (False)
        If true, the dataset will be downloaded from the internet and saved in the root
        directory. If dataset is already downloaded, it is not downloaded again.
    """

    RAW_DATA_FILE = "fer2013.csv"
    RAW_DATA_FOLDER = "fer2013"

    resources = [
        (
            "https://www.dropbox.com/s/2rehtpc6b5mj9y3/fer2013.tar.gz?dl=1",
            "ca95d94fe42f6ce65aaae694d18c628a",
        )
    ]

    # torch Tensor filename
    data_files = {
        "train": "training.pt",
        "validation": "validation.pt",
        "test": "test.pt",
    }

    classes = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    def __init__(self, root: str, split: str = "train", download: bool = False):
        self.root = root
        split = split.strip().lower()
        if split not in self.data_files:
            raise ValueError(
                "Data Partition not recognised. Accepted values are 'train', 'validation', 'test'."
            )
        if download:
            self.download()  # download, preprocess, and store FER data
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        self.split = split
        data_file = self.data_files[self.split]
        data_filepath = self.processed_folder / data_file
        # load serialisation of dataset as torch.Tensors
        self.data, self.targets = torch.load(data_filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """

        Parameters
        ----------
        index : int
            Index of the sample

        Returns
        -------
        tuple
            (Image, Target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = torch.unsqueeze(img, 0)
        return img, target

    def _check_exists(self):
        for data_fname in self.data_files.values():
            data_file = self.processed_folder / data_fname
            if not data_file.exists():
                return False
        return True

    def download(self):
        """Download the FER data if it doesn't already exist in the processed folder"""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[-1].split("?")[0]
            download_and_extract_archive(
                url, download_root=str(self.raw_folder), filename=filename, md5=md5
            )

        # process and save as torch files
        def _set_partition(label: str) -> str:
            if label == "Training":
                return "train"
            if label == "PrivateTest":
                return "validation"
            return "test"

        print("Processing...", end="")
        raw_data_filepath = self.raw_folder / self.RAW_DATA_FOLDER / self.RAW_DATA_FILE
        fer_df = pd.read_csv(
            raw_data_filepath, header=0, names=["emotion", "pixels", "partition"]
        )
        fer_df["partition"] = fer_df.partition.apply(_set_partition)
        fer_df.emotion = pd.Categorical(fer_df.emotion)

        for partition in ("train", "validation", "test"):
            dataset = fer_df[fer_df["partition"] == partition]
            images = self._images_as_torch_tensors(dataset)
            labels = self._labels_as_torch_tensors(dataset)
            data_file = self.processed_folder / self.data_files[partition]
            with open(data_file, "wb") as f:
                torch.save((images, labels), f)
        print("Done!")

    def _images_as_torch_tensors(self, dataset: pd.DataFrame) -> torch.Tensor:
        """
        Extract all the pixel from the input dataframes, and convert images in
        a [sample x features] torch.Tensor

        Parameters
        ----------
        dataset : pd.DataFrame
            The target dataset data_partition (i.e. training, validation, or test)
            as extracted from the original dataset
        Returns
        -------
        torch.Tensor
            [sample x pixels] tensor representing the whole data data_partition as
            torch Tensor.
        """
        imgs_np = (dataset.pixels.map(self._to_numpy)).values
        imgs_np = np.array(np.concatenate(imgs_np, axis=0))
        samples_no, pixels = imgs_np.shape
        new_shape = (samples_no, int(sqrt(pixels)), int(sqrt(pixels)))
        return torch.from_numpy(imgs_np).view(new_shape)

    @staticmethod
    def _labels_as_torch_tensors(dataset: pd.DataFrame) -> torch.Tensor:
        """Extract labels from pd.Series and convert into torch.Tensor"""
        labels_np = dataset.emotion.values.astype(np.int)
        return torch.from_numpy(labels_np)

    @staticmethod
    def _to_numpy(pixels: str) -> np.ndarray:
        """Convert one-line string pixels into NumPy array, adding the first
        extra axis (sample dimension) later used as the concatenation axis"""
        img_array = np.fromstring(pixels, dtype="uint8", sep=" ")[np.newaxis, ...]
        return img_array

    # Properties
    # for public API
    # --------------

    @property
    def processed_folder(self):
        return Path(self.root) / self.__class__.__name__ / "processed"

    @property
    def raw_folder(self):
        return Path(self.root) / self.__class__.__name__ / "raw"

    @property
    def partition(self):
        return self.split

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def idx_to_class(self):
        return {v: k for k, v in self.class_to_idx.items()}

    @staticmethod
    def classes_map():
        return {i: c for i, c in enumerate(FER.classes)}


class FERVision(FER, VisionDataset):
    def __init__(
        self,
        root: str = ".",
        download: bool = True,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        FER.__init__(self, root=root, download=download, split=split)
        VisionDataset.__init__(self, root=root, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, target
