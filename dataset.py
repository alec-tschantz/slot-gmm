from PIL import Image
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, h5_path: str, split: str = "train", masks: bool = False, factors: bool = False):
        super().__init__()
        self.h5_path = str(Path(h5_path))
        self.split = split
        self.masks = masks
        self.factors = factors

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for training.

        Args:
            img (np.ndarray): [H,W,C] image.

        Returns:
            np.ndarray: [C,H,W] image.
        """
        img = Image.fromarray(np.uint8(img))
        img = np.transpose(np.array(img), (2, 0, 1))

        img = img / 255.0
        img = (img * 2) - 1
        return img

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            int: Length of dataset.
        """
        with h5py.File(self.h5_path, "r") as data:
            data_size, _, _, _ = data[self.split]["imgs"].shape
            return data_size

    def __getitem__(self, i: int) -> dict:
        """Get item from dataset.

        Args:
            i (int): Index of item.

        Returns:
            dict: Dictionary of item. Keys are "imgs", "masks", and "factors".
        """
        with h5py.File(self.h5_path, "r") as data:
            outs = {}
            outs["image"] = self.preprocess(data[self.split]["imgs"][i].astype("float32")).astype("float32")
            if self.masks:
                outs["masks"] = np.transpose(data[self.split]["masks"][i].astype("float32"), (0, 3, 1, 2))
            if self.factors:
                outs["factors"] = data[self.split]["factors"][i]
            return outs
