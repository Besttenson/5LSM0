import random
import imageio
import torch
from torchvision.datasets import Cityscapes
import numpy as np



class CityScapesDataset(Cityscapes):
    def __init__(self, root: str,
                 split: str = "train",
                 mode: str = "fine",
                 target_type="semantic",
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(CityScapesDataset, self).__init__(root,
                                                split,
                                                mode,
                                                target_type,
                                                transform,
                                                target_transform,
                                                transforms)
        self.means = np.array([103.939, 116.779, 123.68]) / 255.
        self.n_class = 19
        self.new_h = 256
        self.new_w = 256

    def __getitem__(self, index):
        img = imageio.imread(self.images[index], pilmode='RGB')
        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = imageio.imread(self.targets[index][i])
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]
        h, w, _ = img.shape
        top = random.randint(0, h - self.new_h)
        left = random.randint(0, w - self.new_w)
        img = img[top:top + self.new_h, left:left + self.new_w]
        label = target[top:top + self.new_h, left:left + self.new_w]

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample

    def __len__(self) -> int:
        return len(self.images)

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelTrainIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"