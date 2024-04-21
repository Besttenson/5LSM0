from __future__ import print_function

import random
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transform
from torchvision.datasets import Cityscapes
import numpy as np
import time
import os
from argparse import ArgumentParser
from model import Model
from baseline import ModelU
from torchsummary import summary
"""def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--w_decay", type=float, default=1e-5)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--base_model", type=str, default="DeepLabv3Plus")
    parser.add_argument("--loss", type=str, default="CrossEntropy")

    return parser"""


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


def dice(pred, label):
    pred = (pred > 0).float()
    return 2. * (pred * label).sum() / (pred + label).sum()


def train(train_loader, val_loader, epochs, scheduler, optimizer, use_gpu, model, criterion, n_class, score_dir,
          IU_scores, pixel_scores):
    train_loss = []
    train_dice = []
    for epoch in range(epochs):
        scheduler.step()
        training_loss = 0
        training_dice = 0
        ts = time.time()
        for iter, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            dice_loss = dice(pred=outputs, label=labels)
            optimizer.step()

            training_loss += loss.item()
            training_dice += dice_loss.item()
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}, dice: {}".format(epoch, iter, loss.item(),dice_loss.item()))

        train_loss.append(training_loss)
        train_dice.append(training_dice)
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        np.save(os.path.join(score_dir, "training_loss"), train_loss)
        np.save(os.path.join(score_dir, "training_dice"), train_dice)
        val(epoch, model, val_loader, use_gpu, criterion, n_class, score_dir, IU_scores, pixel_scores)
    return model


def val(epoch, model, val_loader, use_gpu, criterion, n_class, score_dir, IU_scores, pixel_scores):
    val_loss = []
    val_dice = []
    model.eval()
    total_ious = []
    pixel_accs = []
    valid_loss = 0
    val_dice_loss = 0
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['Y'].cuda())
        else:
            inputs, labels = Variable(batch['X']), Variable(batch['Y'])
        output = model(inputs)
        loss = criterion(output, labels)
        dice_loss = dice(output, labels)
        valid_loss += loss.item()
        val_dice_loss += dice_loss.item()
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t, n_class))
            pixel_accs.append(pixel_acc(p, t))
    val_loss.append(valid_loss)
    val_dice.append(val_dice_loss)

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "val_acc"), pixel_scores)

    np.save(os.path.join(score_dir, "validation_loss"), val_loss)
    np.save(os.path.join(score_dir, "validation_dice"), val_dice)


# Calculates class intersections over unions
def iou(pred, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


n_class = 19
batch_size = 8
epochs = 20
lr = 1e-4
w_decay = 1e-5
step_size = 50
gamma = 0.5
model_name = "model_unet_2" + ".pth"
use_gpu = torch.cuda.is_available()

mytransformsImage = transform.Compose(
    [
        transform.ToTensor(),
        transform.Resize((256, 256))
    ]
)

mytransformsLabel = transform.Compose(
    [
        transform.ToTensor(),
    ]
)

train_dataset = CityScapesDataset(root="E:\download\data", split='train', mode='fine',
                                  target_type='semantic', transform=mytransformsImage,
                                  target_transform=mytransformsLabel)
val_dataset = CityScapesDataset(root="E:\download\data", split='val', mode='fine',
                                target_type='semantic', transform=mytransformsImage, target_transform=mytransformsLabel)
"""# Parameters
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Splitting the dataset
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])"""
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=1)

model = ModelU().cuda()
print(summary(model, (3, 224, 224)))
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores_unet_2")
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)
model = train(train_loader, val_loader, epochs, scheduler, optimizer, use_gpu, model, criterion, n_class, score_dir,
              IU_scores, pixel_scores)
torch.save(model.state_dict(), model_name)
