
from loss_function import dice, iou, pixel_acc
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import numpy as np
import time
import os
from argparse import ArgumentParser
from model import Model
from baseline import ModelU
from preprocessing import CityScapesDataset
from torchsummary import summary

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Root directory of dataset where directory leftImg8bit and gtFine or gtCoarse are located")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--w_decay", type=float, default=1e-5)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--base_model", type=str, default="DeepLabv3Plus", help="DeepLabv3Plus or baseline")
    parser.add_argument("--loss", type=str, default="CrossEntropy", help="CrossEntropy or BCE")

    return parser

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

    model.eval()

    val_loss = []
    val_dice = []
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




def main(args):
    n_class = 19
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    w_decay = args.w_decay
    step_size = args.step_size
    gamma = args.gamma
    model_name = "model_" + args.base_model + ".pth"
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

    train_dataset = CityScapesDataset(root=args.data_path, split='train', mode='fine',
                                      target_type='semantic', transform=mytransformsImage,
                                      target_transform=mytransformsLabel)
    val_dataset = CityScapesDataset(root=args.data_path, split='val', mode='fine',
                                    target_type='semantic', transform=mytransformsImage, target_transform=mytransformsLabel)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=1)

    if args.base_model == "DeepLabv3Plus":
        model = Model().cuda()
    elif args.base_model == "baseline":
        model = ModelU().cuda()

    print(summary(model, (3, 224, 224)))

    if args.loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "BCE":
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                    gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    # create dir for score
    score_dir = os.path.join(args.base_model)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    IU_scores = np.zeros((epochs, n_class))
    pixel_scores = np.zeros(epochs)
    model = train(train_loader, val_loader, epochs, scheduler, optimizer, use_gpu, model, criterion, n_class, score_dir,
                  IU_scores, pixel_scores)
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
