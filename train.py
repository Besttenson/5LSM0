"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torch
from torch import nn, optim
import torchvision.transforms as transforms
import wandb

class CityScapesDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        # for mask postprocessing
        valid_classes = list(filter(lambda x : x.ignore_in_eval == False, self.dataset.classes))
        self.class_names = [x.name for x in valid_classes] + ['void']
        self.id_map = {old_id : new_id for (new_id, old_id) in enumerate([x.id for x in valid_classes])}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        for cur_id in label.unique():
            cur_id = cur_id.item()
            if cur_id not in self.id_map.keys():
                label[label == cur_id] = 250 
            else:
                label[label == cur_id] = self.id_map[cur_id] 
        label[label == 250] = 19
        label = label.squeeze(0)
        return img, label

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


    
    
def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    
    img_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((256, 512)),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    target_transform = transforms.Compose([
      transforms.ToTensor(),
      # Nearest interpolation to preserve label integrity, need to *255 to 
      transforms.Resize((256, 512), interpolation=transforms.functional.InterpolationMode.NEAREST),
      transforms.Lambda(lambda x : (x * 255).to(torch.long))])
    
    # data loading
    trainset = CityScapesDatasetWrapper(Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=img_transform, target_transform=target_transform, transforms=None))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # visualize example images

    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
  
    # training/validation loop
    for epoch in range(args.epochs):
      model.train()
      for i, (images,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # save model

    torch.save(model.state_dict(),'model.path')
    # visualize some results
    wandb.log({'loss':loss})
    
    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
