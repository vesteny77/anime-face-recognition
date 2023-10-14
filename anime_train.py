import zipfile
import os, os.path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from pathlib import Path
import time
import copy

# import pytorch
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models

num_classes = 1000

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def print_opts(opts):
    """
    Print all the parameters in opts before training starts.
    """
    print('=' * 79)
    print('Opts'.center(79))
    print('-' * 79)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(79))
    print('=' * 79)


def get_anime_loader(opts):
    """
    Retunrs a dictionary containing two key-value pairs:
        'train': training dataloader
        'test': validation dataloader
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(opts.image_size),
            transforms.CenterCrop(opts.crop_size),
            transforms.RandomHorizontalFlip(p=opts.flip_prob),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(opts.image_size),
            transforms.CenterCrop(opts.crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    dataset_path = "dataset/"
    image_datasets = {category: datasets.ImageFolder(os.path.join(dataset_path, category),
                                                     data_transforms[category])
                      for category in ['train', 'test']}
    
    dataloader_dict = {category: torch.utils.data.DataLoader(image_datasets[category],
                                                         batch_size=opts.batch_size,
                                                         shuffle=True,
                                                         num_workers=opts.num_workers)
                   for category in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print(f"dataset_sizes = {dataset_sizes}")
    print(f"Total number of images = {dataset_sizes['train'] + dataset_sizes['test']}")
    # class_names = image_datasets['train'].classes
    # print(f"class_name 0 = {class_names[0]}")
    
    return dataloader_dict

def get_dataset_sizes(opts):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(opts.image_size),
            transforms.CenterCrop(opts.crop_size),
            transforms.RandomHorizontalFlip(p=opts.flip_prob),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(opts.image_size),
            transforms.CenterCrop(opts.crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    dataset_path = "dataset/"
    image_datasets = {category: datasets.ImageFolder(os.path.join(dataset_path, category),
                                                     data_transforms[category])
                      for category in ['train', 'test']}
    return {x: len(image_datasets[x]) for x in ['train', 'test']}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_training_curve_loss(train_losses, valid_losses, save_path):
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_path + "/resnet_pretrained_training_curve_loss.jpg")

    
def plot_training_curve_acc(train_acc, valid_acc, save_path):
    plt.figure()
    plt.plot(train_acc, "ro-", label="Train")
    plt.plot(valid_acc, "go-", label="Validation")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(save_path + "/resnet_pretrained_training_curve_acc.jpg"))

    
def googlenet_train_loop(model, criterion, optimizer, scheduler, epochs, dataloader_dict, dataset_sizes):
    train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('=' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs, aux_outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) # loss function

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_losses.append(epoch_loss)
                valid_acc.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # save model weights
    print("Saving Model...")
    torch.save(best_model_wts, 'googlenet_scratch_weights.pth')
    
    # plot training curve
    print("Plotting Training Curve...")
    plot_training_curve_loss(train_losses, valid_losses, "plots")
    return model


def resnet_train_loop(model, criterion, optimizer, scheduler, epochs, dataloader_dict, dataset_sizes):
    train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('=' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) # loss function

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_losses.append(epoch_loss)
                valid_acc.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            print(f"Saving Resnet Model Weights for Epoch {epoch}...")
            torch.save(model.state_dict(), f'resnet_all_weights/resnet_pretrained_weights_epoch_{epoch}.pth')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # save model weights
    print("Saving Best Model...")
    torch.save(best_model_wts, 'resnet_all_weights/resnet_best_pretrained_weights.pth')
    
    # plot training curve
    # print("Plotting Training Curve...")
    # plot_training_curve_loss(train_losses, valid_losses, "plots")
    return model

# identity function for fine-tuning with arcface
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    

args = AttrDict()
args_dict = {
    'image_size': 256,  # 299
    'crop_size': 224, # 299
    'flip_prob': 0.3,
    'batch_size': 32,
    'num_workers': 2,
    'epochs': 10,
    'lr': 1e-3,
    'model': 'ResNet',
    'loss_function': 'CrossEntropyLoss',
    'optimizer': 'Adam',  # 'Adam', 'SGD'
    'step_size': 7,  # for learning rate scheduler
    'gamma': 0.1,  # for learning rate scheduler
    'requires_grad': False
}

args.update(args_dict)
print("Loading dataset...")
dataset_sizes = get_dataset_sizes(args)
dataloader_dict = get_anime_loader(args)

def train(opts):
    train_model, criterion, optimizer, scheduler = None, None, None, None
    
    if opts.model.lower() == 'googlenet':
        train_model = models.inception_v3(pretrained=True)
    else:
        train_model = models.resnet50(pretrained=True)
    
    for param in train_model.parameters():
        param.requires_grad = opts.requires_grad
        
    num_in_features = train_model.fc.in_features
    
    # Handle the auxilary net (only for googlenet)
    if opts.model == 'GoogLeNet':
        aux_num_ftrs = train_model.AuxLogits.fc.in_features
        train_model.AuxLogits.fc = nn.Linear(aux_num_ftrs, num_classes)
    
    # Handle the primary net (same for both googlenet and resnet)
    primary_num_ftrs = train_model.fc.in_features
    train_model.fc = nn.Linear(primary_num_ftrs, num_classes)
    
    if opts.loss_function == 'CrossEntropyLoss':
        train_model.fc = nn.Linear(num_in_features, num_classes)
        criterion = nn.CrossEntropyLoss()
    else:
        train_model.fc = nn.Linear(num_in_features, num_classes)
        criterion = AngularPenaltySMLoss(loss_type=opts.loss_function)
        criterion = criterion.to(device)
        
    if opts.optimizer == 'SGD':
        optimizer = optim.SGD(train_model.parameters(), lr=opts.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(train_model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
    
    # move model to GPU
    train_model = train_model.to(device)
    print(f"Model moved to GPU: {device}")
    
    if opts.model.lower() == 'googlenet':
        print("Training GoogLeNet...")
        googlenet_train_loop(train_model, criterion, optimizer,
                             scheduler, opts.epochs, dataloader_dict, dataset_sizes)
    else:
        print("Training ResNet...")
        resnet_train_loop(train_model, criterion, optimizer,
                          scheduler, opts.epochs, dataloader_dict, dataset_sizes)

print_opts(args)
train(args)
