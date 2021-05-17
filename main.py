import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision


np.random.seed(100)
torch.manual_seed(100)
debug = True


def datasplit(ttrain, tval, root, split, csv_file, debug=False):
    """
    Split the dataset into train, test and validate
    Args:
        - ttrain: training transform
        - tval: validation/test transform
        - root_dir(str): root directory of the dataset
        - split(list): list with [train, validation, test] size in %
    """
    df = pd.read_csv(os.path.join(root, csv_file))
    df = df.sample(frac=1).reset_index(drop=True)
    split = [int(split[0] * len(df)), int((split[0] + split[1]) * len(df))]
    train_df, val_df, test_df = np.split(df, split)
    if debug:
        print(df["NumOfEllipses"].value_counts())
        print(df["NumOfPolygons"].value_counts())
        print(df[["NumOfEllipses", "NumOfPolygons"]].value_counts())
        print(train_df["NumOfEllipses"].value_counts())
        print(train_df["NumOfPolygons"].value_counts())
        print(train_df[["NumOfEllipses", "NumOfPolygons"]].value_counts())
        print(val_df["NumOfEllipses"].value_counts())
        print(val_df["NumOfPolygons"].value_counts())
        print(val_df[["NumOfEllipses", "NumOfPolygons"]].value_counts())
        print(test_df["NumOfEllipses"].value_counts())
        print(test_df["NumOfPolygons"].value_counts())
        print(test_df[["NumOfEllipses", "NumOfPolygons"]].value_counts())
    return ShapeDataset(train_df, root, ttrain),  ShapeDataset(val_df, root, tval),\
           ShapeDataset(test_df, root, tval)


class ShapeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.annotations = df
        self.img_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx, :]
        img = os.path.join(str(row["NumOfEllipses"]), row["Filename"])
        img = read_image(os.path.join(self.img_dir, img)).float()
        if self.transform:
            img = self.transform(img)
        return img, row["NumOfEllipses"], row["NumOfPolygons"]


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.dropout = nn.Dropout(.5)
        self.linear1 = nn.Linear(1000, 128)
        self.linear2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def confusion_matrix(true_label, pred_label, num_class):
    """
    Compute the confusion matrix
    Args:
        - true_label(list of int): list of true labels
        - pred_label(list of int): list of predicted labels

    Returns:
        - confusion matrix
    """
    cm = np.zeros((num_class, num_class))
    for i, j in zip(true_label, pred_label):
        cm[i, j] += 1
    return cm


def plot_confusion_matrix(conf_mat, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.

    Args:
        - conf_mat (np.array): matrix

        - classes (int): number of classes

        - normalize (bool): to apply a normalization on cm

        - title (str): title of the plot

        - cmap (plt.cm): to specify a plot cmap

    Returns:
        - axe (plt.axes): Axes object

    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        conf_mat = conf_mat.astype(
            'float') / conf_mat.sum(axis=1)[:, np.newaxis]

    fig, axe = plt.subplots()
    img = axe.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    axe.figure.colorbar(img, ax=axe)
    axe.set(xticks=np.arange(conf_mat.shape[1]),
            yticks=np.arange(conf_mat.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(axe.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    plot_cm = np.round(conf_mat, decimals=2)
    fmt = 'float' if normalize else 'int'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            axe.text(j, i, plot_cm[i, j].astype(fmt),
                     ha="center", va="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, axe


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25,
                device="cuda", writer=None, cathegory='ellipse'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {k: len(v.dataset)for k, v in dataloaders.items()}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n = 0
            for data in dataloaders[phase]:
                inputs = data[0].to(device)
                if cathegory == 'ellipse':
                    labels = labels[1].to(device)
                elif cathegory == 'polygone':
                    labels = labels[2].to(device)
                else:
                    labels = labels[1:]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                writer.add_scalar("Loss/" + phase, loss, epoch)
                n += len(labels)
                writer.add_scalar("Accuracy/" + phase,
                                  running_corrects.double() / n, epoch)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloader, device, writer=None, debug='debug.csv', cathegory='ellipse'):
    pred = []
    true = []
    running_corrects = 0
    # Iterate over data.
    n = 0
    model.eval()
    errors = {i: [] for i in range(6)}
    #invTrans = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with open(debug, 'w') as f:
        if cathegory == 'ellipse':
            f.write('ellipses, polygones, predicted ellipses\n')
        elif cathegory == 'polygone':
            f.write('polygone, ellipses, predicted polygones\n')
    for data in dataloader:
        inputs = data[0].to(device)
        if cathegory == 'ellipse':
            labels = data[1].to(device)
            others = data[2]
        elif cathegory == 'polygone':
            labels = data[2].to(device)
            others = data[1]
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels.data)
        running_corrects += torch.sum(preds == labels.data)
        n += len(labels)
        writer.add_scalar("Accuracy/test", running_corrects.double() / n, 0)
        writer.add_scalar("Batch_accuracy/test",
                          corrects.double() / len(labels), 0)
        with open(debug, 'a') as f:
            for i, img in enumerate(inputs):
                if labels[i] != preds[i]:
                    errors[int(preds[i].detach().cpu().numpy())].append(img.detach().cpu()) #invTrans(img))
                f.write(str(labels[i]) + ',' + str(others[i]) + ',' + str(preds[i]) + '\n')
            
        pred.extend(list(preds.cpu().detach().numpy()))
        true.extend(list(labels.cpu().detach().numpy()))
    for key, val in errors.items():
        if len(val):
            val = torch.stack(val)
            writer.add_image(str(key), torchvision.utils.make_grid(val), 0)
    cm = confusion_matrix(true, pred, 6)
    fg, axe = plot_confusion_matrix(cm, [str(i) for i in range(6)],
                                    normalize=False, title="Confusion Matrix")
    writer.add_figure("Confusion Matrix", fg)
    fg, axe = plot_confusion_matrix(cm, [str(i) for i in range(6)],
                                    normalize=True, title="Confusion Matrix")
    writer.add_figure("Confusion Matrix Normilized", fg)



def main():
    device = "cuda"
    t = None #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    writer = SummaryWriter()
    trainset, valset, testset = datasplit(t,
                                          t,
                                          "./shapes_dataset_HR",
                                          [.60, .10, .30],
                                          "labels.csv",
                                          True)
    dataloaders = {}
    dataloaders["val"] = DataLoader(valset, batch_size=320, shuffle=False)
    dataloaders["test"] = DataLoader(testset, batch_size=100, shuffle=False)
    dataloaders["train"] = DataLoader(trainset, batch_size=320, shuffle=True)
    model = BaselineModel()#nn.Sequential(models.resnet18(pretrained=True), nn.Linear(1000, 6))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders,
    #                       num_epochs=11, device="cuda", writer=writer, cathegory='ellipse)
    model.load_state_dict(torch.load('model.pth'))
    torch.save(model.state_dict(), 'model.pth')
    test_model(model, dataloaders["test"], device, writer, 'ellipse.csv', 'ellipse')


if __name__ == "__main__":
    main()
