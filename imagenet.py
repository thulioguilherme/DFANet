import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms

from torch.nn import CrossEntropyLoss

import torch.optim as optim
from torch.optim import lr_scheduler

from models.xception import XceptionA

from tqdm import tqdm

import os


# #{ calculate_topk_accuracy()

def calculate_topk_accuracy(outputs, labels, topk=(1,)):
    max_k = max(topk)
    batch_size = labels.size(0)

    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()

    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

# #}

# #{ compute_top_accuracy()

def compute_top_accuracy(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    outputs_tensor = torch.cat(all_outputs)
    labels_tensor = torch.cat(all_labels)

    top1_acc, top5_acc = calculate_topk_accuracy(outputs_tensor, labels_tensor, topk=(1, 5))

    return top1_acc, top5_acc

# #}

if __name__=='__main__':

    home_dir = os.path.expanduser('~')
    data_dir = os.path.join(home_dir, 'data/imagenet-val')
    print('Data directory:', data_dir)

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(
        root=data_dir,
        transform=data_transforms
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    num_classes = len(train_dataset.classes)
    print(f'Number of classes in ImageNet: {num_classes}')

    xception = XceptionA(1000)

    criterion = CrossEntropyLoss()

    optimizer = optim.SGD(
        xception.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=4e-5,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.94
    )

    # Training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    xception = xception.to(device)

    num_epochs = 50
    for epoch in range(num_epochs):
        # Set model to training mode
        xception.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = xception(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        print('Epoch loss: {:.4f}'.format(epoch_loss))

    print('Training finished!')

    torch.save(xception.state_dict(), 'xception.pth')
    print('Weights saved to xception.pth!')

    top1_acc, top5_acc = compute_top_accuracy(xception, train_loader, device)
    print(f'Validation Top-1 Accuracy: {top1_acc.item():.2f}%')
    print(f'Validation Top-5 Accuracy: {top5_acc.item():.2f}%')

