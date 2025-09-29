import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms

from torch.cuda.amp import autocast, GradScaler

from torch.nn import CrossEntropyLoss

import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm

import datetime

import sys
import yaml

from pathlib import Path

# #{ include this project packages

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# #}

from models.xception import XceptionA, init_weights


# #{ read_config_file()

def read_config_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            return yaml_content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

# #}


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


# #{ generate_model_filename()

def generate_model_filename(model_name, top1_acc):
    current_datetime = datetime.datetime.now()

    formatted_accuracy = f'{top1_acc.item():.1f}'.replace('.', '_')

    datetime_str = current_datetime.strftime('%y-%m-%d_%H-%M-%S')

    filename = f'{model_name}_top1_acc_{formatted_accuracy}_{datetime_str}.pth'

    return filename

# #}


if __name__ == '__main__':

    # allow CuDNN to auto-tune
    torch.backends.cudnn.benchmark = True

    scaler = GradScaler()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} as device')

    this_file_dir = Path(__file__).resolve().parent
    configs = read_config_file(this_file_dir / '../configs/imagenet.yaml')['train']

    home_path = Path().home()
    train_dir = home_path / '../app/data/imagenet-val'
    val_dir = home_path / '../app/data/imagenet-val'

    # #{ prepare Imagenet dataset loaders

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder(
        root=train_dir,
        transform=train_transforms
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs['batch_size'] // configs['accumulation_steps'],
        shuffle=True,
        num_workers=8
    )

    num_classes = len(train_dataset.classes)
    print(f'Number of classes in ImageNet: {num_classes}')

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = ImageFolder(
        root=val_dir,
        transform=val_transforms
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=configs['batch_size'],
        shuffle=True,
        num_workers=8
    )

    # #}

    xception = XceptionA(num_classes=num_classes)
    init_weights(xception)

    criterion = CrossEntropyLoss()

    optimizer = None
    if configs['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(
            xception.parameters(),
            lr=configs['optimizer']['initial_learning_rate'],
            momentum=configs['optimizer']['momentum'],
            weight_decay=configs['optimizer']['weight_decay'],
            nesterov=True
        )
    else:
        print('Error: no valid optimizer')
        exit(1)

    xception = xception.to(device)

    num_epochs = configs['num_epochs']
    for epoch in range(num_epochs):

        if epoch == num_epochs // 2:
            print(f'Changing learning rate at epoch {epoch+1}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = configs['optimizer']['last_learning_rate']

        # #{ training loop

        xception.train()
        train_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f'(Train) Epoch {epoch+1}/{num_epochs}', unit='batch')

        for i, (inputs, labels) in enumerate(train_loader_tqdm):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = xception(inputs)
                    loss = criterion(outputs, labels)

                    if configs['accumulation_steps'] > 1:
                        loss = loss / configs['accumulation_steps']

                scaler.scale(loss).backward()

                if (i + 1) % configs['accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0) * configs['accumulation_steps']

        train_epoch_loss = train_loss / len(train_loader.dataset)
        print('  Loss: {:.4f}'.format(train_epoch_loss))

        # #}

        # #{ evaluation loop

        xception.eval()
        val_loss = 0.0

        val_loader_tqdm = tqdm(val_loader, desc=f'(Val) Epoch {epoch+1}/{num_epochs}', unit='batch')

        with torch.no_grad():
            with autocast():
                for inputs, labels in val_loader_tqdm:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = xception(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)

        val_loss_epoch = val_loss / len(val_loader.dataset)
        print('  Loss: {:.4f}'.format(val_loss_epoch))

        # #}

    print('Training finished!')

    top1_acc, top5_acc = compute_top_accuracy(xception, val_loader, device)
    print(f'Validation Top-1 Accuracy: {top1_acc.item():.2f}%')
    print(f'Validation Top-5 Accuracy: {top5_acc.item():.2f}%')

    model_filename = generate_model_filename('xception', top1_acc)
    torch.save(xception.state_dict(), this_file_dir / model_filename)
    print(f'Weights saved to {model_filename}!')
