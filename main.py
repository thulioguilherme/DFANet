from cityscape import DatasetTrain ,DatasetVal
import argparse
from torch.utils.data import  DataLoader
from pathlib import Path
import yaml
from trainer import Trainer
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler

from loss import CrossEntropyLoss2d

import os

from models.dfanet import DFANet, load_dfanet_backbone_weights
from config.settings import DFANetConfig


if __name__=='__main__':

    # #{ Load dataset

    home_dir = os.path.expanduser('~')
    data_dir= os.path.join(home_dir, 'data/cityscapes')
    print('Data directory:', data_dir)

    train_dataset = DatasetTrain(
        cityscapes_data_path=data_dir,
        cityscapes_meta_path=data_dir
    )

    val_dataset = DatasetVal(
        cityscapes_data_path=data_dir,
        cityscapes_meta_path=data_dir
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )

    # #}

    # #{ Create model and prepare training

    dfanet_config = DFANetConfig()
    dfanet = DFANet(dfanet_config)

    load_dfanet_backbone_weights(dfanet, backbone_weights_path='xception.pth')

    # Cross-entropy error at each pixel over the categories as loss function
    loss_function = CrossEntropyLoss(ignore_index=255)

    # Mini-batch stochastic gradient descent (SGD) with momentum of 0.9 and weight decay of 1e-5
    optimizer = optim.SGD(
        dfanet.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.00001
    )

    # Poly learning rate policy where initial rate is multiplied by (1 - iter/max_iter)^power
    max_iter = 100
    power = 0.9
    lr_fc = lambda iteration: (1 - iteration/40000000)**0.9

    # Base learning rate is set as 2e-1 and power is 0.9
    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lr_fc,-1)

    # #}

    trainer = Trainer('training', optimizer, exp_lr_scheduler, dfanet, dfanet_config, './log')
    trainer.train(train_loader, val_loader, loss_function, max_iter)
    trainer.evaluate(val_loader)

    print('Finished training!')
