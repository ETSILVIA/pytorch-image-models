import argparse
import os
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from example_models.cifar10.vgg import VGG

from npruner.utils.compress_utils import compression

logging.basicConfig(level=logging.DEBUG)

prune_config = {
    'l1filter': {
        'dataset_name': 'mnist',
        'model_name': 'naive',
        'input_shape': [1, 1, 28, 28],
        'prune_type': 'l1'
    },
    'l1filter_vgg': {
        'dataset_name': 'cifar10',
        'model_name': 'vgg16',
        'input_shape': [1, 3, 32, 32],
        'prune_type': 'l1'
    },
    'l2filter': {
        'dataset_name': 'mnist',
        'model_name': 'naive',
        'input_shape': [1, 1, 28, 28],
        'prune_type': 'l2'
    }
}


def get_data_loaders(dataset_name='mnist', batch_size=128):
    assert dataset_name in ['cifar10', 'mnist']

    if dataset_name == 'cifar10':
        ds_class = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.MNIST
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        ds_class = datasets.MNIST
        MEAN, STD = (0.1307,), (0.3081,)

    train_loader = DataLoader(
        ds_class(
            './data', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ds_class(
            './data', train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class NaiveModelA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(1, 50, 5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(70, 50, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.fc1 = nn.Linear(2450, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=2)
        self.conv3 = nn.Conv2d(70, 50, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.fc1 = nn.Linear(2450, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.max_pool2d(x1, 2, 2)
        x = F.relu(self.bn2(self.conv2(x1)))
        x = torch.cat([x1, x], dim=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_name='naive'):
    assert model_name in ['naive', 'vgg16', 'vgg19']

    if model_name == 'naive':
        return NaiveModelA()
    elif model_name == 'vgg16':
        return VGG(16)
    else:
        return VGG(19)


def train(model, train_loader):
    model.train()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logging.info('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))
    return model


def test(model, test_loader, **kwargs):
    print(kwargs, "nothing happened to these args.")
    model.eval()
    model.cuda()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    logging.info('Loss: {}  Accuracy: {})\n'.format(test_loss, acc))
    return acc


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    model_name = prune_config[args.pruner_name]['model_name']
    dataset_name = prune_config[args.pruner_name]['dataset_name']
    train_loader, test_loader = get_data_loaders(dataset_name, args.batch_size)
    model = create_model(model_name).cuda()
    if args.resume_from is not None and os.path.exists(args.resume_from):
        logging.info('loading checkpoint {} ...'.format(args.resume_from))
        model.load_state_dict(torch.load(args.resume_from))
        test(model, test_loader)
    else:
        # STEP.1 Train from scratch
        if args.multi_gpu and torch.cuda.device_count():
            model = nn.DataParallel(model)
        logging.info('start training')
        pretrain_model_path = os.path.join(
            args.checkpoints_dir, 'pretrain_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
        for epoch in range(args.pretrain_epochs):
            train(model, train_loader)
            test(model, test_loader)
        torch.save(model.state_dict(), pretrain_model_path)

    # STEP.2 Automatic compression
    logging.info('start model pruning...')
    # pruner needs to be initialized from a model not wrapped by DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module

    dummy_input = [torch.randn(prune_config[args.pruner_name]['input_shape'])]
    exclude_layers = ['conv3']

    model = compression(model=model, val_func=test, val_loader=test_loader, dummy_input=dummy_input, ori_metric=0.99,
                        metric_thres=0.01, exclude_layers=exclude_layers, single_process_mode=False)

    pruned_model_path = os.path.join(args.checkpoints_dir,
                                     'pruned_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
    torch.save(model, pruned_model_path)

    from thop import profile
    macs, params = profile(copy.deepcopy(model).to('cpu'), inputs=dummy_input, verbose=False)
    logging.info(
        "MACs and Params after compression: MACs: {} G, Params: {} M".format(macs / 1000000000, params / 1000000))
    model = model.to(device)

    # STEP.3 Finetune
    logging.info('start finetuning')
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(args.finetune_epochs):
        # pruner.update_epoch(epoch)
        logging.info('# Epoch {} #'.format(epoch))
        train(model, train_loader)
        test(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruner_name", type=str, default="l2filter", help="pruner name")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrain_epochs", type=int, default=1, help="training epochs before model pruning")
    parser.add_argument("--finetune_epochs", type=int, default=2, help="finetuning epochs after model pruning")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="checkpoints directory")
    parser.add_argument("--resume_from", type=str, default=None, help="pretrained model weights")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs for training")

    args = parser.parse_args()
    main(args)
