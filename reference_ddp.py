import argparse
import os
import time

from filelock import FileLock

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import torch.distributed

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                    help='use mixed precision for training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def train_mixed_precision(epoch, scaler):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = F.cross_entropy(output, target)

        scaler.scale(loss).backward()
        # Make sure all async allreduces are done
        optimizer.synchronize()
        # In-place unscaling of all gradients before weights update
        scaler.unscale_(optimizer)
        with optimizer.skip_synchronize():
            scaler.step(optimizer)
        # Update scaler in case of overflow/underflow
        scaler.update()

        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss Scale: {}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                       100. * batch_idx / len(train_loader), loss.item(), scaler.get_scale()))


def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Distributed Data-Parallel Training (DDP)
    assert torch.distributed.is_available(), '"torch.distributed" package is not available.'
    assert torch.distributed.is_torchelastic_launched(), 'Run the python process with "torch.distributed.run".'
    assert torch.distributed.is_nccl_available(), 'NCCL backend is not available.'
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    assert torch.distributed.is_initialized()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(local_rank)
    else:
        if args.use_mixed_precision:
            raise ValueError("Mixed precision is only supported with cuda enabled.")

    if args.use_mixed_precision and int(torch.__version__.split('.')[1]) < 6:
        raise ValueError("Mixed precision is using torch.cuda.amp.autocast(), which requires torch >= 1.6.0")

    # Horovod: limit # of CPU threads to be used per worker. (OMP_NUM_THREADS와 동일)
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    data_dir = 'data'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
    )

    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
    )

    model = Net().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = world_size if not args.use_adasum else 1

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum)

    scaler = torch.cuda.amp.GradScaler()
    images_per_sec = []
    for epoch in range(1, args.epochs + 1):
        epoch_time = time.time()
        if args.use_mixed_precision:
            train_mixed_precision(epoch, scaler)
        else:
            train(epoch)
        epoch_time = time.time() - epoch_time
        images_per_sec.append(str(round(len(train_loader.dataset) / epoch_time)) + '\n')
        # Keep test in full precision since computation is relatively light.
        test()

    if local_rank == 0:
        with open(f'np{world_size}_images_per_epoch.txt', 'w', encoding='utf-8') as f:
            f.writelines(images_per_sec)
