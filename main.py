import os
import random
import time

import horovod.torch as hvd
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.utils.tensorboard
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # log_softmax를 수행해야하지만 loss function에 포함되어 있으므로 생략.
        return x


class CustomLeNet(nn.Module):
    def __init__(self):
        super(CustomLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # log_softmax를 수행해야하지만 loss function에 포함되어 있으므로 생략.
        return x


def train(model, trainloader, criterion, optimizer, amp_enabled, device):
    model.train()

    train_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    for batch_idx, (images, targets) in enumerate(tqdm.tqdm(trainloader, desc='Train', leave=False)):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        optimizer.synchronize()
        scaler.unscale_(optimizer)
        with optimizer.skip_synchronize():
            scaler.step(optimizer)
        scaler.update()

        train_loss += loss
        pred = torch.argmax(outputs, dim=1)
        correct += torch.eq(pred, targets).sum()

    train_loss /= len(trainloader)
    train_loss = hvd.allreduce(train_loss, op=hvd.Average)

    correct = hvd.allreduce(correct, op=hvd.Sum)
    accuracy = correct / len(trainloader.dataset) * 100
    return train_loss, accuracy


def evaluate(model, testloader, criterion, amp_enabled, device):
    model.eval()

    test_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    for images, targets in tqdm.tqdm(testloader, desc='Eval', leave=False):
        images, targets = images.to(device), targets.to(device)

        with torch.cuda.amp.autocast(amp_enabled):
            with torch.no_grad():
                outputs = model(images)
            test_loss += criterion(outputs, targets)
            pred = torch.argmax(outputs, dim=1)
            correct += torch.eq(pred, targets).sum()

    test_loss /= len(testloader)
    test_loss = hvd.allreduce(test_loss, op=hvd.Average)

    correct = hvd.allreduce(correct, op=hvd.Sum)
    accuracy = correct / len(testloader.dataset) * 100
    return test_loss, accuracy


if __name__ == '__main__':
    # 0. Hyper parameters
    batch_size = 256
    epoch = 10
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001

    num_workers = 4
    pin_memory = True
    amp_enabled = True

    use_adasum = True   # horovod
    use_fp16_compressor = True  # horovod

    # Pytorch reproducibility
    reproducibility = True
    if reproducibility:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        random.seed(0)
        torch.use_deterministic_algorithms(True)

    # Horovod: initialize library
    hvd.init()
    assert hvd.is_initialized()
    local_rank = hvd.local_rank()
    torch.set_num_threads(1)    # 프로세스당 사용되는 CPU 스레드의 수를 조절 (OMP_NUM_THREADS와 동일)

    # (DEBUG) Horovod: horovod의 상태를 출력
    os.makedirs('debug', exist_ok=True)
    with open(f'debug/local_rank{local_rank}_state.txt', 'w', encoding='utf-8') as f:
        f.write(f'size: {hvd.size()}\n')
        f.write(f'local_size: {hvd.local_size()}\n')
        f.write(f'cross_size: {hvd.cross_size()}\n')
        f.write(f'rank: {hvd.rank()}\n')
        f.write(f'local_rank: {hvd.local_rank()}\n')
        f.write(f'cross_rank: {hvd.cross_rank()}\n')
        f.write(f'mpi_threads_supported: {hvd.mpi_threads_supported()}\n')
        f.write(f'mpi_enabled: {hvd.mpi_enabled()}\n')
        f.write(f'mpi_built: {hvd.mpi_built()}\n')
        f.write(f'gloo_enabled: {hvd.gloo_enabled()}\n')
        f.write(f'nccl_built: {hvd.nccl_built()}\n')
        f.write(f'ddl_built: {hvd.ddl_built()}\n')
        f.write(f'ccl_built: {hvd.ccl_built()}\n')
        f.write(f'cuda_built: {hvd.cuda_built()}\n')
        f.write(f'rocm_built: {hvd.rocm_built()}\n')

    # Horovod: scaling up learning rate.
    if use_adasum:
        if hvd.nccl_built():
            lr_scaler = hvd.local_size()
        else:
            lr_scaler = 1
    else:
        lr_scaler = hvd.size()
    lr *= lr_scaler

    # Device (local_rank 지정)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    # 1. Dataset (sampler 사용)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=hvd.size(), rank=hvd.rank())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, sampler=train_sampler,
                                              num_workers=num_workers, pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset, batch_size, sampler=test_sampler,
                                             num_workers=num_workers, pin_memory=pin_memory)

    # 2. Model
    model = LeNet().to(device)
    model_name = model.__str__().split('(')[0]

    # 3. Loss function, optimizer, scaler
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.FP16Compressor if use_fp16_compressor else hvd.Compression.NoneCompressor

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters(), compression,
                                         op=hvd.Adasum if use_adasum else hvd.Average)

    # 4. Tensorboard
    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name + time.strftime('_%y%m%d-%H%M%S')))
        writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))
        tqdm_disabled = False
    else:
        writer = None
        tqdm_disabled = True

    # 5. Train and test
    prev_accuracy = 0
    for eph in tqdm.tqdm(range(epoch), desc='Epoch', disable=tqdm_disabled):
        trainloader.sampler.set_epoch(eph)

        train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, amp_enabled, device)
        test_loss, test_accuracy = evaluate(model, testloader, criterion, amp_enabled, device)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, eph)
            writer.add_scalar('Loss/test', test_loss, eph)
            writer.add_scalars('Loss/mix', {'train': train_loss, 'test': test_loss}, eph)
            writer.add_scalar('Accuracy/train', train_accuracy, eph)
            writer.add_scalar('Accuracy/test', test_accuracy, eph)
            writer.add_scalars('Accuracy/mix', {'train': train_accuracy, 'test': test_accuracy}, eph)

        if local_rank == 0:
            # Save latest model weight
            os.makedirs('weights', exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join('weights', f'{model_name}_latest.pth'))

            # Save best accuracy model
            if test_accuracy > prev_accuracy:
                torch.save(state_dict, os.path.join('weights', f'{model_name}_best_accuracy.pth'))
                prev_accuracy = test_accuracy
    writer.close()
