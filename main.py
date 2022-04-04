import os
import random
import time

import filelock
import horovod.torch as hvd
import numpy as np
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision
import tqdm


def train(model, trainloader, criterion, optimizer, device, scaler=None):
    model.train()

    train_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    for images, targets in tqdm.tqdm(trainloader, desc='Train', leave=False,
                                     disable=False if hvd.local_rank() == 0 else True):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True if scaler is not None else False):
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
    return train_loss.item(), accuracy.item()


def evaluate(model, testloader, criterion, amp_enabled, device):
    model.eval()

    test_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    for images, targets in tqdm.tqdm(testloader, desc='Eval', leave=False,
                                     disable=False if hvd.local_rank() == 0 else True):
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
    return test_loss.item(), accuracy.item()


if __name__ == '__main__':
    # Hyper parameters
    batch_size = 256
    epoch = 5
    lr = 0.1
    momentum = 0.9
    weight_decay = 0
    num_workers = 4
    pin_memory = True
    amp_enabled = False
    use_fp16_compressor = False  # horovod
    reproducibility = True

    # Pytorch reproducibility
    if reproducibility:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        random.seed(0)

    # 1. Horovod: initialize library
    hvd.init()
    assert hvd.is_initialized()
    local_rank = hvd.local_rank()
    torch.set_num_threads(1)    # 프로세스당 사용되는 CPU 스레드의 수를 조절 (OMP_NUM_THREADS와 동일)

    # 2. Horovod: local_rank로 GPU 고정
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    # 3. Dataset (sampler 사용)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    with filelock.FileLock('horovod.lock'):
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=hvd.size(), rank=hvd.rank())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, sampler=train_sampler,
                                              num_workers=num_workers, pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset, batch_size, sampler=test_sampler,
                                             num_workers=num_workers, pin_memory=pin_memory)

    # Model
    with filelock.FileLock('horovod.lock'):
        model = torchvision.models.resnet101(num_classes=10).to(device)
    model_name = model.__str__().split('(')[0]

    # Loss function, optimizer, scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 4. Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # (optional) Horovod: compression algorithm.
    compression = hvd.Compression.fp16 if use_fp16_compressor else hvd.Compression.none

    # 5. Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters(), compression)

    # Tensorboard
    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))
        writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))
        tqdm_disabled = False
    else:
        writer = None
        tqdm_disabled = True

    # Train and test
    prev_accuracy = 0
    images_per_sec = []
    for eph in tqdm.tqdm(range(epoch), desc='Epoch', disable=tqdm_disabled):
        trainloader.sampler.set_epoch(eph)

        epoch_time = time.time()
        train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device, scaler)
        epoch_time = time.time() - epoch_time
        images_per_sec.append(str(round(len(trainloader.dataset) / epoch_time)) + '\n')

        test_loss, test_accuracy = evaluate(model, testloader, criterion, amp_enabled, device)
        scheduler.step()

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

    if writer is not None:
        writer.close()

    if local_rank == 0:
        with open(f'np{hvd.size()}_images_per_epoch.txt', 'w', encoding='utf-8') as f:
            f.writelines(images_per_sec)
