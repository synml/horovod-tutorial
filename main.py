import os
import random

import filelock
import horovod.torch as hvd
import numpy as np
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision
import tqdm

import eval
import model
import train


if __name__ == '__main__':
    # Hyper parameters
    batch_size = 256
    epoch = 100
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0001
    num_workers = 4
    pin_memory = True
    amp_enabled = True
    use_fp16_compressor = True  # horovod
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

    # 3. Horovod: scaling up learning rate.
    lr_scaler = hvd.size()
    lr *= lr_scaler

    # 1. Dataset (sampler 사용)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    with filelock.FileLock('horovod.lock'):
        trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=hvd.size(), rank=hvd.rank())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, sampler=train_sampler,
                                              num_workers=num_workers, pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset, batch_size, sampler=test_sampler,
                                             num_workers=num_workers, pin_memory=pin_memory)

    # 2. Model
    with filelock.FileLock('horovod.lock'):
        model = model.LeNet().to(device)
    model_name = model.__str__().split('(')[0]

    # 3. Loss function, optimizer, scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 4. Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # (optional) Horovod: compression algorithm.
    compression = hvd.Compression.fp16 if use_fp16_compressor else hvd.Compression.none

    # 5. Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters(), compression)

    # 4. Tensorboard
    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))
        writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))
        tqdm_disabled = False
    else:
        writer = None
        tqdm_disabled = True

    # 5. Train and test
    prev_accuracy = 0
    for eph in tqdm.tqdm(range(epoch), desc='Epoch', disable=tqdm_disabled):
        trainloader.sampler.set_epoch(eph)

        train_loss, train_accuracy = train.train(model, trainloader, criterion, optimizer, scheduler, device, scaler)
        test_loss, test_accuracy = eval.evaluate(model, testloader, criterion, amp_enabled, device)

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
