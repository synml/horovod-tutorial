import horovod.torch as hvd
import torch
import tqdm


def train(model, trainloader, criterion, optimizer, device, scaler=None):
    model.train()

    train_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    for images, targets in tqdm.tqdm(trainloader, desc='Train', leave=False,
                                     disable=False if hvd.local_rank == 0 else True):
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
