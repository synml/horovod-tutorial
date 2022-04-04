import horovod.torch as hvd
import torch
import tqdm


def evaluate(model, testloader, criterion, amp_enabled, device):
    model.eval()

    test_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    for images, targets in tqdm.tqdm(testloader, desc='Eval', leave=False,
                                     disable=False if hvd.local_rank == 0 else True):
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
