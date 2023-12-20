import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, device, train_loader, test_loader, criterion, optimizer, epoch, log_interval=10):
    model.train()
    matric = []
    for e in tqdm(range(epoch)):
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                line = 'Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    get_accuracy(model, device, test_loader))
                print(line)
                matric.append(line)
        return matric


def get_accuracy(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # disable gradient calculation
        for data, target in tqdm(test_loader):
            data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # count correct predictions

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))