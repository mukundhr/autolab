import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.cnn import CNN
from database import init_db, save_experiment


def run_experiment(config):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, transform=transform),
        batch_size=1000
    )

    model = CNN(config["filters"], config["num_layers"])

    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    # train for 1 epoch (fast experiments)
    model.train()

    for data, target in train_loader:
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

    # evaluate
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()

    accuracy = 100 * correct / len(test_loader.dataset)

    return accuracy

if __name__ == "__main__":

    init_db()

    config = {
        "filters": 64,
        "lr": 0.001,
        "optimizer": "adam",
        "num_layers": 2
    }

    hypothesis = "Baseline experiment"

    acc = run_experiment(config)

    save_experiment(config, acc, hypothesis, 0)

    print("Accuracy:", acc)