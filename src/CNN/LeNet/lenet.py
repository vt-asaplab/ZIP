import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)

def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    elif name == "elu":
        return nn.ELU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise ValueError(f"Unsupported activation '{name}'. Choose from: gelu, elu, selu.")

class LeNetAct(nn.Module):
    def __init__(self, act_name: str):
        super().__init__()
        act = lambda: make_activation(act_name)

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.act1  = act()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.act2  = act()
        self.pool2 = nn.AvgPool2d(2, 2)

        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.act3  = act()
        self.fc2   = nn.Linear(120, 84)
        self.act4  = act()
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        return self.fc3(x)

def train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader, 1):
        data, target = data.to(device).double(), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(loader)}]  Loss: {loss.item():.4f}")

def test(model, device, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device).double(), target.to(device)
            out = model(data)
            total_loss += criterion(out, target).item() * data.size(0)
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    total_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    print(f"Test Loss: {total_loss:.4f}  Acc: {acc:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeNet on MNIST with selectable activation")
    parser.add_argument("--act", type=str, default="elu", choices=["gelu", "elu", "selu"],
                        help="activation to use: gelu (tanh approx), elu, selu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Using activation: {args.act}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    model     = LeNetAct(args.act).to(device).double()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    param_dict = {name: p.detach().cpu().numpy().tolist() for name, p in model.named_parameters()}
    out_name = f"lenet_model_params.json"
    with open(out_name, "w") as f:
        json.dump(param_dict, f, indent=2)
