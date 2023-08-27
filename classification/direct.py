import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from clip import load
from tqdm import tqdm
from constants import *

# data processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR100(root='../Datasets', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='../Datasets', train=False, download=True, transform=transform)

trainset.targets = [CIFAR20_LABELS[target] for target in trainset.targets]
testset.targets = [CIFAR20_LABELS[target] for target in testset.targets]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.clip_model, _ = load("ViT-B/32", device=device, jit=False)
        self.clip_model = self.clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(512, 20)  # CLIP的ViT-B/32模型的特征大小是512

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        return self.classifier(features)


model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()

# choose a kind of optimizer

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):
    train_loop = tqdm(trainloader, total=len(trainloader), leave=True)
    for images, labels in train_loop:
        images, labels = images.float().to(device), labels.long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loop.set_description(f"Epoch [{epoch}/{10}]")
        train_loop.set_postfix(loss=loss.item())


model.eval()
correct = 0
total = 0
with torch.no_grad():
    test_loop = tqdm(testloader, total=len(testloader), leave=True)
    for images, labels in test_loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on direct classification: {100 * correct / total}%')
