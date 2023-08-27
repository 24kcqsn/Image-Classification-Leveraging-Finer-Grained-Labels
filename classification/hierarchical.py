import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import clip
from constants import *

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='../Datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.CIFAR100(root='../Datasets', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()

for param in clip_model.parameters():
    param.requires_grad = False


class Classifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(Classifier, self).__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        return self.fc(features)


model = Classifier(clip_model, 100).to(device)

criterion = nn.CrossEntropyLoss()

# choose a kind of optimizer

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
# optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # images = images.half()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

model.eval()
correct = 0
total = 0
supercorrect = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predicted_superclasses = torch.tensor(CIFAR20_LABELS[predicted.cpu().numpy()]).to(device)
        true_superclasses = torch.tensor(CIFAR20_LABELS[labels.cpu().numpy()]).to(device)

        supercorrect += (predicted_superclasses == true_superclasses).sum().item()

print(f"Accuracy on the 100 classes: {100 * correct / total:.2f}%")
print(f"Accuracy on the hierarchical classification(20 superclasses): {100 * supercorrect / total:.2f}%")
