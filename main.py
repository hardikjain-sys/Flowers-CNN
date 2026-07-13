import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

print("MPS", torch.backends.mps.is_available())

data = "flower_photos"
trainTransform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valTransform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

trainData = datasets.ImageFolder(root=data, transform=trainTransform)
valData = datasets.ImageFolder(root=data, transform=valTransform)

datasetSize = len(trainData)
trainSize = int(0.8 * datasetSize)
valSize = datasetSize - trainSize

generator = torch.Generator().manual_seed(42)

trainSubset, valSubset = random_split(
    range(datasetSize),
    [trainSize, valSize],
    generator=generator
)

trainSet = torch.utils.data.Subset(trainData, trainSubset.indices)
valSet = torch.utils.data.Subset(valData, valSubset.indices)

batchSize = 32

trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False)
# images, labels = next(iter(trainLoader))

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128*16*16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x= self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=5)
# output = model(images)
# print(output.shape)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)
epochs = 25

for epoch in range(epochs):
    model.train()
    rLoss = 0.0
    correct = 0
    total = 0

    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        rLoss += loss.item()
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    trainLoss = rLoss / len(trainLoader)
    trainAcc = correct/total * 100


    model.eval()

    testCorrect = 0
    testTotal = 0
    testLoss = 0.0

    with torch.no_grad():
        for images, labels in valLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            testLoss += loss.item()
            _, predicted = torch.max(outputs, 1)
            testTotal += labels.size(0)
            testCorrect += (predicted == labels).sum().item()


    testLoss = testLoss / len(valLoader)
    testAcc = testCorrect / testTotal * 100

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss {trainLoss}, Train Acc {trainAcc}%,"
          f"Val Loss {testLoss}, Val Acc {testAcc}%")

torch.save(model.state_dict(), "flowerSaved.pth")
print("Model saved")


###################################

testTransform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

testData = datasets.ImageFolder(root="testPhotos", transform=testTransform)
testLoader = DataLoader(testData, batch_size=32, shuffle=False)

model = CNN(num_classes=5)
model.load_state_dict(torch.load("flowerSaved.pth", map_location=device))
model = model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            true_label = testData.classes[labels[i]]
            pred_label = testData.classes[predicted[i]]
            print(f" True: {true_label}, Predicted {pred_label}")

testAcc = correct / total * 100
print(f"\n unseen images accuracy: {testAcc}%")

