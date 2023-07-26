import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from test_model import create_model

# モデルの作成
model = create_model()

# MNISTデータセットの前処理を定義
# ... 以前のコードの内容をここに追加 ...
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNISTデータセットの読み込み
# ... 以前のコードの内容をここに追加 ...
train_dataset = datasets.MNIST(root='./test_torch/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./test_torch/data', train=False, download=True, transform=transform)

# DataLoaderの作成
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 損失関数とオプティマイザの定義
# ... 以前のコードの内容をここに追加 ...
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# モデルの学習
# ... 以前のコードの内容をここに追加 ...
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

print("Finished Training")

# モデルの評価
# ... 以前のコードの内容をここに追加 ...
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total}%")