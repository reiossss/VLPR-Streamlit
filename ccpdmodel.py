import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transform import provinces, alphabets, ads


# 定义数据集类
class CCPDDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        self.imgs = []
        self.labels = []
        with open(os.path.join(root_dir, phase, 'rec.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                img_path, label = line.strip().split('\t')
                self.imgs.append(os.path.join(root_dir, img_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 将标签转换为索引
        province_idx = provinces.index(label[0])
        alphabet_idx = alphabets.index(label[1])
        ads_indices = [ads.index(ch) for ch in label[2:]]

        label_tensor = torch.tensor([province_idx, alphabet_idx] + ads_indices)

        return image, label_tensor


# 数据转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 加载数据集
root_dir = 'C:/Users/peter/Desktop/plate_recognize/CCPD2020/PPOCR'
train_dataset = CCPDDataset(root_dir, 'train', transform=transform)
val_dataset = CCPDDataset(root_dir, 'val', transform=transform)
test_dataset = CCPDDataset(root_dir, 'test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义模型
class CCPDModel(nn.Module):
    def __init__(self):
        super(CCPDModel, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.base_model = models.resnet18(weights=weights)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 65 * 8)  # 8个字符，每个字符有65个类别

    def forward(self, x):
        x = self.base_model(x)
        return x.view(x.size(0), 8, 65)  # 将输出调整为 (batch_size, 7, 65)


model = CCPDModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # 记录损失和准确率
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = 0
            corrects = 0
            for i in range(8):
                loss += criterion(outputs[:, i, :], labels[:, i])
                _, preds = torch.max(outputs[:, i, :], 1)
                corrects += torch.sum(preds == labels[:, i])

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += corrects

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / (len(train_loader.dataset) * 8)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                loss = 0
                corrects = 0
                for i in range(8):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                    _, preds = torch.max(outputs[:, i, :], 1)
                    corrects += torch.sum(preds == labels[:, i])

            val_loss += loss.item() * inputs.size(0)
            val_corrects += corrects

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / (len(val_loader.dataset) * 8)

        val_losses.append(val_loss)
        val_accs.append(val_acc.item())

        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    # 画损失曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 画准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accs, label='Train Accuracy')
    plt.plot(range(num_epochs), val_accs, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model


# 训练模型
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

# 保存模型
torch.save(model.state_dict(), 'ccpd_model.pth')


# 测试函数
def evaluate_model(model, test_loader):
    model.eval()
    test_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            corrects = 0
            for i in range(8):
                _, preds = torch.max(outputs[:, i, :], 1)
                corrects += torch.sum(preds == labels[:, i])

        test_corrects += corrects

    test_acc = test_corrects.double() / (len(test_loader.dataset) * 8)
    print('Test Acc: {:.4f}'.format(test_acc))


# 测试模型
evaluate_model(model, test_loader)
