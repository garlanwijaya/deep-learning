# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import time
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm # Menggunakan tqdm standar, bukan dari notebook
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# =============================================================================
# 2. DEFINISI KELAS (DATASET & MODEL)
# =============================================================================

class MakananDataset(Dataset):
    """
    Dataset kustom untuk memuat gambar makanan dan labelnya.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.img_labels['label'].unique()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = str(self.img_labels.loc[idx, 'filename'])
        img_path = os.path.join(str(self.img_dir), img_name)
        image = Image.open(img_path).convert('RGB')
        label_name = self.img_labels.loc[idx, 'label']
        label = self.class_to_idx[label_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Model Arsitektur Plain-34 (Tanpa Residual Connection) ---
class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        # Tidak ada residual connection (out += identity)
        out = F.relu(out)
        return out

class Plain34(nn.Module):
    def __init__(self, num_classes=5):
        super(Plain34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [PlainBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Model Arsitektur ResNet-34 (Dengan Residual Connection) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        # Residual connection ditambahkan di sini
        out += identity
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# =============================================================================
# 3. FUNGSI TRAINING
# =============================================================================

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, epochs, device):
    """
    Fungsi generik untuk melatih model PyTorch.
    """
    print(f"\n🔥 Memulai Training Model {model_name}...")
    start_time = time.time()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Fase Training
        model.train()
        train_loss, train_correct = 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Fase Validasi
        model.eval()
        val_loss, val_correct = 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validasi]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Hitung & Simpan Hasil Epoch
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / len(val_loader.dataset)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    end_time = time.time()
    print(f"\n✅ Training Selesai! Total waktu: {(end_time - start_time)/60:.2f} menit")
    return history

# =============================================================================
# 4. BLOK EKSEKUSI UTAMA
# =============================================================================

if __name__ == '__main__':
    # --- Konfigurasi & Hyperparameter ---
    MODEL_TO_TRAIN = 'resnet'  # Ubah di sini: 'plain' atau 'resnet'
    
    EPOCHS = 5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Path ke file dan folder (sesuaikan jika perlu)
    csv_path = './IF25-4041-dataset/train.csv'
    img_dir_path = './IF25-4041-dataset/train/'
    
    print(f"Menggunakan device: {DEVICE}")
    print(f"Model yang akan dilatih: {MODEL_TO_TRAIN.upper()}-34")

    # --- Persiapan Data ---
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Buat dataset & split
    full_dataset = MakananDataset(csv_file=csv_path, img_dir=img_dir_path, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Terapkan transformasi yang sesuai
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # Buat DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Jumlah data training: {len(train_dataset)}")
    print(f"Jumlah data validasi: {len(val_dataset)}")
    print(f"Kelas yang terdeteksi: {full_dataset.classes}")
    
    # --- Inisialisasi & Training Model ---
    num_classes = len(full_dataset.classes)
    
    if MODEL_TO_TRAIN == 'plain':
        model = Plain34(num_classes=num_classes).to(DEVICE)
        model_name = "Plain-34"
    elif MODEL_TO_TRAIN == 'resnet':
        model = ResNet34(num_classes=num_classes).to(DEVICE)
        model_name = "ResNet-34"
    else:
        raise ValueError("MODEL_TO_TRAIN harus 'plain' atau 'resnet'")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Jalankan training
    training_history = train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE
    )
    
    # Di sini kamu bisa menambahkan kode untuk menyimpan model atau memvisualisasikan history
    # Contoh: torch.save(model.state_dict(), f'{model_name.lower()}_model.pth')
    print("\nProses selesai.")