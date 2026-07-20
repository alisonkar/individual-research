import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_model():
    # 1. 연산 장치 및 경로 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. CIFAR-10 데이터셋 로드
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 3. 아주 간단한 3층 CNN 모델 설계 (nn.Sequential 방식)
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 10)
    ).to(device)
    
    # 4. 손실함수 및 최적화 도구 설정 (Cross-Entropy & SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 5. 딱 1에포크(Epoch)만 돌려보는 train loop 5단계
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # [5단계 흐름]
        optimizer.zero_grad()               # 1) zero_grad
        outputs = model(inputs)             # 2) forward
        loss = criterion(outputs, targets)  # 3) loss 계산
        loss.backward()                     # 4) backward
        optimizer.step()                    # 5) step
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            break # 제대로 도는지 체크용이므로 1번만 찍고 나간다

if __name__ == "__main__":
    train_model()
