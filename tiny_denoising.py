import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# 1. 고성능 GPU 설정 및 결과 저장 경로
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULT_DIR = '/content/drive/MyDrive/individual_research/results/week1'
os.makedirs(RESULT_DIR, exist_ok=True)

# 2. CIFAR-10 규격의 가상 데이터를 메모리에 생성
clean_images = torch.rand(64, 3, 32, 32).to(device)
mock_labels = torch.zeros(64, dtype=torch.long).to(device)

dataset = TensorDataset(clean_images.cpu(), mock_labels.cpu())
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 가우시안 노이즈 생성 (시그마 25/255)
sigma = 25 / 255.0
noise = torch.randn_like(clean_images) * sigma
noisy_images = torch.clamp(clean_images + noise, 0.0, 1.0)

# 3. 이미지 복원용 3층 CNN 모델 설계(small CNN)
class TinyDenoisingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

model = TinyDenoisingCNN().to(device)
criterion = nn.MSELoss()

# Adam 옵티마이저 및 lr=1e-3 설정
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 2000 Iteration 본격 학습 수행
iteration = 0
max_iterations = 2000
loss_history = []

model.train()
print("Using device:", device)
print("--- 2000 Iteration 본격 학습 시작 ---")

while iteration < max_iterations:
    for clean_batch, _ in train_loader:
        if iteration >= max_iterations: break
        
        clean_batch = clean_batch.to(device)
        noise_batch = torch.randn_like(clean_batch) * sigma
        noisy_batch = torch.clamp(clean_batch + noise_batch, 0.0, 1.0)
        
        optimizer.zero_grad()
        outputs = model(noisy_batch)
        loss = criterion(outputs, clean_batch)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        iteration += 1
        
        if iteration % 200 == 0:
            print(f"Iteration {iteration}/{max_iterations} | Loss: {loss.item():.4f}")

print("--- 2000 Iteration 훈련 완료 ---")

# 5. 결과 격자 이미지 및 Loss Curve 자동 저장
model.eval()
with torch.no_grad():
    final_outputs = torch.clamp(model(noisy_images), 0.0, 1.0)  # 범위 클리핑 추가
    num_samples = 4
    
    # 1) Grid 저장
    grid_orig = vutils.make_grid(clean_images[:num_samples].cpu(), nrow=num_samples, normalize=True)
    grid_noisy = vutils.make_grid(noisy_images[:num_samples].cpu(), nrow=num_samples, normalize=True)
    grid_restored = vutils.make_grid(final_outputs[:num_samples].cpu(), nrow=num_samples, normalize=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].imshow(grid_orig.permute(1, 2, 0)); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(grid_noisy.permute(1, 2, 0)); axes[1].set_title("Noisy"); axes[1].axis("off")
    axes[2].imshow(grid_restored.permute(1, 2, 0)); axes[2].set_title("Restored"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'denoising_grid_results.png'), dpi=300)
    plt.close()
    
    # 2) Loss Curve 저장
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='blue', label='Train Loss')
    plt.title('Denoising CNN Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'loss_curve.png'), dpi=300)
    plt.close()

print("\ 모든 결과가 구글 드라이브에 자동 저장되었음")
