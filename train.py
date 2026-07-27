import math
import torch
import torch.nn as nn

#Loss: L1Loss
#Optimizer: Adam
#Learning rate: 1e-4
#Batch size: Day 1–2에서 설정한 train_loader의 8
def calculate_psnr(pred, target):
    """복원 이미지와 HR 이미지 사이의 PSNR을 계산합니다."""
    pred = pred.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()

    if mse == 0:
        return 100.0

    return 10 * math.log10(1.0 / mse)


@torch.no_grad()
def validate(model, val_loader, device):
    """Validation 데이터 전체의 평균 PSNR을 계산합니다."""
    model.eval()
    total_psnr = 0.0
    image_count = 0

    for batch in val_loader:
        bicubic = batch["bicubic"].to(device)
        hr = batch["hr"].to(device)

        output = model(bicubic)

        # Batch 크기를 반영해 이미지 단위 평균 PSNR을 계산합니다.
        for i in range(output.size(0)):
            total_psnr += calculate_psnr(output[i], hr[i])
            image_count += 1

    return total_psnr / image_count


def train_model(
    model,
    train_loader,
    device,
    iterations,
    lr=1e-4,
    val_loader=None,
    val_interval=100
):
    """정해진 iteration 수만큼 SRCNN을 학습합니다."""
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    val_steps = []
    val_psnrs = []

    train_iterator = iter(train_loader)
    model.train()

    for step in range(1, iterations + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        bicubic = batch["bicubic"].to(device)
        hr = batch["hr"].to(device)

        # 1. 이전 iteration에서 남은 gradient를 초기화합니다.
        optimizer.zero_grad()

        # 2. Bicubic 이미지를 입력해 복원 이미지를 만듭니다.
        output = model(bicubic)

        # 3. 복원 결과와 HR 이미지 사이의 L1 loss를 계산합니다.
        loss = criterion(output, hr)

        # 4. Loss를 기준으로 각 parameter의 gradient를 계산합니다.
        loss.backward()

        # 5. Adam optimizer로 model parameter를 갱신합니다.
        optimizer.step()

        losses.append(loss.item())

        if step == 1 or step % val_interval == 0:
            message = (
                f"Step {step:4d}/{iterations} | "
                f"L1 {loss.item():.6f}"
            )

            if val_loader is not None:
                score = validate(model, val_loader, device)
                val_steps.append(step)
                val_psnrs.append(score)
                message += f" | Val PSNR {score:.2f} dB"
                model.train()

            print(message)

    return losses, val_steps, val_psnrs
