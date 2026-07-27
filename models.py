import torch.nn as nn


class SRCNN(nn.Module):
    """Bicubic 이미지를 입력받아 HR 이미지를 복원하는 SRCNN."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # 9×9 convolution으로 넓은 영역의 이미지 특징을 추출합니다.
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),

            # 1×1 convolution으로 추출된 특징을 조합합니다.
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),

            # 5×5 convolution으로 RGB 고해상도 이미지를 복원합니다.
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)
