
"""
Week 3: Exposure-dependent Motion Blur Toy Degradation
=======================================================

1. 실제 FMA-Net++의 blur 생성 개념
-----------------------------------
FMA-Net++에서는 여러 고속 프레임을 exposure interval 동안 시간적으로
누적하거나 평균하여 motion blur가 포함된 영상을 생성한다.

실제 blur는 다음 요소에 의해 결정된다.

    Exposure duration + Motion trajectory
        → Spatially varying motion blur

따라서 실제 blur kernel은 영상 전체에서 동일한 직선 kernel이 아니라,
물체의 움직임과 위치에 따라 달라질 수 있다.


2. Week 3 toy experiment의 단순화
----------------------------------
현재 실험에서는 단일 HR 이미지만 사용하므로 실제 시간 방향 motion
trajectory를 얻을 수 없다.

따라서 다음과 같은 global line motion kernel로 단순화한다.

    K_{e,theta} = MotionKernel(L(e), theta)

여기서

    e       : exposure level
    L(e)    : exposure에 따라 증가하는 kernel length
    theta   : motion direction
    K       : 영상 전체에 동일하게 적용되는 line kernel

Exposure별 kernel length:

    Exposure 5:1 → length 3
    Exposure 5:2 → length 5
    Exposure 5:3 → length 9
    Exposure 5:4 → length 13
    Exposure 5:5 → length 17

<참고>
length 3, 5, 9, 13, 17은 FMA-Net++ 논문에서 지정한 수치가 아니다.
Exposure가 길어질수록 blur extent가 커진다는 관계를 단일 이미지에서
확인하기 위해 정한 toy experiment의 설정값이다.


3. 전체 toy degradation
------------------------
최종 degradation은 다음 수식으로 정의한다.

    y = clip(U_s(D_s(B_{e,theta}(x))) + n, 0, 1)

처리 순서:

    HR image x
    → Exposure-dependent motion blur B_{e,theta}
    → Bicubic downsampling D_s
    → Bicubic upsampling U_s
    → Gaussian noise n
    → [0, 1] clipping
    → Degraded image y

Gaussian noise는 FMA-Net++의 기본 REDS-ME/RE 합성 과정 자체를
그대로 재현한 요소가 아니라, Week 3 toy degradation 식을 구현하기
위해 추가한 약한 sensor-like noise이다.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ================================================================
# 1. Exposure level과 toy motion-kernel length의 대응 관계
# ================================================================
EXPOSURE_TO_LENGTH = {
    1: 3,    # Exposure 5:1: 가장 짧은 exposure
    2: 5,    # Exposure 5:2
    3: 9,    # Exposure 5:3
    4: 13,   # Exposure 5:4
    5: 17,   # Exposure 5:5: 가장 긴 exposure
}


def make_motion_kernel(length, angle):
    """
    길이와 방향이 지정된 2D line motion kernel을 생성한다.

    실제 FMA-Net++의 spatially varying blur kernel을 그대로 구현한
    것이 아니라, 단일 이미지 실험을 위한 global line kernel이다.

    Args:
        length (int):
            Kernel 크기이자 근사된 motion trajectory의 길이.
            중심을 명확하게 정의하기 위해 양의 홀수를 사용한다.

        angle (float):
            Motion 방향을 degree 단위로 지정한다.

            0도  : 수평
            45도 : 대각선
            90도 : 수직

    Returns:
        np.ndarray:
            [length, length] 크기의 normalized motion kernel.
            Kernel 전체 원소의 합은 1이다.
    """

    if not isinstance(length, int) or length < 1 or length % 2 == 0:
        raise ValueError("length는 양의 홀수여야 합니다.")

    kernel = np.zeros((length, length), dtype=np.float32)

    center = (length - 1) / 2.0
    radius = center

    # 직선 motion blur에서 0도와 180도는 동일한 방향이므로
    # angle을 [0, 180) 범위로 변환한다. (180포함 안함)
    angle = float(angle) % 180.0
    angle_rad = np.deg2rad(angle)

    # Kernel 중심에서 선분 양 끝점까지의 이동량
    dx = radius * np.cos(angle_rad)
    dy = radius * np.sin(angle_rad)

    # 회전된 선이 끊겨 보이지 않도록 선분 위를 촘촘하게 sampling한다.
    num_samples = max(length * 16, 1)

    x_coordinates = np.linspace(
        center - dx,
        center + dx,
        num_samples
    )

    y_coordinates = np.linspace(
        center - dy,
        center + dy,
        num_samples
    )

    # 연속 좌표를 가장 가까운 pixel 좌표로 변환한다.
    columns = np.rint(x_coordinates).astype(np.int64)
    rows = np.rint(y_coordinates).astype(np.int64)

    columns = np.clip(columns, 0, length - 1)
    rows = np.clip(rows, 0, length - 1)

    # 동일한 pixel에 해당하는 sample을 누적한다.
    np.add.at(kernel, (rows, columns), 1.0)

    kernel_sum = kernel.sum()

    if kernel_sum <= 0:
        raise RuntimeError("Motion kernel 생성에 실패했습니다.")

    # Convolution 전후 평균 밝기가 달라지지 않도록 합을 1로 정규화
    kernel /= kernel_sum

    return kernel


def make_exposure_kernel(exposure_level, angle):
    """
    Exposure level과 motion angle에 따라 motion kernel을 생성한다.

        K_{e,theta} = MotionKernel(L(e), theta)

    Exposure level이 증가하면 L(e)가 증가하므로 blur extent도 증가한다.

    Args:
        exposure_level (int):
            1~5 중 하나이며 다음 exposure에 대응한다.

            1 → 5:1
            2 → 5:2
            3 → 5:3
            4 → 5:4
            5 → 5:5

        angle (float):
            Motion 방향(degree).

    Returns:
        np.ndarray:
            Exposure-dependent normalized motion kernel.
    """

    if exposure_level not in EXPOSURE_TO_LENGTH:
        raise ValueError(
            "exposure_level은 1, 2, 3, 4, 5 중 하나여야 합니다."
        )

    kernel_length = EXPOSURE_TO_LENGTH[exposure_level]

    return make_motion_kernel(
        length=kernel_length,
        angle=angle
    )


def apply_motion_blur(image, kernel):
    """
    입력 영상에 exposure-dependent motion blur를 적용한다.

        x_b = B_{e,theta}(x) = K_{e,theta} * x

    RGB 각 channel에 동일한 kernel을 depthwise convolution으로 적용한다.

    Args:
        image (torch.Tensor):
            [C, H, W] 형태이며 pixel 범위는 [0, 1].

        kernel (np.ndarray):
            make_exposure_kernel()로 생성한 2D kernel.

    Returns:
        torch.Tensor:
            원본과 동일한 [C, H, W] 크기의 blurred image.
    """

    if image.ndim != 3:
        raise ValueError("image는 [C, H, W] 형태여야 합니다.")

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel은 정사각형 2D 배열이어야 합니다.")

    image_batch = image.unsqueeze(0)  # [1, C, H, W]

    kernel_tensor = torch.as_tensor(
        kernel,
        device=image.device,
        dtype=image.dtype
    )

    kernel_size = kernel_tensor.shape[0]
    padding = kernel_size // 2
    channels = image.shape[0]

    # RGB 각 channel에 동일한 kernel을 독립적으로 적용
    weight = kernel_tensor.reshape(
        1, 1, kernel_size, kernel_size
    ).repeat(
        channels, 1, 1, 1
    )

    # Zero padding으로 인한 검은 테두리를 줄이기 위해 reflect padding 사용
    padded_image = F.pad(
        image_batch,
        (padding, padding, padding, padding),
        mode="reflect"
    )

    blurred = F.conv2d(
        padded_image,
        weight,
        groups=channels
    )

    return blurred.squeeze(0)


def apply_toy_degradation(
    image,
    exposure_level,
    angle,
    scale=2,
    noise_sigma=2.0 / 255.0,
    seed=None,
):
    """
    전체 toy degradation을 수행한다.

    전체 수식:

        y = clip(U_s(D_s(B_{e,theta}(x))) + n, 0, 1)

    단계별 정의:

        x_b  = B_{e,theta}(x)
        x_LR = D_s(x_b)
        x_up = U_s(x_LR)
        n    ~ N(0, sigma^2)
        y    = clip(x_up + n, 0, 1)

    처리 순서:

        x → Blur → Downsample → Upsample → Noise → Clip

    Args:
        image (torch.Tensor):
            [C, H, W] 형태의 HR image.
            Pixel 범위는 [0, 1].

        exposure_level (int):
            1~5 중 하나. Exposure 5:1~5:5에 대응한다.

        angle (float):
            Motion 방향(degree).

        scale (int):
            Spatial downsampling 배율.
            scale=2이면 높이와 너비를 절반으로 축소한다.

        noise_sigma (float):
            Additive Gaussian noise의 표준편차.
            기본값 2/255는 비교적 약한 noise이다.

        seed (int or None):
            동일한 Gaussian noise를 재현하기 위한 random seed.

    Returns:
        dict:
            kernel, blurred, lr, upsampled, noise, degraded를 반환한다.
    """

    if image.ndim != 3:
        raise ValueError("image는 [C, H, W] 형태여야 합니다.")

    if image.min().item() < 0.0 or image.max().item() > 1.0:
        raise ValueError("image의 pixel 범위는 [0, 1]이어야 합니다.")

    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale은 1 이상의 정수여야 합니다.")

    if noise_sigma < 0:
        raise ValueError("noise_sigma는 0 이상이어야 합니다.")

    _, height, width = image.shape

    lr_height = height // scale
    lr_width = width // scale

    if lr_height < 1 or lr_width < 1:
        raise ValueError("입력 영상 크기에 비해 scale이 너무 큽니다.")

    # Step 1. Exposure-dependent motion kernel 생성
    # K_{e,theta} = MotionKernel(L(e), theta)
    kernel = make_exposure_kernel(
        exposure_level=exposure_level,
        angle=angle
    )

    # Step 2. Motion blur
    # x_b = B_{e,theta}(x)
    blurred = apply_motion_blur(
        image=image,
        kernel=kernel
    )

    # Step 3. Bicubic downsampling
    # x_LR = D_s(x_b)
    lr = F.interpolate(
        blurred.unsqueeze(0),
        size=(lr_height, lr_width),
        mode="bicubic",
        align_corners=False,
        antialias=True
    ).squeeze(0)

    lr = lr.clamp(0.0, 1.0)

    # Step 4. Bicubic upsampling
    # x_up = U_s(x_LR)
    upsampled = F.interpolate(
        lr.unsqueeze(0),
        size=(height, width),
        mode="bicubic",
        align_corners=False
    ).squeeze(0)

    upsampled = upsampled.clamp(0.0, 1.0)

    # Step 5. Gaussian noise
    # n ~ N(0, sigma^2)
    if seed is not None:
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int(seed))

        noise = torch.randn(
            image.shape,
            generator=generator,
            device=image.device,
            dtype=image.dtype
        )
    else:
        noise = torch.randn_like(image)

    noise = noise * noise_sigma

    # Step 6. 최종 degradation
    # y = clip(x_up + n, 0, 1)
    degraded = torch.clamp(
        upsampled + noise,
        min=0.0,
        max=1.0
    )

    return {
        "kernel": kernel,
        "blurred": blurred,
        "lr": lr,
        "upsampled": upsampled,
        "noise": noise,
        "degraded": degraded,
    }
