
import math
import torch
import pandas as pd

from skimage.metrics import structural_similarity


def calculate_psnr(pred, target):
    """[C,H,W] 이미지 두 장의 PSNR을 계산합니다."""

    pred = pred.detach().clamp(0.0, 1.0)
    target = target.detach().clamp(0.0, 1.0)

    mse = torch.mean((pred - target) ** 2).item()

    if mse == 0:
        return 100.0

    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(pred, target):
    """[C,H,W] RGB 이미지 두 장의 SSIM을 계산합니다."""

    pred_np = (
        pred.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )

    target_np = (
        target.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )

    return float(
        structural_similarity(
            target_np,
            pred_np,
            data_range=1.0,
            channel_axis=2
        )
    )


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    """
    각 Validation 이미지에 대해 Bicubic과 SRCNN의
    PSNR 및 SSIM을 계산합니다.
    """

    model.eval()

    rows = []
    image_number = 1

    for batch in val_loader:
        bicubic = batch["bicubic"].to(device)
        hr = batch["hr"].to(device)

        srcnn = model(bicubic).clamp(0.0, 1.0)

        for index in range(hr.size(0)):
            bicubic_psnr = calculate_psnr(
                bicubic[index],
                hr[index]
            )

            srcnn_psnr = calculate_psnr(
                srcnn[index],
                hr[index]
            )

            bicubic_ssim = calculate_ssim(
                bicubic[index],
                hr[index]
            )

            srcnn_ssim = calculate_ssim(
                srcnn[index],
                hr[index]
            )

            rows.append({
                "image": f"val_{image_number:02d}",
                "bicubic_psnr": bicubic_psnr,
                "srcnn_psnr": srcnn_psnr,
                "psnr_improvement": (
                    srcnn_psnr - bicubic_psnr
                ),
                "bicubic_ssim": bicubic_ssim,
                "srcnn_ssim": srcnn_ssim,
                "ssim_improvement": (
                    srcnn_ssim - bicubic_ssim
                )
            })

            image_number += 1

    result_df = pd.DataFrame(rows)

    if len(result_df) == 0:
        raise RuntimeError(
            "평가할 Validation 이미지가 없습니다."
        )

    mean_row = {
        "image": "MEAN",
        "bicubic_psnr": result_df[
            "bicubic_psnr"
        ].mean(),
        "srcnn_psnr": result_df[
            "srcnn_psnr"
        ].mean(),
        "psnr_improvement": result_df[
            "psnr_improvement"
        ].mean(),
        "bicubic_ssim": result_df[
            "bicubic_ssim"
        ].mean(),
        "srcnn_ssim": result_df[
            "srcnn_ssim"
        ].mean(),
        "ssim_improvement": result_df[
            "ssim_improvement"
        ].mean()
    }

    result_df = pd.concat(
        [
            result_df,
            pd.DataFrame([mean_row])
        ],
        ignore_index=True
    )

    return result_df
