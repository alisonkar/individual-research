
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class DIV2KSubDataset(Dataset):
    def __init__(
        self,
        image_paths,
        patch_size=48,
        scale=2,
        random_crop=True
    ):
        self.image_paths = list(image_paths)
        self.patch_size = patch_size
        self.scale = scale
        self.lr_size = patch_size // scale
        self.random_crop = random_crop

        if patch_size % scale != 0:
            raise ValueError(
                f"patch_size({patch_size})는 "
                f"scale({scale})로 나누어져야 합니다."
            )

        if len(self.image_paths) == 0:
            raise RuntimeError("사용할 DIV2K 이미지가 없습니다.")

        missing_paths = [
            path
            for path in self.image_paths
            if not os.path.isfile(path)
        ]

        if missing_paths:
            raise FileNotFoundError(
                f"존재하지 않는 이미지가 있습니다: "
                f"{missing_paths[0]}"
            )

    def __len__(self):
        return len(self.image_paths)

    def _crop_hr(self, image):
        """원본 이미지에서 HR patch를 잘라냅니다."""
        width, height = image.size

        if width < self.patch_size or height < self.patch_size:
            raise ValueError(
                f"이미지 크기 {image.size}가 "
                f"patch size {self.patch_size}보다 작습니다."
            )

        if self.random_crop:
            # 학습 데이터: 무작위 crop
            left = random.randint(
                0,
                width - self.patch_size
            )
            top = random.randint(
                0,
                height - self.patch_size
            )
        else:
            # Validation 데이터: 중앙 crop
            left = (width - self.patch_size) // 2
            top = (height - self.patch_size) // 2

        return image.crop((
            left,
            top,
            left + self.patch_size,
            top + self.patch_size
        ))

    @staticmethod
    def _to_tensor(image):
        """PIL 이미지를 [0, 1] 범위 Tensor로 변환합니다."""
        return pil_to_tensor(image).float() / 255.0

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            hr_original = image.convert("RGB")

        # HR patch 생성
        hr_image = self._crop_hr(hr_original)

        # HR을 scale 배율만큼 축소해 LR 생성
        lr_image = hr_image.resize(
            (self.lr_size, self.lr_size),
            resample=Image.Resampling.BICUBIC
        )

        # LR을 원래 HR 크기로 Bicubic 확대
        bicubic_image = lr_image.resize(
            (self.patch_size, self.patch_size),
            resample=Image.Resampling.BICUBIC
        )

        return {
            "lr": self._to_tensor(lr_image),
            "bicubic": self._to_tensor(bicubic_image),
            "hr": self._to_tensor(hr_image),
            "name": os.path.basename(image_path)
        }
