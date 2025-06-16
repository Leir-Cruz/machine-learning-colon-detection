import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2



class LC25000DatasetConfig:
    VAL_SIZE = 0.2
    SEED = 0x40

    TEST_TRANSFORMS = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    TRAIN_TRANSFORMS = A.Compose([
        A.RGBShift(
            r_shift_limit=15,
            g_shift_limit=15,
            b_shift_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.ToGray(p=0.1),
        A.UnsharpMask(p=0.1),
        A.HorizontalFlip(p=0.3),
        A.SafeRotate(limit=30),
        A.HueSaturationValue(p=0.2),
        A.Blur(p=0.1),
        A.GaussNoise(p=0.1),
        A.ISONoise(p=0.2),
        A.Sharpen(p=0.1),
        A.CLAHE(p=0.05),
        A.ElasticTransform(p=0.1),
        A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])