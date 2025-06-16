from dataset_config import LC25000DatasetConfig
from lc25000_dataset import LC25000Dataset
from torch.utils.data import  DataLoader


def get_dataloaders(df_train, df_validation, df_test):

    dataset_train = LC25000Dataset(
        df_train,
        transforms=LC25000DatasetConfig.TRAIN_TRANSFORMS,
        target_column="label",
    )

    dataset_validation = LC25000Dataset(
        df_validation,
        transforms=LC25000DatasetConfig.TEST_TRANSFORMS,
        target_column="label",
    )

    dataset_test = LC25000Dataset(
        df_test,
        transforms=LC25000DatasetConfig.TEST_TRANSFORMS,
        target_column="label",
    )

    dataloader_train = DataLoader(dataset_train)
    dataloader_validation = DataLoader(dataset_validation)
    dataloader_test = DataLoader(dataset_test)


    return dataloader_train, dataloader_validation, dataloader_test