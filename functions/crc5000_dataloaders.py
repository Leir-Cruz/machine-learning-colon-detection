
def get_dataloaders(df_train, df_validation, df_test, batch_size = 1, num_workers = 0):
    dataset_train = CRC5000DatasetMemory(
        df_train,
        transforms=CRC5000DatasetConfig.TRAIN_TRANSFORMS,
        target_column="label",
    )

    dataset_validation = CRC5000DatasetMemory(
        df_validation,
        transforms=CRC5000DatasetConfig.TEST_TRANSFORMS,
        target_column="label",
    )

    dataset_test = CRC5000DatasetMemory(
        df_test,
        transforms=CRC5000DatasetConfig.TEST_TRANSFORMS,
        target_column="label",
    )

    dataloader_train = DataLoader(dataset_train, batch_size= batch_size,pin_memory = True, shuffle= True, num_workers = num_workers)
    dataloader_validation = DataLoader(dataset_validation, batch_size= batch_size, pin_memory = True,shuffle= False, num_workers = num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size= batch_size, pin_memory = True,shuffle= False, num_workers = num_workers)

    return dataloader_train, dataloader_validation, dataloader_test