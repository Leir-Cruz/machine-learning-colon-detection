from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os

class LC25000DatasetMemory(Dataset):
    def __init__(self, dataframe, transforms=None, target_column="label"):
        self.dataframe = dataframe
        self.transforms = transforms
        self.target_column = target_column

        # Carregar todas as imagens para memória (em forma de tensores)
        self.images = []
        self.labels = []

        for idx, row in dataframe.iterrows():
            # Carregar imagem da URL ou caminho
            image_path = row['image_path']  # Supondo que o caminho da imagem está em 'image_path'
            image = Image.open(image_path).convert('RGB')
            # Converter para tensor
            image_tensor = transforms.ToTensor()(image)
            
            # Armazenar na lista
            self.images.append(image_tensor)
            self.labels.append(row[target_column])  # Supondo que a label está na coluna 'label'

        # Converter as listas para tensores de uma vez (isso carrega tudo em memória)
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, label