# Objetivo do trabalho
Trabalho de visão computacional voltado para detecção e classificação de câncer de colo intestino através de imagens de endoscopia.

## Dataset utilizado
- *https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images*, LC25000 

## Nesse repositório
O Repositório busca manter centralizdas todas as funções necessárias para o uso da rede neural, desde a leitura dos dados até até as funções auxíliares e os hiper-parâmentros de treinamento.

## No google Colab
Por motivos de limitação de hardware, o projeto é rodado com o uso da ferramenta google colab, apenas responsável pela chamada dos modelos, treinamento e teste.

## Artigos referências
- *Very deep Convolutional Networks For Large-Scale Image Recognition* - Karen Simonyan & Andrew Zisserman
- *Deep Residual Learning for Image Recognition* - Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun
- *ImageNet Classification with Deep Convolutional Neural Networks* - Alex Krizhevsky,Ilya Sutskever, Geoffrey E. Hinton
- *Machine Learning-Based Diagnosis and Detection of Liver Cancer: An Approach Enhancement*- Yogesg Kumar, Perneet Kaur, Jyoti Rani
- *Resource Efficient Deep Learning Architectures for Histopathology-Based Colorectal Cancer Detection* - Md. Bipul Hossain1  Mohamed Shaban
- *Comparative Analysis of ResNet Architecture Enhanced with Self-Attention for Colorectal Cancer Detection* - Yonathan Fanuel Mulyadi Fitri Utaminingrum
- *A Hybrid Quantum-Classical Model for Breast Cancer Diagnosis with Quanvolutions* - Yasmin Rodrigues Sobrinho, Enzo Gabriel Batista Soares, Joao Renato Ribeiro Manesco, Jawaher Al-Tuweity,Rafael Gonçalves Pires, Joao Paulo Papa

## Redes neurais usadas
- AlexNet
- VGG16
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet 152
- QuanvolutionModel


## Explicando versões Notebooks
- LC25000: rodando apenas com dataset LC25000
- CRC5000: rodando apenas com dataset CRC5000
- CRC5000-binary-only: rodando apenas com data CRC 5000, no entanto filtrando para classificação binária apenas
- LC25000 + CRC5000: treinando com LC25000 e testando com CRC5000

## Hiperparâmetros usados
### Adam
- learningRate: 1e-4
### SGD
- learningRate: 1e-2
- momentum 0.9
- weight_decay: 5*1e-4
### AdamMax
- learningRate: 1e-2
- b1=0.9 
- b2=0.999 
- weight_decay 1e-4 
- eps= 1e-08
### Scheduler
- step_size=5
- gamma=0.1

## Datasets
- https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist/data
- https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data
- https://huggingface.co/datasets/DykeF/NCTCRCHE100K