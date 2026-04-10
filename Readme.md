# Objetivo do trabalho
Trabalho de visão computacional voltado para detecção e classificação de câncer de colo intestino através de imagens de endoscopia.

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
- ResNet50
- ResNet101
- ResNet 152
- DenseNet-121
- QuanvolutionModel


## Explicando versões Notebooks
- v1: versões que rodam diretamento no google colab, importando dados do kaggle dataset e funções em núvem.
- v2: versões que rodam tanto locamente quanto no google colab, importando os dados via api kaggle, sem necessidade de chave de acesso e importação de funções.
- v3: atualização da v2, rodando com scheduler para cnn clássicas e earlyStopping. Para modelos quanticos, tentativas de rodar com outros tamanho de imagem e uso de todo dataset.

## Hiperparâmetros usados
### V1, V2, V3
- AlexNet: learningRate: 1e-4; optimizer: Adam
- Vgg16: learningRate: 1e-2; optimizer: SGD momentum 0.9 e weight_decay: 5*1e-4
- Resnet18: learningRate: 1e-2; optimizer: AdamMax b1=0.9 b2=0.999 weight_decay 1e-4 eps= 1e-08
- ResNet34: 1e-2; optimizer: AdamMax b1=0.9 b2=0.999 weight_decay 1e-4 eps= 1e-08
- ResNet50: 1e-2; optimizer: AdamMax b1=0.9 b2=0.999 weight_decay 1e-4 eps= 1e-08
- ResNet101: 1e-2; optimizer: AdamMax b1=0.9 b2=0.999 weight_decay 1e-4 eps= 1e-08
- ResNet152: 1e-2; optimizer: AdamMax b1=0.9 b2=0.999 weight_decay 1e-4 eps= 1e-08

