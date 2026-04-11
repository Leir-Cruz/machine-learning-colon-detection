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
- v1: versões que rodam diretamento no google colab, importando dados do kaggle dataset e funções em núvem.
- v2: versões que rodam tanto locamente quanto no google colab, importando os dados via api kaggle, sem necessidade de chave de acesso e importação de funções. AlexNet com Adam, VGG16 com SGD, ResNets com AdamMax.
- v3: atualização da v2, rodando com scheduler para cnn clássicas e earlyStopping. Para modelos quanticos, tentativas de rodar com outros tamanho de imagem e uso de todo dataset.
- v4: AlexNet e VGG16 com adamMax
- v5: AlexNet e VGG16 com adamMax e com scheduler
- v6: AlexNet com SGD e VGG16 com Adam
- v7: AlexNet com SGD e VGG16 com Adam com scheduler
- v8: ResNet (18, 34, 50, 101, 152) com SGD
- v9: ResNet (18, 34, 50, 101, 152) com SGD com scheduler
- v10: ResNet (18, 34, 50, 101, 152) com Adam
- v11: ResNet (18, 34, 50, 101, 152) com Adam com scheduler

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

