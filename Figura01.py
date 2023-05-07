import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carregando Imagens
Imagem = cv2.imread('images/diaretdb1_image003.png')
Imagem2 = cv2.imread('images/diaretdb1_image021.png')

# Diminuindo tamanho das imagens
scalaPercent = 20
widht = int(Imagem.shape[1] * scalaPercent / 100)
height = int(Imagem.shape[0] * scalaPercent / 100)
dim = (widht, height)
I1 = cv2.resize(Imagem, dim, interpolation=cv2.INTER_AREA)
I2 = cv2.resize(Imagem2, dim, interpolation=cv2.INTER_AREA)

#Exibindo Imagens
final = cv2.hconcat((I1, I2))
cv2.imshow("Retiina", final)
