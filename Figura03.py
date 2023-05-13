from Figura01 import *

# Abrindo Imagem
Imagem = cv2.imread('ImagemCortada/Figura03.png')

# Aplicando filtro Gaussiano
Imagem2 = cv2.GaussianBlur(src=Imagem, ksize=(13,13), sigmaX=0)

#Exibindo Imagens
final = cv2.hconcat((Imagem, Imagem2))
cv2.imshow("Imagem Oriiginal e Imagem Com Filtro Gaussiano", final)

