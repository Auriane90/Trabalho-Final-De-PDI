import cv2
import numpy as np
from matplotlib import pyplot as plt

DELAY_CAPTION = 1500
DELAY_BLUR = 100
MAX_KERNEL_LENGTH = 31
src = None
dst = None
def diminuirImagem(imagem) :
    # Diminuindo tamanho das imagens
    scalaPercent = 20
    widht = int(imagem.shape[1] * scalaPercent / 100)
    height = int(imagem.shape[0] * scalaPercent / 100)
    dim = (widht, height)
    return cv2.resize(imagem, dim, interpolation=cv2.INTER_AREA)

def canalVerde(imagem) :
    # Aplicando filtro para eliminar ruido
    Imagem2 = cv2.GaussianBlur(src=imagem, ksize=(3, 3), sigmaX=0)
    # Extracao do canal verde
    Imagem3 = Imagem2.copy()
    Imagem3[:,:,0] = 0
    Imagem3[:,:,2] = 0

    # Conversoes em tons de cinza
    Imagem4 = cv2.cvtColor(Imagem3, cv2.COLOR_BGR2GRAY)
    Imagem5 = cv2.cvtColor(Imagem4, cv2.COLOR_GRAY2BGR)
    Imagem6 = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Contatenado imagens e exibindo
    final = np.concatenate((imagem, Imagem2), axis=1)
    final2 = np.concatenate((Imagem3, Imagem5), axis=1)
    Rfinal = np.concatenate((final, final2), axis=0)
    cv2.imshow("Imagem O, Eliminacao de Ruido, Canal verde, Tons de cinza", Rfinal)
    return(Imagem2, Imagem3)

def histograma(imagem):
    plt.figure()
    plt.title("Histograma")
    plt.hist(imagem.ravel(), 256, [0,255])
    plt.show()

def acharExudates(imagem):
  imagem, imagem2 = canalVerde(imagem)

  # Testes lab
  brightLAB = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
  menosL = cv2.bitwise_not(brightLAB)
  menosLRGB = cv2.cvtColor(menosL, cv2.COLOR_Lab2RGB)
  menosLRGBgray = cv2.cvtColor(menosLRGB, cv2.COLOR_BGR2GRAY)
  histograma(menosLRGBgray)

# Elemento estruturante
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))

# Testes mmorfológicos
  closing = cv2.morphologyEx(menosLRGBgray, cv2.MORPH_CLOSE, kernel)
  opening = cv2.morphologyEx(menosLRGBgray, cv2.MORPH_OPEN, kernel)

  gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
  ret, gradient = cv2.threshold(gradient,15,255,cv2.THRESH_BINARY)
  ret, gradient = cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY)
  erosion = cv2.erode(menosLRGBgray, kernel, iterations=1)
  dilation = cv2.dilate(menosLRGBgray, kernel, iterations=1)

# Preparação top-hat e bot-hat
  tophat = cv2.morphologyEx(menosLRGBgray, cv2.MORPH_TOPHAT, kernel)
  tophat = cv2.bitwise_not(tophat)
  blackhat = cv2.morphologyEx(opening, cv2.MORPH_BLACKHAT, kernel)
  blackhat = cv2.bitwise_not(blackhat)

# Testes de binarização
  th2 = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
  th3 = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  ret, glob = cv2.threshold(opening,127,255,cv2.THRESH_BINARY)

  blackhat = cv2.GaussianBlur(src=blackhat, ksize=(5, 5), sigmaX=0)
  ret, blackhat = cv2.threshold(blackhat, 250, 255, cv2.THRESH_BINARY)

  bot_glob = cv2.bitwise_and(blackhat, glob)

  cv2.imshow("gradiente", gradient)
#  cv2.imshow("erosao", erosion)
#  cv2.imshow("dilatacao", dilation)
#  cv2.imshow("abertura", opening)
#  cv2.imshow("fechamento", closing)
  cv2.imshow("BOT-GLOB", bot_glob)
  cv2.imshow("bothat", blackhat)
#  cv2.imshow("tophat", tophat)
  cv2.waitKey(0)

  thresh4 = cv2.bitwise_not(bot_glob)

  teste = cv2.bitwise_or(gradient, thresh4)
  bot_glob = cv2.bitwise_or(bot_glob, th2)
  bot_glob = cv2.bitwise_not(bot_glob)
  cv2.imshow("teste", teste)

  # Eliminando buracos
  image = bot_glob.copy()
  kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
  image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel2)

  #image = cv2.erode(image, kernel2, iterations=1)

  #image = cv2.dilate(image, kernel2, iterations=1)

  cv2.imshow("buracos", image)

  # contours
  contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Desenhando todos os contours
  image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

  lower_val = np.array([0,0,0])
  upper_val = np.array([40,40,100])
  mask = cv2.inRange(menosLRGB, lower_val, upper_val)

  selected_contours = []
  blank_image = np.zeros(mask.shape, np.uint8)
  for contour in contours:
    area = cv2.contourArea(contour)
    if area < 600:
      selected_contours.append(contour)
      blank_image = np.zeros(mask.shape, np.uint8)
      cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))

  masked = cv2.bitwise_and(thresh4, blank_image)
  Imagem11 = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
  dst = cv2.addWeighted(Imagem11,1,imagem,1,0)

  cv2.imshow("limiarizacao global", glob)
  cv2.imshow("limiarizacao mean", th2)
  cv2.imshow("limiarizacao gauss", th3)
  cv2.waitKey(0)

  cv2.imshow("menosL", menosL)
  cv2.imshow("menosLRGB", menosLRGB)
  cv2.imshow("menosLRGBgray", menosLRGBgray)
  cv2.imshow("menosLRGBgraybi", thresh4)
  cv2.imshow("Juncao", dst)
  cv2.waitKey(0)

  im_v = cv2.hconcat([thresh4, masked])
  cv2.imshow("Figura 06", im_v)
  cv2.waitKey(0)

def preprocessamento(imagem):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  # Testes lab
  brightLAB = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
  print ("brightLAB", brightLAB.shape)
  #canal L
  canal_L = brightLAB[:,:,0]
  menos_canal_L = cv2.bitwise_not(canal_L)
  cv2.imshow("CANAL L NEGATIVO", menos_canal_L)

  for i in range(1, 4, 2):
      dst = cv2.medianBlur(menos_canal_L, i)
  cv2.imshow("medianBlur", dst)

  gradient = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, kernel)
  tophat = cv2.morphologyEx(dst, cv2.MORPH_TOPHAT, kernel)
  bothat = cv2.morphologyEx(dst, cv2.MORPH_BLACKHAT, kernel)

  ret, gradient = cv2.threshold(gradient, 15, 255, cv2.THRESH_BINARY)
  ret, gradient = cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY)
  erosion = cv2.erode(dst, kernel, iterations=1)
  dilation = cv2.dilate(dst, kernel, iterations=1)

  dst2 = cv2.Laplacian(dst, cv2.CV_32F)
  abs_dst = cv2.convertScaleAbs(dst2)
  cv2.imshow("Laplace 16S", abs_dst)

  sharp_kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
  flt_img = cv2.filter2D(src=dst, ddepth=-1, kernel=sharp_kernel)
  Imagem1 = cv2.bitwise_or(flt_img, dst)
  edges = cv2.Canny(Imagem1, 20, 50)

  cv2.imshow("Canny 1", edges)
  cv2.imshow("Laplace + Gauss 1", Imagem1)

  dst2 = cv2.Laplacian(dst, cv2.CV_64F)
  abs_dst = cv2.convertScaleAbs(dst2)
  cv2.imshow("Laplace 64F", dst2)

  Imagem1 = cv2.bitwise_or(abs_dst, dst)

  edges = cv2.Canny(Imagem1, 40, 160)

  cv2.imshow("Canny 2", edges)
  cv2.imshow("Laplace + Gauss 2", Imagem1)
  cv2.waitKey(0)