from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
import skimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
from funcoes import *
import numpy as np

#----------------------- Figura 01 -----------------------#
# Carregando Imagens
Imagem = cv2.imread('images/diaretdb1_image003.png')
Imagem2 = cv2.imread('images/diaretdb1_image021.png')

# Diminuindo tamanho das imagens
I1 = diminuirImagem(Imagem)
I2 = diminuirImagem(Imagem2)

#Exibindo Imagens
final = cv2.hconcat((I1, I2))
cv2.imshow("Figura01", final)


#----------------------- Figura 02 -----------------------#

Imagem = diminuirImagem(cv2.imread("images/diaretdb1_image025.png"))
corte = diminuirImagem(cv2.imread("ImagemCortada/Figura02.jpeg"))
juncao = cv2.addWeighted(Imagem, 1, corte, 1, 0)

cv2.imshow("Figura 02", juncao)


#----------------------- Figura 03 -----------------------#

# Abrindo Imagem
Imagem = cv2.imread('ImagemCortada/Figura03.png')

# Aplicando filtro Gaussiano
Imagem2 = cv2.GaussianBlur(src=Imagem, ksize=(13,13), sigmaX=0)

#Exibindo Imagens
final = cv2.hconcat((Imagem, Imagem2))
cv2.imshow("Figura 03", final)


#----------------------- Figura 04 -----------------------#
Imagem = diminuirImagem(cv2.imread("images/diaretdb1_image019.png"))

# Aplicando filtro para eliminar ruido
Imagem2 = cv2.GaussianBlur(src=Imagem, ksize=(3,3), sigmaX=0)

# Extracao do canal verde
Imagem3 = Imagem.copy()
Imagem3[:,:,0] = 0
Imagem3[:,:,2] = 0

# Conversoes em tons de cinza
Imagem4 = cv2.cvtColor(Imagem, cv2.COLOR_BGR2GRAY)
Imagem4 = cv2.cvtColor(Imagem4, cv2.COLOR_GRAY2BGR)


# Contatenado imagens e exibindo
final = np.concatenate((Imagem, Imagem2), axis=1)
final2 = np.concatenate((Imagem3, Imagem4), axis=1)
Rfinal = np.concatenate((final, final2), axis=0)

cv2.imshow("Figura 04", Rfinal)


#----------------------- Figura 05 -----------------------#
plt.figure()
plt.title("Figura 05")
plt.hist(Imagem4.ravel(), 256, [0,255])
plt.show()

#----------------------- Figura 06 -----------------------#
# Aplicando Binarizacao
ret,thresh1 = cv2.threshold(Imagem4,127,255,cv2.THRESH_BINARY)
#cv2.imshow("Binarizacao", thresh1)

# Eliminando falsos positivos
image = thresh1.copy()
kernel = np.ones((5, 5), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Convertendo para tons de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ceiando Imagem Binaria
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenhando todos os contours
image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

'''
plt.subplot(121),plt.imshow(binary, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''

img = thresh1.copy()
lower_val = np.array([0,0,0])
upper_val = np.array([40,40,100])
mask = cv2.inRange(img, lower_val, upper_val)
selected_contours = []
imagens = []
for contour in contours:
  area = cv2.contourArea(contour)
  if area < 700:
    selected_contours.append(contour)
    blank_image = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    imagens.append(blank_image)

masked = cv2.bitwise_and(thresh1, thresh1, mask=imagens[-3])
#cv2.imshow("eeff", imagens[-3])
#cv2.imshow("Eliminacao De DO", masked)

im_v = cv2.hconcat([thresh1, masked])
cv2.imshow("Figura 06", im_v)


#----------------------- Figura 07 -----------------------#
# Carregando imagens
im = cv2.imread('images/diaretdb1_image015.png')
im2 = cv2.imread('images/diaretdb1_image012.png')
im3 = cv2.imread('images/diaretdb1_image024.png')
I1 = diminuirImagem(im)
I2 = diminuirImagem(im2)
I3 = diminuirImagem(im3)

# Aplicando binarizacao
# Convertendo para tons de cinza
# Imagem 1
g = I1[:,:,1]
alpha = 1.5
beta = 10
out = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)

# Imagem 2
g2 = I2[:,:,1]
out2 = cv2.convertScaleAbs(g2, alpha=alpha, beta=beta)

# Imagem 3
g3 = I3[:,:,1]
out3 = cv2.convertScaleAbs(g3, alpha=alpha, beta=beta)

# Criando Imagem Binaria
_, t1 = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
_, t2 = cv2.threshold(out2, 128 , 255, cv2.THRESH_BINARY)
_, t3 = cv2.threshold(out3, 108, 255, cv2.THRESH_BINARY)

########################### Imagem 1 ###########################
# Eliminando falsos positivos
image = t1.copy()
kernel = np.ones((11, 11), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# contours
contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenhando todos os contours
image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

'''
plt.subplot(121),plt.imshow(t1, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''

img = image.copy()
selected_contours = []
imagens = []
for contour in contours:
  area = cv2.contourArea(contour)
  if area < 2400:
    selected_contours.append(contour)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    #plt.imshow(blank_image)
    #plt.show()
    imagens.append(blank_image)


masked = cv2.bitwise_and(t1, t1, mask=imagens[-1])
m1 = skimage.color.gray2rgb(masked)
rs = cv2.addWeighted(I1, 1, m1, 0.7, 0)


########################### Imagem 2 ###########################
# Eliminando falsos positivos
image = t2.copy()
kernel = np.ones((11, 11), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# contours
contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenhando todos os contours
image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

'''
plt.subplot(121),plt.imshow(t1, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''

img = image.copy()
selected_contours = []
imagens = []
for contour in contours:
  area = cv2.contourArea(contour)
  if area < 200:
    selected_contours.append(contour)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    #plt.imshow(blank_image)
    #plt.show()
    imagens.append(blank_image)

masked = cv2.bitwise_and(t2, t2, mask=imagens[-1])
m2 = skimage.color.gray2rgb(masked)
rs2 = cv2.addWeighted(I2, 1, m2, 0.7, 0)


########################### Imagem 3 ###########################
# Eliminando falsos positivos
image = t3.copy()
kernel = np.ones((11, 11), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# contours
contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenhando todos os contours
image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

'''
plt.subplot(121),plt.imshow(t3, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''

img = image.copy()
selected_contours = []
imagens = []
for contour in contours:
  area = cv2.contourArea(contour)
  if area < 500:
    selected_contours.append(contour)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    #plt.imshow(blank_image)
    #plt.show()
    imagens.append(blank_image)

masked = cv2.bitwise_and(t3, t3, mask=imagens[-1])
m3 = skimage.color.gray2rgb(masked)
rs3 = cv2.addWeighted(I3, 1, m3, 0.7, 0)

# Montando imagem final
final = np.concatenate((I1, m1, rs), axis=1)
final2 = np.concatenate((I2, m2, rs2), axis=1)
final3 = np.concatenate((I3, m3, rs3), axis=1)
Rfinal = np.concatenate((final, final2, final3), axis=0)
cv2.imshow("Figura07", Rfinal)


#----------------------- Figura 08 -----------------------#

DELAY_CAPTION = 1500
DELAY_BLUR = 100
MAX_KERNEL_LENGTH = 31
src = None
dst = None

def preprocessamento(imagem):
  alpha = 1.5
  beta = 10
  out = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  # Testes lab
  brightLAB = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
  #print ("brightLAB", brightLAB.shape)
  #canal L
  canal_L = brightLAB[:,:,0]

  # Aplicando contraste
  menos_canal_L = cv2.bitwise_not(canal_L)
  #cv2.imshow("CANAL L NEGATIVO", menos_canal_L)

  #cv2.imshow("alpha e beta", out)
  for i in range(1, 4, 2):
    blur = cv2.medianBlur(menos_canal_L, i)
  #cv2.imshow("medianBlur", blur)

  gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
  tophat = cv2.morphologyEx(menos_canal_L, cv2.MORPH_TOPHAT, kernel)
  bothat = cv2.morphologyEx(menos_canal_L, cv2.MORPH_BLACKHAT, kernel)

  alpha = 5.0
  beta = 10
  tophat = cv2.convertScaleAbs(tophat, alpha=alpha, beta=beta)
  bothat = cv2.convertScaleAbs(bothat, alpha=alpha, beta=beta)

  #cv2.imshow("tophat", tophat)
  #cv2.imshow("bothat", bothat)

  gradient = cv2.bitwise_not(gradient)
  Imagem1 = cv2.bitwise_and(blur, gradient)
  #cv2.imshow("Blur + Gradient", Imagem1)
  gradient = cv2.convertScaleAbs(gradient, alpha=alpha, beta=beta)


  edges = cv2.Canny(blur, 40, 80)
  topbot = cv2.bitwise_or(bothat, tophat)
  topbot = cv2.morphologyEx(topbot, cv2.MORPH_CLOSE, kernel)
  blur_topbot = cv2.bitwise_not(topbot)
  ret, blur_topbot = cv2.threshold(blur_topbot, 130, 255, cv2.THRESH_BINARY)

  #blur_topbot = cv2.bitwise_not(blur_topbot)
  #cv2.imshow("binarizacao blur", blur_topbot)
  rs = blur_topbot.copy()
  blur_topbot = cv2.bitwise_and(blur_topbot, blur)
  #cv2.imshow("Canny", blur_topbot)
  #cv2.imshow("topbot", topbot)



  blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
  Imagem1 = cv2.cvtColor(blur_topbot, cv2.COLOR_GRAY2BGR)

  im_h = cv2.hconcat([imagem, blur, Imagem1])
  cv2.imshow('Figura 08', im_h)
  cv2.waitKey(0)
  return rs

preprocessamento(diminuirImagem(cv2.imread("images/diaretdb1_image016.png")))


#----------------------- Figura 09 -----------------------#
im = diminuirImagem(cv2.imread("images/diaretdb1_image016.png"))
#cv2.imshow("Imagem Original", im)

g = im[:,:,1]
alpha = 1.5
beta = 10
out = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)
out2 = out.copy()
edges = cv2.Canny(out,20,120)
#cv2.imshow("Canny", edges)


cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(out, cnts, [255,255,255])
#cv2.imshow("Preenchimento", out)


rs = out - edges
_, rs2 = cv2.threshold(rs, 130, 255, cv2.THRESH_BINARY)
#cv2.imshow("Binarizacao", rs2)


final = np.concatenate((edges, out, rs2), axis=1)
cv2.imshow("Figura09", final)


#----------------------- Figura 10 -----------------------#
image = rs2.copy()
kernel = np.ones((3, 3), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)


'''
plt.subplot(121),plt.imshow(rs2, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''

img = image.copy()
selected_contours = []
imagens = []

for contour in contours:
  area = cv2.contourArea(contour)
  if area < 2:
    selected_contours.append(contour)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    #plt.imshow(blank_image)
    #plt.show()
    imagens.append(blank_image)

imResult = imagens[-1]
final = np.concatenate((rs2, imagens[-1]), axis=1)
cv2.imshow("Figura10", final)


#----------------------- Figura 11 -----------------------#
im = diminuirImagem(cv2.imread("images/diaretdb1_image016.png"))
im2 = imResult.copy()

m = skimage.color.gray2rgb(im2)
rss = cv2.addWeighted(im, 1, m, 0.7, 0)

############################ Imagem 02 ############################
im3 = diminuirImagem(cv2.imread("images/diaretdb1_image053.png"))
#cv2.imshow("Imagem Original", im3)

g = im3[:,:,1]
alpha = 1.5
beta = 10
out = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)
out2 = out.copy()
edges = cv2.Canny(out,20,120)
#cv2.imshow("Canny", edges)

cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(out, cnts, [255,255,255])
#cv2.imshow("Preenchimento", out)

rs = out - edges
_, rs2 = cv2.threshold(rs, 130, 255, cv2.THRESH_BINARY)
#cv2.imshow("Binarizacao", rs2)

image = rs2.copy()
kernel = np.ones((5, 5), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

'''
plt.subplot(121),plt.imshow(rs2, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''

img = image.copy()
selected_contours = []
imagens = []

for contour in contours:
  area = cv2.contourArea(contour)
  if area < 2:
    selected_contours.append(contour)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    #plt.imshow(blank_image)
    #plt.show()
    imagens.append(blank_image)

m2 = skimage.color.gray2rgb(imagens[-1])
rss2 = cv2.addWeighted(im3, 1, m2, 0.7, 0)
#cv2.imshow("nnnnnnn", rs2)

############################ Imagem 03 ############################
im4 = diminuirImagem(cv2.imread("images/diaretdb1_image053.png"))
#cv2.imshow("Imagem Original", im3)

g = im4[:,:,1]
alpha = 1.5
beta = 10
out = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)
out2 = out.copy()
edges = cv2.Canny(out,20,120)
#cv2.imshow("Canny", edges)

cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(out, cnts, [255,255,255])
#cv2.imshow("Preenchimento", out)

rs = out - edges
_, rs2 = cv2.threshold(rs, 130, 255, cv2.THRESH_BINARY)
#cv2.imshow("Binarizacao", rs2)

image = rs2.copy()
kernel = np.ones((5, 5), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

'''
plt.subplot(121),plt.imshow(rs2, cmap="gray")
plt.subplot(122),plt.imshow(image)
plt.show()
'''

img = image.copy()
selected_contours = []
imagens = []

for contour in contours:
  area = cv2.contourArea(contour)
  if area < 2:
    selected_contours.append(contour)
    blank_image = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=selected_contours, color=(255,255,255))
    #plt.imshow(blank_image)
    #plt.show()
    imagens.append(blank_image)

m3 = skimage.color.gray2rgb(imagens[-1])
rss3 = cv2.addWeighted(im4, 1, m3, 0.7, 0)
#cv2.imshow("mmmm", rs3)

final = np.concatenate((im, m, rss), axis=1)
final2 = np.concatenate((im3, m2, rss2), axis=1)
final3 = np.concatenate((im4, m3, rss3), axis=1)
Rfinal = np.concatenate((final, final2, final3), axis=0)
cv2.imshow("Figura11", Rfinal)
