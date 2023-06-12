from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
import skimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np

# Carregando imagens
im = cv2.imread('images/diaretdb1_image015.png')
im2 = cv2.imread('images/diaretdb1_image012.png')
im3 = cv2.imread('images/diaretdb1_image024.png')
scalaPercent = 20
widht = int(im.shape[1] * scalaPercent / 100)
height = int(im.shape[0] * scalaPercent / 100)
dim = (widht, height)
I1 = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
I2 = cv2.resize(im2, dim, interpolation=cv2.INTER_AREA)
I3 = cv2.resize(im3, dim, interpolation=cv2.INTER_AREA)

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

#################################### Imagem 1########################################################
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


#################################### Imagem 2 ########################################################
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


#################################### Imagem 3 ########################################################
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






