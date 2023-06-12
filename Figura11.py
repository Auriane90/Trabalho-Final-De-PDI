from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
import skimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
from Figura10 import *
import numpy as np

############################ Imagem 01 ############################
im = cv2.imread("images/diaretdb1_image016.png")
scalaPercent = 20
widht = int(im.shape[1] * scalaPercent / 100)
height = int(im.shape[0] * scalaPercent / 100)
dim = (widht, height)

im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
im2 = imResult.copy()

m = skimage.color.gray2rgb(im2)
rss = cv2.addWeighted(im, 1, m, 0.7, 0)


############################ Imagem 02 ############################
im3 = cv2.imread("images/diaretdb1_image053.png")
im3 = cv2.resize(im3, dim, interpolation=cv2.INTER_AREA)
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
im4 = cv2.imread("images/diaretdb1_image053.png")
im4 = cv2.resize(im4, dim, interpolation=cv2.INTER_AREA)
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
