from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
import skimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np


im = cv2.imread("images/diaretdb1_image016.png")
scalaPercent = 20
widht = int(im.shape[1] * scalaPercent / 100)
height = int(im.shape[0] * scalaPercent / 100)
dim = (widht, height)
im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
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

