from skimage.measure import regionprops, regionprops_table
from skimage.morphology import remove_small_objects
from Figura04 import *
import skimage
from skimage.color import rgb2gray

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
#cv2.imshow("Eliminacao De DO", masked)

im_v = cv2.hconcat([thresh1, masked])
cv2.imshow("Figura 06", im_v)





