from Figura09 import *


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
