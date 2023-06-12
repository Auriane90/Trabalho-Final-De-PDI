from funcoes import *

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
  print ("brightLAB", brightLAB.shape)
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
  cv2.imshow("binarizacao blur", blur_topbot)
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

Imagem1 = diminuirImagem(cv2.imread("images/diaretdb1_image016.png"))


