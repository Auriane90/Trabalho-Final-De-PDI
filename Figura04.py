from Figura01 import *

Imagem = cv2.imread("images/diaretdb1_image019.png")
Imagem = cv2.resize(Imagem, dim, interpolation=cv2.INTER_AREA)

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

cv2.imshow("Imagem O, Eliminacao de Ruido, Canal verde, Tons de cinza", Rfinal)
