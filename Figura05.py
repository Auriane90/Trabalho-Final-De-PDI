from Figura04 import *
from matplotlib import pyplot as plt

plt.figure()
plt.title("Figura 05")
plt.hist(Imagem4.ravel(), 256, [0,255])
plt.show()
