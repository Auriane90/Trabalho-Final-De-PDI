# PROJETO FINAL DE PDI

## DESCRIÇÃO

Este projeto foi proposto com o objetivo de colocar em prática os conceitos estudados ao longo do semestre na disciplina de PDI, ministrada pelo professor Igor Valente. 
Para a realização deste projeto, foi preciso:
1. Escolher um artigo, na área de Processamento de Imagens Digitais, publicado em revista ou congresso;
2. Implementar o que o artigo propõe;
3. Propor melhorias para o artigo;
4. Implementar as melhorias propostas.

## INFORMAÇÕES DESTE PROJETO
- **PROFESSOR ORIENTADOR:** Igor Valente
- **EQUIPE**
  + [Auriane Barbosa](https://github.com/Auriane90)
  + [Luiz Henrique](https://github.com/LuizHenriqueVitorino)
  + [Matheus]()
  + [Yan Falcão](https://github.com/yanfalcao)
- **ARTIGO ESCOLHIDO:** [Detection of Exudates and Microaneurysms in the Retina by Segmentation in Fundus Images](https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S0188-95322021000200105&lang=pt)

## DEPENDÊNCIAS
Para este projeto, fizemos uso das bibliotecas `cv2`, `pyplot` e `numpy`

```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np
```

**OBS:** Podemos instalar as dependências com o gestor de pacotes `pip`. Basta seguir os seguintes passos:
1. Abra um terminal;
2. Atualize o gerenciador de pacotes;
```
sudo apt-get updade
```
3. Instale o `pip`
```
sudo apt-get install python3-pip
```

Perfeito! Agora você está pronto para a intalação dos pacotes.

### Instalação do **cv2**
```
pip install opencv-python
```
### Instalação do **pyplot**
```
python -m pip install -U matplotlib
```
### Instalação do **numpy**
```
pip install numpy
```