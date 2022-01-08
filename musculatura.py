import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def musculatura(img):
    
    imagen = img.copy()


    # 1 - Median Blur y Aumento de contraste
    median_blur_image = cv2.medianBlur(imagen,7) # puedo poner 3,5,7 (5 es 50% de ruido)
    contrast_img = cv2.addWeighted(median_blur_image, 1, np.zeros(imagen.shape, imagen.dtype), 0, 0)

    fondo = contrast_img.copy()
    # 4 - Obtener contornos de nuevo
    ## Volvemos a pasar la imagen a escala de grises
    gray_img = cv2.cvtColor(fondo, cv2.COLOR_RGB2GRAY)

    ## Encuentra el umbral
    retval, thresh = cv2.threshold(gray_img, 80, 250, 0) #variar estos numeros y ver si para todas funciona esos rangos

    ## Usa el findContours() que toma la imagen 
    img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##Dibuje los contornos en la imagen usando el método drawContours()
    cv2.drawContours(fondo, img_contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    # 3 - Eliminamos el fondo
    gray_img = cv2.cvtColor(fondo, cv2.COLOR_RGB2GRAY)

    ## Encuentra el umbral
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ##  Usa el findContours() que toma la imagen 
    img_contours_fondo, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Ordena los contornos
    sombras = list()
    img_contours_fondo = sorted(img_contours_fondo, key=cv2.contourArea)
    for i in img_contours_fondo:
        if cv2.contourArea(i) > 150:     

            # Generate the mask using np.zeros
            mask = np.zeros(fondo.shape[:2], np.uint8)

            # Dibujar los contornos
            cv2.drawContours(mask, [i],-1, 255, -1)

            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(fondo, contrast_img, mask=mask)

            sombras.append(new_img)

    x = sombras[-2]+sombras[-1]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j,2]> 0:
                contrast_img[i,j,2]= 0 


    musc = cv2.imread("musc2.jpeg")
    img_lumbar = cv2.cvtColor(musc,cv2.COLOR_BGR2RGB)
    x = cv2.resize(img_lumbar[60:600, 300:650, :],(256, 320), interpolation = cv2.INTER_AREA)

    os.chdir("/Users/mariamena/Desktop/Adamo/static/zona_a_tratar")
        
    blend = cv2.addWeighted(x, 0.5, contrast_img, 0.9, 0.0)
    plt.imshow(blend)
    plt.savefig('3.png')
    
    return