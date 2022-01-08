import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io
import os

def preprocesamiento (imagen):
    
    img = imagen.copy()

    # 1 - Recortamos la imagen
    imagen = img[180:,:,:]

    # 2 - Median Blur y Aumento de contraste
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

    # Dibujamos una linea para cerrar contornos superiores. 
    contrast_img = cv2.line(contrast_img, 
                     pt1 = (0,0), 
                     pt2 = (256,0), 
                     color =(0, 255, 0), 
                     thickness=1, 
                     lineType=8)


    #-------------------------------   
    # DETECCIÓN DE COLOR
    # Detectemos el color ROJO de una imagen

    # Convertir la imagen en HSV usando cvtColor()
    hsv_img = cv2.cvtColor(contrast_img.copy(), cv2.COLOR_RGB2HSV)

    # Ahora se crea una matriz NumPy para los valores rojos inferiores y los valores rojos 
    # superiores
    redBajo1 = np.array([0, 100, 20], np.uint8)
    redAlto1 = np.array([8, 255, 255], np.uint8)
    redBajo2=np.array([175, 100, 20], np.uint8)
    redAlto2=np.array([179, 255, 255], np.uint8)

    #Creamos la imagen de los rojos(en escala de grises)
    masking = cv2.inRange(hsv_img, redBajo1, redAlto1)
    masking2 = cv2.inRange(hsv_img, redBajo2, redAlto2)

    blend = cv2.addWeighted(masking, 0.8, masking2, 0.8, 0.0)


    # Encontramos en ella los contornos:
    ## Encuentra el umbral
    retval, thresh = cv2.threshold(blend, 80, 250, 0) #variar estos numeros y ver si para todas funciona esos rangos

    ## Usa el findContours() que toma la imagen 
    img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(img_contours)==0:
        print("No se observa zona a tratar")
        plt.imshow(img)
        plt.show()
        return

    # Elimino dibujar los contornos en esta parte. 


    # 4 - Obtener contornos de nuevo
    ## Volvemos a pasar la imagen a escala de grises
    gray_img = cv2.cvtColor(contrast_img, cv2.COLOR_RGB2GRAY)

    ## Encuentra el umbral
    retval, thresh = cv2.threshold(gray_img, 80, 250, 0) #variar estos numeros y ver si para todas funciona esos rangos

    ## Usa el findContours() que toma la imagen 
    img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ## Dibuje los contornos en la imagen usando el método drawContours()
    cv2.drawContours(contrast_img, img_contours, -1, (0, 255, 0), 1, cv2.LINE_AA)


    # 5 - Marcar puntos de los extremos:
    img_contours = sorted(img_contours, key=cv2.contourArea)
    c = img_contours[-2]

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    recorte = contrast_img[extTop[1]:extBot[1],extLeft[0]:extRight[0],:]

    # 6 - Marcamos el centroide:
    # escala de grises
    gray_image = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

    # pasamos a binario la imagen
    ret,thresh = cv2.threshold(gray_image,127,255,0)

    # calculamos los momentos
    M = cv2.moments(thresh)

    # calculamos las coordenadas del centro
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    radio = 3
    # dibujamos el centroide
    cv2.circle(recorte, (cX, cY), radio, (255, 255, 255), -1)
    centroide = (extLeft[0]+cX,extTop[1]+cY)



    # 7 - Definimos los puntos que nos interesan de la zona y dibujamos la línea:
    L = np.array(extLeft )
    R = np.array(extRight)
    T = np.array(extTop)
    B = np.array(extBot)
    C = np.array(centroide)

    LC =tuple(abs(L-C))
    RC =tuple(abs(R-C))
    TC =tuple(abs(C-T))
    BC =tuple(abs(C-B))

    # dibujamos los puntos para el tratamiento:
    if (tuple(sum(x) for x in zip(TC,BC))[1])>(tuple(sum(x) for x in zip(RC,LC))[0]):
        print("De Top a Bot")
        img_pnt = cv2.circle(contrast_img, extTop, radio, (255, 0, 0), -1)
        img_pnt = cv2.circle(img_pnt, extBot, radio, (255, 255, 0), -1)

        # - Dibujar la línea de trabajo
        punto_inicial = extTop
        punto_final = extBot

        img_linea = cv2.line(img[180:,:,:], 
                     extTop, 
                     extBot, 
                     (0, 255, 0), 
                     thickness=2, 
                     lineType=8)


    else:
        img_pnt = cv2.circle(contrast_img, extLeft, radio, (0, 0, 255), -1)
        img_pnt = cv2.circle(img_pnt, extRight, radio, (0, 255, 0), -1)

        if (extRight[0] > centroide[0] > extLeft[0])&(extRight[1] < centroide[1] < extLeft[1])or(extRight[0] > centroide[0] > extLeft[0])&(extLeft[1] < centroide[1] < extRight[1]):
        # - Dibujar la línea de trabajo
            print("De Left a Right")
            punto_inicial = extLeft
            punto_final = extRight

            img_linea = cv2.line(img[180:,:,:], 
                         punto_inicial, 
                         punto_final, 
                         (0, 255, 0), 
                         thickness=2, 
                         lineType=8)

        elif LC > RC:
            print("De Left a Right")
            punto_inicial = extLeft
            punto_final = (centroide[0]+(centroide[0]-extLeft[0]),centroide[1]+(centroide[1]-extLeft[1]))

            img_linea = cv2.line(img[180:,:,:], 
                     punto_inicial, 
                     punto_final, 
                     (0, 255, 0), 
                     thickness=2, 
                     lineType=8)

        elif LC < RC:
            print("De Right a Left")
            punto_inicial = extRight
            punto_final = (centroide[0]-(extRight[0]-centroide[0]),centroide[1]-(extRight[1]-centroide[1]))

            img_linea = cv2.line(img[180:,:,:], 
                     punto_inicial, 
                     punto_final, 
                     (0, 255, 0), 
                     thickness=2, 
                     lineType=8)

        else:
            print("caso extraño")

    os.chdir("/Users/mariamena/Desktop/Adamo/static/zona_a_tratar")
            
    # display the image
    plt.imshow(img)
    plt.title("Punto inicial: {} \n Punto final: {}".format((punto_inicial[0],punto_inicial[1]+180),(punto_final[0],punto_final[1]+180)))
    plt.savefig('1.png')
    

    return 
