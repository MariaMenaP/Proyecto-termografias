import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def info_areas (imagen):
    
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

    # Ahora se crea una matriz NumPy para los valores superiores e inferiores de los colores. 
    redBajo1 = np.array([0, 100, 20], np.uint8)
    redAlto1 = np.array([8, 255, 255], np.uint8)

    redBajo2=np.array([0, 100, 10], np.uint8)
    redAlto2=np.array([10, 255, 250], np.uint8)

    #redBajo2=np.array([156, 100, 20], np.uint8)
    #redAlto2=np.array([180, 255, 255], np.uint8)

    lower_orange = np.array([8, 100, 20], np.uint8)
    upper_orange  = np.array([20, 255, 255], np.uint8)

    lower_yellow = np.array([20, 100, 20], np.uint8)
    upper_yellow  = np.array([45, 255, 255], np.uint8)

    lower_green = np.array([45, 100, 20], np.uint8)
    upper_green  = np.array([95, 170, 255], np.uint8)

    lower_cian = np.array([80, 100, 20], np.uint8)
    upper_cian  = np.array([115, 250, 255], np.uint8)

    colores = list()

    #Creamos la imagen de cada color(en escala de grises)
    a_tratar = cv2.inRange(hsv_img, redBajo2, redAlto2)
    colores.append(a_tratar)
    red = cv2.inRange(hsv_img, redBajo1, redAlto1)
    colores.append(red)
    orange = cv2.inRange(hsv_img, lower_orange, upper_orange)
    colores.append(orange)
    yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    colores.append(yellow)
    green = cv2.inRange(hsv_img, lower_green, upper_green)
    colores.append(green)
    cian = cv2.inRange(hsv_img, lower_cian, upper_cian)
    colores.append(cian)

    indice = ["Zona a tratar", "Red", "Orange", "Yellow","Green",  "Blue"]
    l_cmap = ["Reds","Reds", "Oranges", "Wistia", "YlGn",  "GnBu"]
    subplot = [1,2,3,4,5,6]

    # Encontramos en ella los contornos:
    ## Encuentra el umbral

    plt.figure(figsize=(25, 15))

    for num,cmap,i,color in zip(subplot,l_cmap,indice,colores):
        retval, thresh = cv2.threshold(color, 80, 250, 0) #variar estos numeros y ver si para todas funciona esos rangos

        varias_areas = list()

        ## Usa el findContours() que toma la imagen 
        img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        if len(img_contours)==0:
            
            # Generate the mask using np.zeros
            mask = np.zeros(color.shape[:2], np.uint8)
            
            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(contrast_img.copy(), contrast_img.copy(), mask=mask)

            plt.subplot(3,2,num)
            plt.imshow(new_img, cmap= cmap)
            plt.colorbar() 
            plt.title("-------------------------------------"+"\nNo se observa area de {}".format(i), fontsize = 15)
            plt.tight_layout()

        elif len(img_contours)==1:
            x = cv2.contourArea(img_contours[0])

            # Generate the mask using np.zeros
            mask = np.zeros(color.shape[:2], np.uint8)

            # Dibujar los contornos
            cv2.drawContours(mask, img_contours,-1, 255, -1)

            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(contrast_img.copy(), contrast_img.copy(), mask=mask)

            plt.subplot(3,2,num)
            plt.imshow(new_img, cmap= cmap)
            plt.colorbar()
            plt.title("-------------------------------------"+"\nEl area del color {} es de {}".format(i,x), fontsize = 15)
            plt.tight_layout()

        elif len(img_contours)==2:
            for j in range(len(img_contours)):
                y = cv2.contourArea(img_contours[j])
                varias_areas.append(y)

            # Generate the mask using np.zeros
            mask = np.zeros(color.shape[:2], np.uint8)

            # Dibujar los contornos
            cv2.drawContours(mask, img_contours,-1, 255, -1)

            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(contrast_img.copy(), contrast_img.copy(), mask=mask)


            plt.subplot(3,2,num)
            plt.imshow(new_img, cmap= cmap)
            plt.colorbar()
            plt.title("\n".join(["El area {} del color {} es de {}".format((p+1),i,a) for p,a in enumerate(varias_areas)])+
                      "\n-------------------------------------"+
                      "\nEl area total del color {} es de {}".format(i,sum(varias_areas)),fontsize = 15)
            plt.tight_layout()

        elif len(img_contours)==3:
            for j in range(len(img_contours)):
                y = cv2.contourArea(img_contours[j])
                varias_areas.append(y)

            # Generate the mask using np.zeros
            mask = np.zeros(color.shape[:2], np.uint8)

            # Dibujar los contornos
            cv2.drawContours(mask, img_contours,-1, 255, -1)

            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(contrast_img.copy(), contrast_img.copy(), mask=mask)

            plt.subplot(3,2,num)
            plt.imshow(new_img, cmap= cmap)
            plt.colorbar()
            plt.title("\n".join(["El area {} del color {} es de {}".format((p+1),i,a) for p,a in enumerate(varias_areas)])+
                      "\n-------------------------------------"+
                      "\nEl area total del color {} es de {}".format(i,sum(varias_areas)),fontsize = 15)
            plt.tight_layout()

        elif len(img_contours)==4:
            for j in range(len(img_contours)):
                y = cv2.contourArea(img_contours[j])
                varias_areas.append(y)

            # Generate the mask using np.zeros
            mask = np.zeros(color.shape[:2], np.uint8)

            # Dibujar los contornos
            cv2.drawContours(mask, img_contours,-1, 255, -1)

            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(contrast_img.copy(), contrast_img.copy(), mask=mask)

            plt.subplot(3,2,num)
            plt.imshow(new_img, cmap= cmap)
            plt.colorbar()
            plt.title("\n".join(["El area {} del color {} es de {}".format((p+1),i,a) for p,a in enumerate(varias_areas)])+
                      "\n-------------------------------------"+
                      "\nEl area total del color {} es de {}".format(i,sum(varias_areas)),fontsize = 15)
            plt.tight_layout()

        elif len(img_contours)==5:
            for j in range(len(img_contours)):
                y = cv2.contourArea(img_contours[j])
                varias_areas.append(y)

            # Generate the mask using np.zeros
            mask = np.zeros(color.shape[:2], np.uint8)

            # Dibujar los contornos
            cv2.drawContours(mask, img_contours,-1, 255, -1)

            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(contrast_img.copy(), contrast_img.copy(), mask=mask)

            plt.subplot(3,2,num)
            plt.imshow(new_img, cmap= cmap)
            plt.colorbar()
            plt.title("\n".join(["El area {} del color {} es de {}".format((p+1),i,a) for p,a in enumerate(varias_areas)])+
                      "\n-------------------------------------"+
                     "\nEl area total del color {} es de {}".format(i,sum(varias_areas)),fontsize = 15)
            plt.tight_layout()

        else:
            for j in range(len(img_contours)):
                y = cv2.contourArea(img_contours[j])
                varias_areas.append(y)

            # Generate the mask using np.zeros
            mask = np.zeros(color.shape[:2], np.uint8)

            # Dibujar los contornos
            cv2.drawContours(mask, img_contours,-1, 255, -1)

            # Aplicar el operador bitwise_and
            new_img = cv2.bitwise_and(contrast_img.copy(), contrast_img.copy(), mask=mask)

            plt.subplot(3,2,num)
            plt.imshow(new_img, cmap= cmap)
            plt.colorbar()
            plt.title("Hay mas de 5 areas en {}.".format(i)+"\n-------------------------------------"+"\nEl area total del color {} es de {}".format(i,sum(varias_areas)),fontsize = 15)
            plt.tight_layout()

    os.chdir("/Users/mariamena/Desktop/Adamo/static/zona_a_tratar")
    plt.savefig('2.png')

    return 