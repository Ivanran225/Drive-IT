"""Coment section:
Este codigo esta god
Hace lo de bgr y tono de grises para sacar color blanco y saca rango de colores para sacar amarillo (nose como funciona pero funciona)
Tambien hay un coso que marca con cv2.lines las lineas amarillas
"""
from mss import mss
import cv2
from PIL import Image
import numpy as np
#from time import sleep 
#from time import time
mon = {'top': 30, 'left':0, 'width':720, 'height':480}
sct = mss()

def scren_record():
    sct_img = sct.grab(mon)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    global img_bgr
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #cv2.imshow('test', np.array(img_bgr))

def scale_img():
    global image, frame
    image = np.array(img_bgr)
    frame = np.array(img_bgr)
    lineas = np.array(img_bgr)

    scale_percent = 60 # percent of original size
    width = int(720 * scale_percent / 100)
    height = int(480 * scale_percent / 100)
    dim = (width, height)

    #dibujar lineas de referencia
    #hoprizontal rojo
    cv2.line(lineas,(240,0),(240,480),(0,0,255),5)
    cv2.line(lineas,(480,0),(480,480),(0,0,255),5)
    #vertical azul
    cv2.line(lineas,(0,400),(720,400),(255),5)

    resized = cv2.resize(lineas, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Resized image", resized)

def split_image_detectar_amarillo():
    # difference between red and green
    # this will likely discard white
    b,g,r = cv2.split(image)
    dif = cv2.subtract(g, b)
    # cv2.imshow("diff", dif)

    ret, thresh = cv2.threshold(dif, 70, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Amarillo", thresh)
    edges = cv2.Canny(thresh, 75, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=300)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 250,0), 5)
            #cv2.imshow("Lienas amarillas", frame)

def detectar_calcular_lineas():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,220, 255, cv2.THRESH_BINARY)

    izquierda = 1
    derecha = 1

    for j in range(len(thresh[400])):
        if thresh[400][j]==255:
            if j < 240:
                izquierda = j
            elif j > 480:
                derecha = j

    aprom = 1
    cantsum = 1
    promedio = 1
    aprom1 = 1
    cantsum1 = 1
    promedio1 = 1

    aprom = aprom + izquierda
    cantsum = cantsum + 1
    promedio = aprom/cantsum
    
    aprom1 = aprom1 + derecha
    cantsum1 = cantsum1 + 1
    promedio1 = aprom1/cantsum1
    adividir = promedio + promedio1
    resultado = 1
    resultado = adividir/2

    print(int(resultado))

    # se mueve de 140 a 230
    #suma los valores a una variable y la divide por la cantidad de valores que se sumaron y tenes el promedio pa

    cv2.imshow("Blanco", thresh)

def changed_perspective():
    pts1 = np.float32([[240,140], [480,140], [1,480], [720, 480]])
    pts2 = np.float32([[1, 1], [480, 1], [1, 480], [720, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(np.array(img_bgr), matrix, (720, 480))
    # Wrap the transformed image

while 1:
    scren_record()

    scale_img()

    #split_image_detectar_amarillo()

    detectar_calcular_lineas()

    #changed_perspective()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    #cv2.imshow('Changed perspective', result) # Transformed Capture