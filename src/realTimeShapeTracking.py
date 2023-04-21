import cv2
import numpy as np
import time

# Función que rastrea la forma
def detectShape():
    # Se inicializa la transmisión de la cámara web por defecto
    # En caso de contar con varias cámaras, se debe cambiar el valor de 0 por el correspondiente
    # en el array de dispositivos de entrada
    sampleVideo = cv2.VideoCapture(0)
    sampleVideo.set(cv2.CAP_PROP_FPS, 5)
    # Creamos una ventana
    cv2.namedWindow('Reconocimiento', cv2.WINDOW_AUTOSIZE)
    x = 0
    y = 0
    try:
        while True:
            # Leemos el último fotograma
            rc, img = sampleVideo.read()
            pressedKey = cv2.waitKey(2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Se setea el threshold para la imagen en escala de grises
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # Encontramos los contornos en la imagen
            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Para comprobar que el contorno a leer no es el primero, que es el que detecta por defecto openCV
            i = 0
            # En contours se almacenan las formas que se detectaron
            for contour in contours:
                # Ignoramos el primer contorno
                if i == 0:
                    i = 1
                    continue
                # Aproximamos los polígonos
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                # Obtenemos el área
                area = cv2.contourArea(contour)
                if (area >= 20):
                    # Encontramos el centro de la imagen
                    M = cv2.moments(contour)
                    if M['m00'] != 0.0:
                        x = int(M['m10']/M['m00'])
                        y = int(M['m01']/M['m00'])
                    coordinates = 'x: ' + str(x) + ', y: ' + str(y)
                    # Se reconoce el triángulo
                    if len(approx) == 3:
                        cv2.putText(img, 'T', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
                        cv2.putText(img, coordinates, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                        cv2.drawContours(img, [contour], 0, (120, 255, 255), 5)
                    cv2.imshow("Reconocimiento", img)
                #time.sleep(0.002)

    except KeyboardInterrupt as e:
        cv2.destroyAllWindows()
        exit(0)

if __name__ == '__main__':
    detectShape()