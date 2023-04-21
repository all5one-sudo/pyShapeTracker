import cv2
import numpy as np

# Leemos la imagen
img = cv2.imread('src/shapesExample.png')
# La pasamos a escala de grises
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
	# Encontramos el centro de la imagen
	M = cv2.moments(contour)
	if M['m00'] != 0.0:
		x = int(M['m10']/M['m00'])
		y = int(M['m01']/M['m00'])
	coordinates = 'x: ' + str(x) + ', y: ' +  str(y)
	# Se reconoce el triángulo
	if len(approx) == 3:
		cv2.putText(img, 'T', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		cv2.putText(img, coordinates, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		cv2.drawContours(img, [contour], 0, (120, 255, 255), 5)

# Se muestra la imagen luego de reconocer los contornos
cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
