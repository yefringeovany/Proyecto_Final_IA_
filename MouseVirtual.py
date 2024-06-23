import cv2
import numpy as np
import autopy
import SeguimientoManos

# Configuración de la cámara
anchocam, altocam = 640, 480
cuadro = 100
anchopanta, altopanta = autopy.screen.size()
sua = 5
pubix, pubiy = 0, 0
cubix, cubiy = 0, 0

# Inicialización de la cámara
cap = cv2.VideoCapture(0)
cap.set(3, anchocam)
cap.set(4, altocam)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Declaramos el detector
detector = SeguimientoManos.DetectorManos(maxManos=1)

# Crear la ventana y redimensionarla
cv2.namedWindow("Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mouse", 1600, 900)

# Variables para el contador de distancia
distancia_total = 0

print("Iniciando bucle principal...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir frame. Exiting ...")
        break

    frame = detector.encontramos(frame)
    lista, bbox = detector.encontrarPosicion(frame)

    # Inicializar dedos con un valor por defecto
    dedos = [0, 0, 0, 0, 0]

    if len(lista) != 0:
        x1, y1 = lista[8][1:]
        x2, y2 = lista[12][1:]

        dedos = detector.dedosArriba()
        cv2.rectangle(frame, (cuadro, cuadro), (anchocam - cuadro, altocam - cuadro), (0, 255, 0), 2)

        if dedos[1] == 1 and dedos[2] == 0:
            x3 = np.interp(x1, (cuadro, anchocam - cuadro), (0, anchopanta))
            y3 = np.interp(y1, (cuadro, altocam - cuadro), (0, altopanta))
            cubix = pubix + (x3 - pubix) / sua
            cubiy = pubiy + (y3 - pubiy) / sua

            autopy.mouse.move(anchopanta - cubix, cubiy)
            cv2.circle(frame, (x1, y1), 10, (0, 255, 255), cv2.FILLED)

            # Calcula la distancia movida
            distancia_movida = np.sqrt((cubix - pubix) ** 2 + (cubiy - pubiy) ** 2)
            distancia_total += distancia_movida

            # Actualiza las posiciones anteriores
            pubix, pubiy = cubix, cubiy

        if dedos[1] == 1 and dedos[2] == 1:
            longitud, frame, linea = detector.distancia(8, 12, frame)
            if longitud < 30:
                cv2.circle(frame, (linea[4], linea[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Muestra el contador de distancia en la cámara
    cv2.putText(frame, f"Factor de Movimiento: {int(distancia_total)} px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Añadir más elementos visuales
    cv2.putText(frame, "Modo: " + (
        "Mover" if dedos[1] == 1 and dedos[2] == 0 else "Clic" if dedos[1] == 1 and dedos[2] == 1 else "Sin accion"),
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Cambié el color a verde (0, 255, 0)

    # Mostrar un contorno en las manos detectadas
    if bbox:
        cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (255, 0, 0), 2)

    cv2.imshow("Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
