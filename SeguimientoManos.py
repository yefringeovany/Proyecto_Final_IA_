import math
import cv2
import mediapipe as mp

class DetectorManos:
    def __init__(self, mode=False, maxManos=2, Confdeteccion=0.5, Confsegui=0.5):
        self.mode = mode
        self.maxManos = int(maxManos)
        self.Confdeteccion = float(Confdeteccion)
        self.Confsegui = float(Confsegui)

        self.mpManos = mp.solutions.hands
        self.manos = self.mpManos.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxManos,
            min_detection_confidence=self.Confdeteccion,
            min_tracking_confidence=self.Confsegui
        )
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    def encontramos(self, frame, dibujar=True):
        imgColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgColor)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpManos.HAND_CONNECTIONS)
        return frame

    def encontrarPosicion(self, frame, ManoNum=0, dibujar=True):
        xLista = []
        yLista = []
        bbox = []
        self.lista = []

        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                xLista.append(cx)
                yLista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujar:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

            xmin, xmax = min(xLista), max(xLista)
            ymin, ymax = min(yLista), max(yLista)
            bbox = xmin, ymin, xmax, ymax

            if dibujar:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lista, bbox

    def dedosArriba(self):
        dedos = []
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        return dedos

    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        x1, y1 = self.lista[p1][1:]
        x2, y2 = self.lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        longitud = math.hypot(x2 - x1, y2 - y1)
        return longitud, frame, [x1, y1, x2, y2, cx, cy]

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        exit()

    detector = DetectorManos()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir frame. Exiting ...")
            break

        frame = detector.encontramos(frame)
        lista, bbox = detector.encontrarPosicion(frame)

        if len(lista) != 0:
            print(lista[4])

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
