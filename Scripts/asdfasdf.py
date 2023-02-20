from Utilities2 import *
import cv2, serial, time, math, os, subprocess
import scipy.io as sio
import numpy as np

# Coordenadas de los vertices un bounding box
def Vertices(detection):        
    x1 = int(detection.xmin * width)
    x2 = int(detection.xmax * width)
    y1 = int(detection.ymin * height)
    y2 = int(detection.ymax * height)
    return x1, x2, y1, y2

##################### inicialización de variables #####################
times = []  # Muestras de los tiempos de ejecución del programa desde la primera captura de frame
fps = []    # Muestras de los fps de ejecución del programa desde la primera captura de frame
d = []      # Muestras de la distancias a la detecion más cercana con visión monocular
z = []      # Muestras de las distancias a los obstáculos en la ROI central con visión estereoscópica
h = []      # Muestras de las distancias horizontales en piexeles de la detección más cercana al centro de la imagen
v = []      # Muestras de las distancias verticales en piexeles de la detección más cercana al centro de la imagen 

# Muestras de los mensajes de activación de la interfaz de comunicación
haptic_messages  = []   # Muestras de los mensajes de activación del vibrador UP de la interfaz háptica
buzzer_messages  = []   # Muestras de los mensajes de activación del buzzer
nearest_labels = []     # Muestras de la etiquta de la deteción más cercana

# Muestras de temperatura
chipTemperatures = []       # Muestras de la temperatura del chip
cpuTemperature = []         # Muestras de la temperatura del CPU
max6675Temperature = []     # Muestras de la temperatura del sensor DHT22

# Declaración de variables
mentioned_object = False    # bandera booleana para evitar que se repita el nombre del objeto detectado
mentioned_hand_use = False  # bandera booleana para evitar que se repita el la mencion TTS del uso de la mano
frame_start_time = 0        # tiempo de inicio de captura de frame
frames_timer = 0            # tiempo de captura de frame
frames_counter = 0          # contador de frames
start_move_time = 0         # tiempo de inicio de movimiento
start_buzze_time = 0        # tiempo de inicio de sonido del buzzer

# Rutas de los modelos YOLOv8n, YOLOv7t, YOLOv7s, YOLOv5n
SCRIPT_DIR = Path(__file__).resolve().parent
SingsYOLOv8n_MODEL = str(SCRIPT_DIR / "../Models/Sings/YOLOv8n/YOLOv8n_openvino_2021.4_6shave.blob")
YOLOv8n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n_openvino_2021.4_6shave.blob")
SingsYOLOv7s_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s_openvino_2021.4_6shave.blob")
SingsYOLOv7t_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t_openvino_2021.4_6shave.blob")
SingsYOLOv5n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n_openvino_2021.4_6shave.blob")
SingsYOLOv8n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n.json")
YOLOv8n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/YOLOv8n/YOLOv8n.json")
SingsYOLOv7t_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t.json")
SingsYOLOv7s_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s.json")
SingsYOLOv5n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n.json")

# Listas de los modelos y sus respectivas configuraciones
SingsYOLO_MODELS = [SingsYOLOv8n_MODEL, SingsYOLOv7t_MODEL, SingsYOLOv5n_MODEL] 
SingsYOLO_CONFIGS = [SingsYOLOv8n_CONFIG, SingsYOLOv7t_CONFIG, SingsYOLOv5n_CONFIG]

visualize = True
max_frames = 1000

for i in range(len(SingsYOLO_MODELS)):

    model_name = SingsYOLO_CONFIGS[i].split('/')[-1].split('.')[0]

    data_only_yolo = DepthYoloHandTracker(
        use_depth=False,
        use_hand = False, 
        yolo_model = SingsYOLO_MODELS[i], 
        yolo_configurations = SingsYOLO_CONFIGS[i])

    ################## TEST DE YOLO ##################
    successful_detections = 0
    failed_detections = 0
    no_detections = 0

    loop_start_time = time.time()
    for i in range(max_frames):
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = data_only_yolo.next_frame()

        # Contador de detecciones
        if len(yolo_detections) > 0:
            for detection in yolo_detections:

                label_number = detection.label
                
                if label_number == 5:
                    successful_detections += 1
                else:
                    failed_detections += 1

                if visualize:
                    x1, x2, y1, y2 = int(detection.xmin*width), int(detection.xmax*width), int(detection.ymin*height), int(detection.ymax*height)
                    cv2.putText(frame, str(label_number), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
        else:
            no_detections += 1

        # Contador de FPS 
        frames_counter += 1
        current_time = time.time()
        times.append(current_time - loop_start_time) # Almacenar el tiempo de ejecución
        if (current_time - frames_timer) > 1:
            fps.append( frames_counter / (current_time - frames_timer) )
            frames_counter = 0
            frames_timer = current_time

        if visualize: cv2.imshow("frame", frame)

        # Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q}
        if cv2.waitKey(1) in [27, 32, ord('q')]:
            break

        # Imprimir el estado de la detección
        print("Frame: ", str(i),
            model_name, "FPS: {:.2f}".format(fps[-1]),
            "successful detections: ", successful_detections,
            "failed detections: ", failed_detections,
            "no detections: ", no_detections,
            sep='\t' , end='\r')

    data_only_yolo.exit()
    print(str(i), model_name, "FPS: {:.2f}".format(np.mean(fps)), sep='\t' , end='\n')


    