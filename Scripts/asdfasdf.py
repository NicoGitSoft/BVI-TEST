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

# Muestras de temperatura
chipTemperatures = []       # Muestras de la temperatura del chip
cpuTemperature = []         # Muestras de la temperatura del CPU
max6675Temperature = []     # Muestras de la temperatura del sensor DHT22

# Declaración de variables
frame_start_time = 0        # tiempo de inicio de captura de frame
frames_timer = 0            # tiempo de captura de frame
frames_counter = 0          # contador de frames

# Rutas de los modelos YOLOv8n, YOLOv7t, YOLOv7s, YOLOv5n
SCRIPT_DIR = Path(__file__).resolve().parent

SingsYOLOv8n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n_openvino_2021.4_6shave.blob")
SingsYOLOv7s_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s_openvino_2021.4_6shave.blob")
SingsYOLOv7t_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t_openvino_2021.4_6shave.blob")
SingsYOLOv5n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n_openvino_2021.4_6shave.blob")

SingsYOLOv8n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n.json")
SingsYOLOv7t_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t.json")
SingsYOLOv7s_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s.json")
SingsYOLOv5n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n.json")

YOLOv8n_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8n/YOLOv8n_openvino_2021.4_6shave.blob")
YOLOv8s_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8s/YOLOv8s_openvino_2021.4_6shave.blob")
YOLOv7s_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7s/YOLOv7s_openvino_2021.4_6shave.blob")
YOLOv7t_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7t/YOLOv7t_openvino_2021.4_6shave.blob")
YOLOv5n_MODEL = str(SCRIPT_DIR / "../Models/YOLO/YOLOv5n/YOLOv5n_openvino_2021.4_6shave.blob")

YOLOv8n_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8n/YOLOv8n.json")
YOLOv8s_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv8s/YOLOv8s.json")
YOLOv7s_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7s/YOLOv7s.json")
YOLOv7t_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv7t/YOLOv7t.json")
YOLOv5n_CONFIG = str(SCRIPT_DIR / "../Models/YOLO/YOLOv5n/YOLOv5n.json")


# Listas de los modelos y sus respectivas configuraciones
SingsYOLO_MODELS = [SingsYOLOv7t_MODEL, SingsYOLOv5n_MODEL] 
SingsYOLO_CONFIGS = [SingsYOLOv7t_CONFIG, SingsYOLOv5n_CONFIG]

visualize = True
max_frames = 100

for j in range(len(SingsYOLO_MODELS)):

    model_name = SingsYOLO_CONFIGS[j][-17:-5]

    ################## TEST USANDO SOLO YOLO ##################
    data_only_yolo = DepthYoloHandTracker(
        use_depth = False,
        use_hand = False, 
        yolo_model = SingsYOLO_MODELS[j], 
        yolo_configurations = SingsYOLO_CONFIGS[j])

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
        else:
            no_detections += 1

        # Contador de FPS
        frames_counter += 1
        current_time = time.time()
        if (current_time - frames_timer) > 1:
            fps.append( frames_counter / (current_time - frames_timer) )
            frames_counter = 0
            frames_timer = current_time
        #print(model_name, "Frame: ", str(i), "FPS: {:.2f}".format(fps[-1]), end='\r')

    # Cerrar conexión con la cámara y mostrar resultados
    data_only_yolo.exit()
    only_yolo_fps = sum(fps) / len(fps)

    ################## TEST USANDO YOLO + DEPTH ##################
    data_yolo_depth = DepthYoloHandTracker(
        use_depth = True,
        use_hand = False,
        yolo_model = SingsYOLO_MODELS[j],
        yolo_configurations = SingsYOLO_CONFIGS[j])

    successful_detections = 0
    failed_detections = 0
    no_detections = 0

    loop_start_time = time.time()
    for i in range(max_frames):
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = data_yolo_depth.next_frame()

        # Contador de detecciones
        if len(yolo_detections) > 0:
            for detection in yolo_detections:
                label_number = detection.label              
                if label_number == 5:
                    successful_detections += 1
                else:
                    failed_detections += 1
        else:
            no_detections += 1

        # Contador de FPS
        frames_counter += 1
        current_time = time.time()
        if (current_time - frames_timer) > 1:
            fps.append( frames_counter / (current_time - frames_timer) )
            frames_counter = 0
            frames_timer = current_time
        #print(model_name, "Frame: ", str(i), "FPS: {:.2f}".format(fps[-1]), end='\r')
        
    # Cerrar conexión con la cámara y mostrar resultados
    data_yolo_depth.exit()
    yolo_depth_fps = sum(fps) / len(fps)

    ################## TEST USANDO YOLO + DEPTH + HAND.blob ##################
    data_yolo_depth_hand = DepthYoloHandTracker(
        use_depth = True,
        use_hand = True,
        use_mediapipe=False,
        yolo_model = SingsYOLO_MODELS[j],
        yolo_configurations = SingsYOLO_CONFIGS[j])

    successful_detections = 0
    failed_detections = 0
    no_detections = 0
    
    loop_start_time = time.time()
    for i in range(max_frames):
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = data_yolo_depth_hand.next_frame()

        # Contador de detecciones
        if len(yolo_detections) > 0:
            for detection in yolo_detections:
                label_number = detection.label              
                if label_number == 5:
                    successful_detections += 1
                else:
                    failed_detections += 1
        else:
            no_detections += 1

        # Contador de FPS
        frames_counter += 1
        current_time = time.time()
        if (current_time - frames_timer) > 1:
            fps.append( frames_counter / (current_time - frames_timer) )
            frames_counter = 0
            frames_timer = current_time
        #print(model_name, "Frame: ", str(i), "FPS: {:.2f}".format(fps[-1]), end='\r')
        
    # Cerrar conexión con la cámara y mostrar resultados
    data_yolo_depth_hand.exit()
    yolo_depth_hand_fps = sum(fps) / len(fps)

    ################## TEST USANDO YOLO + DEPTH + mediapipe ##################

    data_yolo_depth_mediapipe = DepthYoloHandTracker(
        use_depth = True,
        use_hand = True,
        use_mediapipe=True,
        yolo_model = SingsYOLO_MODELS[j],
        yolo_configurations = SingsYOLO_CONFIGS[j])

    successful_detections = 0
    failed_detections = 0
    no_detections = 0

    loop_start_time = time.time()
    for i in range(max_frames):
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = data_yolo_depth_mediapipe.next_frame()

        # Contador de detecciones
        if len(yolo_detections) > 0:
            for detection in yolo_detections:
                label_number = detection.label              
                if label_number == 5:
                    successful_detections += 1
                else:
                    failed_detections += 1
        else:
            no_detections += 1

        # Contador de FPS
        frames_counter += 1
        current_time = time.time()
        if (current_time - frames_timer) > 1:
            fps.append( frames_counter / (current_time - frames_timer) )
            frames_counter = 0
            frames_timer = current_time
        #print(model_name, "Frame: ", str(i), "FPS: {:.2f}".format(fps[-1]), end='\r')
        
    # Cerrar conexión con la cámara y mostrar resultados
    data_yolo_depth_mediapipe.exit()
    yolo_depth_mediapipe_fps = sum(fps) / len(fps)

    
    print("\nFPS promedio para solo YOLO: {:.2f}".format(only_yolo_fps))
    print("FPS promedio para YOLO + DEPTH: {:.2f}".format(yolo_depth_fps))
    print("FPS promedio para YOLO + DEPTH + HAND.blob: {:.2f}".format(yolo_depth_hand_fps))
    print("FPS promedio para YOLO + DEPTH + mediapipe: {:.2f}".format(yolo_depth_mediapipe_fps))
    print("Test finalizado para el modelo", model_name, "en", time.time() - loop_start_time, "segundos \n")



# FPS promedio para solo YOLO: 13.19
# FPS promedio para YOLO + DEPTH: 12.68
# FPS promedio para YOLO + DEPTH + HAND.blob: 9.20
# FPS promedio para YOLO + DEPTH + mediapipe: 9.58
# Test finalizado para el modelo SingsYOLOv7t en 10.807883262634277 segundos 
# 
# FPS promedio para solo YOLO: 10.47
# FPS promedio para YOLO + DEPTH: 10.97
# FPS promedio para YOLO + DEPTH + HAND.blob: 9.60
# FPS promedio para YOLO + DEPTH + mediapipe: 10.04
# Test finalizado para el modelo SingsYOLOv5n en 8.000313758850098 segundos

# FPS promedio para solo YOLO: 12.97
# FPS promedio para YOLO + DEPTH: 12.55
# FPS promedio para YOLO + DEPTH + HAND.blob: 8.54
# FPS promedio para YOLO + DEPTH + mediapipe: 9.22
# Test finalizado para el modelo SingsYOLOv7t en 9.47022008895874 segundos 
# 
# FPS promedio para solo YOLO: 10.17
# FPS promedio para YOLO + DEPTH: 10.70
# FPS promedio para YOLO + DEPTH + HAND.blob: 9.39
# FPS promedio para YOLO + DEPTH + mediapipe: 9.86
# Test finalizado para el modelo SingsYOLOv5n en 6.888955593109131 segundos




# RASPBERRY PI 4B 4GB
# FPS promedio para solo YOLO: 13.16
# FPS promedio para YOLO + DEPTH: 12.73
# FPS promedio para YOLO + DEPTH + HAND.blob: 9.52
# FPS promedio para YOLO + DEPTH + mediapipe: 8.31
# Test finalizado para el modelo SingsYOLOv7t en 18.395132541656494 segundos 