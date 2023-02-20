from Utilities2 import *
import cv2, serial, time, math, os, subprocess
import scipy.io as sio
import numpy as np

def TEST(Device):

    # Inicialización de variables
    frames_counter = 0
    initial_timed_time = 0
    successful_detections = 0
    failed_detections = 0
    no_detections = 0

    # Inicialización del loop
    loop_start_time = time.time()
    for i in range(max_frames):
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = Device.next_frame()

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
        timed_time = time.time() - initial_timed_time
        if timed_time > 1:
            fps.append( frames_counter / timed_time )
            frames_counter = 0
            initial_timed_time = current_time
        
        # Mostrar resultados por pantalla
        print(f"Modelo {model_name} - FPS: {fps[-1]:.2f} - Detecciones: {successful_detections} - Falsos positivos: {failed_detections} - No detecciones: {no_detections}", end="\r")

    # Cerrar conexión con la cámara y mostrar resultados
    Device.exit()
    average_fps = sum(fps) / len(fps)

    return average_fps, successful_detections, failed_detections, no_detections


##################### inicialización de variables #####################
times = []  # Muestras de los tiempos de ejecución del programa desde la primera captura de frame
fps = []    # Muestras de los fps de ejecución del programa desde la primera captura de frame

# Muestras de temperatura
chipTemperatures = []       # Muestras de la temperatura del chip
cpuTemperature = []         # Muestras de la temperatura del CPU
max6675Temperature = []     # Muestras de la temperatura del sensor DHT22

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
SingsYOLO_MODELS = [ SingsYOLOv8n_MODEL] 
SingsYOLO_CONFIGS = [ SingsYOLOv8n_CONFIG]

visualize = True
max_frames = 1000
##################### Ejecución de los modelos #####################
for j in range(len(SingsYOLO_MODELS)):

    model_name = SingsYOLO_CONFIGS[j][-17:-5]
    ################## TEST USANDO SOLO YOLO ##################
    only_yolo = DepthYoloHandTracker(
        use_depth = False,
        use_hand = False, 
        yolo_model = SingsYOLO_MODELS[j], 
        yolo_configurations = SingsYOLO_CONFIGS[j])
    data_only_yolo = TEST(only_yolo)
    ################## TEST USANDO YOLO + DEPTH ##################
    data_yolo_depth = DepthYoloHandTracker(
        use_depth = True,
        use_hand = False,
        yolo_model = SingsYOLO_MODELS[j],
        yolo_configurations = SingsYOLO_CONFIGS[j])
    data_yolo_depth = TEST(data_yolo_depth)
    ################## TEST USANDO YOLO + DEPTH + HAND.blob ##################
    data_yolo_depth_hand = DepthYoloHandTracker(
        use_depth = True,
        use_hand = True,
        use_mediapipe=False,
        yolo_model = SingsYOLO_MODELS[j],
        yolo_configurations = SingsYOLO_CONFIGS[j])
    data_yolo_depth_hand = TEST(data_yolo_depth_hand)
    ################## TEST USANDO YOLO + DEPTH + mediapipe ##################
    data_yolo_depth_mediapipe = DepthYoloHandTracker(
        use_depth = True,
        use_hand = True,
        use_mediapipe=True,
        yolo_model = SingsYOLO_MODELS[j],
        yolo_configurations = SingsYOLO_CONFIGS[j])
    data_yolo_depth_mediapipe = TEST(data_yolo_depth_mediapipe)
    ################## RESULTADOS POR MODELO ##################
    os.system('clear')
    print(model_name + " TEST RESULTS (AverageFPS, SuccessfulDetections, FailedDetections, noDetections)")
    print("Solo YOLO: ", data_only_yolo)
    print("YOLO + DEPTH: ", data_yolo_depth)
    print("YOLO + DEPTH + HAND.blob: ", data_yolo_depth_hand)
    print("YOLO + DEPTH + mediapipe: ", data_yolo_depth_mediapipe) 


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
#
# FPS promedio para solo YOLO: 15.05
# FPS promedio para YOLO + DEPTH: 15.12
# FPS promedio para YOLO + DEPTH + HAND.blob: 8.18
# FPS promedio para YOLO + DEPTH + mediapipe: 6.48
# Test finalizado para el modelo SingsYOLOv5n en 25.22651219367981 segundos 

# SingsYOLOv5n TEST RESULTS (AverageFPS, Successful Detections, Failed Detections)
# Solo YOLO:  (15.672798937832647, 77, 7, 16)
# YOLO + DEPTH:  (15.065592922996133, 77, 4, 19)
# YOLO + DEPTH + HAND.blob:  (8.12590883272291, 88, 1, 11)
# YOLO + DEPTH + mediapipe:  (9.08966711943187, 77, 8, 15)
# SingsYOLOv5n TEST RESULTS (AverageFPS, Successful Detections, Failed Detections)
# Solo YOLO:  (18.428572825216673, 978, 9, 13)
# YOLO + DEPTH:  (17.710747915439462, 977, 7, 16)
# YOLO + DEPTH + HAND.blob:  (10.407295514840431, 994, 3, 3)
# YOLO + DEPTH + mediapipe:  (11.6001992307358, 977, 6, 17)

# SingsYOLOv8n TEST RESULTS (AverageFPS, SuccessfulDetections, FailedDetections, noDetections)
# Solo YOLO:  (14.205662813280256, 0, 0, 1000)
# YOLO + DEPTH:  (13.97212681490595, 0, 0, 1000)
# YOLO + DEPTH + HAND.blob:  (6.434149805453163, 0, 0, 1000)
# YOLO + DEPTH + mediapipe:  (7.481393394874306, 0, 0, 1000)