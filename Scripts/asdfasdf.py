from Utilities2 import *
from prettytable import PrettyTable
import time, os

def TEST(Device):

    # Inicialización de variables
    frames_counter = 0
    initial_timed_time = 0
    successful_detections = 0
    failed_detections = 0
    no_detections = 0
    fps = []

    # Inicialización del loop
    loop_start_time = time.time()
    for i in range(max_frames):
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = Device.next_frame()

        # Contador de detecciones
        detections_number = len(yolo_detections)
        if detections_number > 0:
            failed_detections += sum([1 for i in range(detections_number) if yolo_detections[i].label != 5])
            successful_detections += 1 if detections_number != failed_detections else 0
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
        print(f"FPS: {fps[-1]:.2f} - Detecciones: {successful_detections} - Falsos positivos: {failed_detections} - No detecciones: {no_detections}", end="\r")

    # Cerrar conexión con la cámara y mostrar resultados
    Device.exit()
    average_fps = sum(fps) / len(fps)

    return average_fps, successful_detections, failed_detections, no_detections

# RUTAS DE LOS MODELOS
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
SingsYOLO_MODELS = [ SingsYOLOv7t_MODEL ]
SingsYOLO_CONFIGS = [SingsYOLOv7t_CONFIG ]

# Tabla de resultados
table = PrettyTable()
table.field_names = ["Yolo Model", "YoloDepth", "YoloDepth + HandTrackerVPU", "YoloDepth + HandTrackerCPU", "Susccessful detections", "Failed detections", "No detections"]

visualize = True
max_frames = 1000
##################### Ejecución de los modelos #####################
for i, model_path in enumerate(SingsYOLO_MODELS):

    config_path = SingsYOLO_CONFIGS[i]
    model_name = os.path.basename(model_path).split(".")[0].split("_")[0]
    
    # TEST 1: YOLO with spatial detection in the VPU.  
    DepthYolo = DepthYoloHandTracker(use_depth = True, use_hand = False, yolo_model = model_path, yolo_configurations = config_path)
    data_DepthYolo = TEST(DepthYolo)

    # TEST 2: YOLO with spatial detection and tracking of the finger of the user on the VPU.
    DepthYolo_HandTrackerVPU = DepthYoloHandTracker(use_depth = True, use_hand = True, use_mediapipe=False, yolo_model = model_path, yolo_configurations = config_path)
    data_DepthYolo_HandTrackerVPU = TEST(DepthYolo_HandTrackerVPU)

    # TEST 3: YOLO with spatial detection in the VPU, and tracking of the finger of the user in the CPU.
    DepthYolo_HandTrackerCPU = DepthYoloHandTracker(use_depth = True, use_hand = True, use_mediapipe=True, yolo_model = model_path, yolo_configurations = config_path)
    data_DepthYolo_HandTrackerCPU = TEST(DepthYolo_HandTrackerCPU)

    # Agregar fila de datos a la tabla
    table.add_row([
        model_name, "{:.2f} fps".format(data_DepthYolo[0]),
        "{:.2f} fps".format(data_DepthYolo_HandTrackerVPU[0]),
        "{:.2f} fps".format(data_DepthYolo_HandTrackerCPU[0]),
        "{:.2f} %".format( (data_DepthYolo[1]+data_DepthYolo_HandTrackerVPU[1]+data_DepthYolo_HandTrackerCPU[1])/(3*max_frames)*100 ),
        "{:.2f} %".format( (data_DepthYolo[2]+data_DepthYolo_HandTrackerVPU[2]+data_DepthYolo_HandTrackerCPU[2])/(3*max_frames)*100 ),
        "{:.2f} %".format( (data_DepthYolo[3]+data_DepthYolo_HandTrackerVPU[3]+data_DepthYolo_HandTrackerCPU[3])/(3*max_frames)*100 )
    ])

    print(table)


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

# +----------------------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |         Yolo Model         | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +----------------------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv7t_openvino_2021 | 10.68 fps |          5.00 fps          |          6.11 fps          |        69.33 %         |       0.00 %      |    30.67 %    |
# | SingsYOLOv5n_openvino_2021 |  6.82 fps |          5.37 fps          |          5.86 fps          |        67.33 %         |      11.33 %      |    21.33 %    |
# +----------------------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv8n | 10.67 fps |          4.30 fps          |          5.36 fps          |         0.00 %         |       0.00 %      |    100.00 %   |
# | SingsYOLOv7s |  3.17 fps |          2.22 fps          |          1.95 fps          |        143.33 %        |       0.00 %      |    23.33 %    |
# | SingsYOLOv7t |  2.28 fps |          2.41 fps          |          2.68 fps          |        70.67 %         |       0.00 %      |    29.33 %    |
# | SingsYOLOv5n |  2.89 fps |          2.97 fps          |          3.15 fps          |        67.33 %         |      17.33 %      |    15.33 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv7s |  1.17 fps |          0.87 fps          |          0.96 fps          |        77.33 %         |       0.00 %      |    22.67 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv8n | 12.10 fps |          5.45 fps          |         12.04 fps          |         0.00 %         |       0.00 %      |    100.00 %   |
# | SingsYOLOv7s |  1.17 fps |          0.67 fps          |          1.17 fps          |        83.67 %         |       0.00 %      |    16.33 %    |
# | SingsYOLOv7t | 11.84 fps |          4.18 fps          |         11.66 fps          |        84.00 %         |       0.00 %      |    16.00 %    |
# | SingsYOLOv5n | 14.36 fps |          4.13 fps          |         13.12 fps          |        31.00 %         |       0.67 %      |    16.33 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+



# RASPBERRY PI 4B 4GB

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv8n | 13.90 fps |          4.82 fps          |          7.77 fps          |        98.17 %         |       0.00 %      |     1.83 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv8n | 13.84 fps |          4.59 fps          |          6.90 fps          |        97.87 %         |       0.00 %      |     2.13 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+


# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv8n | 13.84 fps |          4.59 fps          |          6.90 fps          |        97.87 %         |       0.00 %      |     2.13 %    |
# | SingsYOLOv7s |  1.19 fps |          0.72 fps          |          1.19 fps          |        98.10 %         |       0.00 %      |     1.90 %    |
# | SingsYOLOv7t | 13.89 fps |          5.50 fps          |          7.10 fps          |        18.67 %         |       0.00 %      |    81.33 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+


# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv8n | 13.84 fps |          4.59 fps          |          6.90 fps          |        97.87 %         |       0.00 %      |     2.13 %    |
# | SingsYOLOv7s |  1.19 fps |          0.72 fps          |          1.19 fps          |        98.10 %         |       0.00 %      |     1.90 %    |
# | SingsYOLOv7t | 13.89 fps |          5.50 fps          |          7.10 fps          |        18.67 %         |       0.00 %      |    81.33 %    |
# | SingsYOLOv5n | 17.32 fps |          5.19 fps          |          7.02 fps          |        33.00 %         |       0.07 %      |     1.30 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv7t | 13.90 fps |          3.85 fps          |          7.06 fps          |        35.07 %         |       0.00 %      |    64.93 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+

# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# |  Yolo Model  | YoloDepth | YoloDepth + HandTrackerVPU | YoloDepth + HandTrackerCPU | Susccessful detections | Failed detections | No detections |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
# | SingsYOLOv7t | 13.89 fps |          4.48 fps          |          7.07 fps          |        74.20 %         |       0.00 %      |    25.80 %    |
# +--------------+-----------+----------------------------+----------------------------+------------------------+-------------------+---------------+
