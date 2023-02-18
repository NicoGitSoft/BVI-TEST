"""
La idea es que este script se ejecute en la raspberry pi y que muestre el FPS de la camara OAK-D
usando los diferentes modelos de yolo, y así mismo trabajando en conjunto con la visión estéreo y la detección de manos.
"""

from Utilities2 import *
import cv2, time, csv


def TestFPS(number_of_frames, opt1, opt2, YOLO_MODEL, YOLO_CONFIG):
    OAK_D = DepthYoloHandTracker(use_hand=opt1,
                                use_depth=opt2,
                                yolo_model=YOLO_MODEL,
                                yolo_configurations=YOLO_CONFIG)

    confidences = []
    frames_with_detections = 0 # Number of photograms on which detections were found
    
    fps = []
    for i in range(number_of_frames):
        star_time = time.time() # Tiempo inicial
        frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = OAK_D.next_frame()
        if len(yolo_detections) > 0:
            frames_with_detections += 1
            for detection in yolo_detections:
                confidences.append(detection.confidence*100)
        end_time = time.time() # Tiempo final
        fps.append(1/(end_time-star_time))
        print(f"FPS: {fps[-1]:.2f} | Frames with detections: {frames_with_detections}", hands, end='\r')
        

    OAK_D.exit()                            # Cerrar la camara OAK-D
    
    AverageConfidence = sum(confidences)/len(confidences) if confidences else 0
    AverageFPS = sum(fps)/len(fps)
    
    return AverageFPS, AverageConfidence, frames_with_detections


# Rutas de los modelos neuronales de la mano
SCRIPT_DIR = Path(__file__).resolve().parent

# Rutas de los modelos YOLOv8n, YOLOv7t, YOLOv7s, YOLOv5n 
SingsYOLOv8n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n_openvino_2021.4_6shave.blob")
SingsYOLOv7s_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s_openvino_2021.4_6shave.blob")
SingsYOLOv7t_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t_openvino_2021.4_6shave.blob")
SingsYOLOv5n_MODEL = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n_openvino_2021.4_6shave.blob")
SingsYOLOv8n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv8n/SingsYOLOv8n.json")
SingsYOLOv7t_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7t/SingsYOLOv7t.json")
SingsYOLOv7s_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv7s/SingsYOLOv7s.json")
SingsYOLOv5n_CONFIG = str(SCRIPT_DIR / "../Models/Sings/SingsYOLOv5n/SingsYOLOv5n.json")

# Listas de los modelos y sus respectivas configuraciones
SingsYOLO_MODELS = [SingsYOLOv8n_MODEL, SingsYOLOv7t_MODEL, SingsYOLOv7s_MODEL, SingsYOLOv5n_MODEL] 
SingsYOLO_CONFIGS = [SingsYOLOv8n_CONFIG, SingsYOLOv7t_CONFIG, SingsYOLOv7s_CONFIG, SingsYOLOv5n_CONFIG]

# Pruebas de los modelos
for i in range(4):
    model_name = SingsYOLO_CONFIGS[i].rstrip('.json').split('/')[-1]
    data_only_yolo = TestFPS(100, False, False, SingsYOLO_MODELS[i], SingsYOLO_CONFIGS[i])
    print(data_only_yolo)
