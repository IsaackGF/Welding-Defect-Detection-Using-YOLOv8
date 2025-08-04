# -*- coding: utf-8 -*-
# Importar la clase YOLO del paquete Ultralytics
from ultralytics import YOLO

# --- Configuración de parámetros de entrenamiento ---
# 1. Ruta del modelo pre-entrenado
#    - Puede ser un modelo oficial como 'yolov8n.pt', 'yolov8s.pt', etc.
#      (si no existe localmente, se descargará automáticamente)
#    - O puede ser un modelo guardado de un entrenamiento previo ('last.pt') para continuar entrenando
model_path = 'yolov8n.pt'  # Usamos yolov8n como modelo base

# 2. Ruta del archivo de configuración del dataset
#    - Este es tu archivo data.yaml que define la estructura del dataset
data_yaml_path = '/root/autodl-tmp/defectos_de_soldaduras/dataset.yaml'  # ¡Ajusta esta ruta!

# 3. Parámetros de entrenamiento
num_epochs = 100      # Número total de iteraciones (épocas)
image_size = 640      # Tamaño de entrada de las imágenes (múltiplo de 32)
batch_size = 16     # Número de imágenes por lote (ajustar según tu GPU)
gpu_device = 0
workers= 0
optimizer='Adam' # Número de GPU a usar (0 para la primera GPU, 'cpu' para CPU)

# 4. Configuración para guardar resultados
project_name = 'defectos_de_soldaduras' # Nombre del proyecto
run_name = 'experimento1'    # Nombre de esta ejecución específica

# --- Inicio del entrenamiento ---

if __name__ == '__main__':
    print("--- Cargando modelo y configuración ---")
    # Cargar el modelo

    model = YOLO(model_path)
    print(f"Modelo cargado: {model_path}")
    print(f"Archivo de ddataset: {data_yaml_path}")
    print(f"Parámetros: épocas: epochs={num_epochs}, imgsz={image_size}, batch={batch_size}, device={gpu_device}")
    print(f"Resultados se guardarán en: {project_name}/{run_name}")
    print("--- Iniciando entrenamiento ---")

    # Iniciar el entrenamiento

    # Explicación de parámetros:
    #   data: ruta del archivo de configuración del dataset
    #   epochs: número de épocas de entrenamiento
    #   imgsz: tamaño de imagen de entrada
    #   batch: tamaño del batch (lote)
    #   device: dispositivo de entrenamiento (GPU o CPU)
    #   project: nombre de la carpeta del proyecto donde se guardarán los resultados
    #   name: nombre específico para esta ejecución de entrenamiento
    #   exist_ok: si es True, no dará error si la carpeta project/name ya existe, sobrescribirá o continuará escribiendo (por defecto False)
    #   patience: número de épocas para early stopping, si no hay mejora en las métricas de validación durante 'patience' épocas, se detiene el entrenamiento (por defecto 50 o 100, depende de la versión)
    #   optimizer: selección del optimizador ('SGD', 'Adam', 'AdamW', etc., por defecto 'auto')
    #   lr0: tasa de aprendizaje inicial (por defecto 0.01)
    #   ... hay muchos otros parámetros que se pueden ajustar, consulta la documentación oficial para más detalles

    results = model.train(
        data=data_yaml_path,
        epochs=num_epochs,
        imgsz=image_size,
        batch=batch_size,
        device=gpu_device,
        project=project_name,
        name=run_name,
        exist_ok=False, # Si la carpeta ya existe, no se sobrescribirá,  # sino que creará nuevas carpetas como run_name2, run_name3...
        patience=100,   # Número de épocas para early stopping (puede descomentarse)
        # optimizer='AdamW', # Opcional: probar diferentes optimizadores
        # lr0=0.001, # Opcional: ajustar la tasa de aprendizaje inicial
    )

    print("--- Entrenamiento completado ---")
    print(f"Resultados guardados en: {project_name}/{run_name}")
    # El objeto 'results' contiene información del proceso de entrenamiento,
    # pero normalmente nos enfocamos más en los archivos guardados
    # print(results)  # Puedes descomentar para inspeccionar el contenido completo