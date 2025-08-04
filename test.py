# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO

# Sugerencia para el nombre del nuevo archivo: yolo_eval_and_predict.py

# --- Configuración unificada ---
# 1. Ruta del modelo
model_path = '/root/autodl-tmp/defectos_de_soldaduras/defectos_de_soldaduras/experimento1/weights/best.pt'  # !!! Importante: modifica esto con la ruta real de tu archivo best.pt !!!

# 2. Archivo de configuración del dataset (principalmente para evaluación del modelo)
data_yaml_path = '/root/autodl-tmp/defectos_de_soldaduras/dataset.yaml' # !!! Importante: modifica con tu propia ruta del archivo data.yaml !!!
# !!! Asegúrate que el archivo yaml tenga definido test: ./images/test o similar !!!


# 3. Ruta de la carpeta con imágenes para predicción individual
#    Asumimos que quieres predecir imágenes del conjunto de prueba. Si es otra carpeta, modifica esta ruta.
predict_input_images_folder = '/root/autodl-tmp/defectos_de_soldaduras/valid/images'  # !!! Importante: modifica con la ruta de tu carpeta de imágenes a predecir !!!

# 4. Parámetros generales
image_size = 640  # Tamaño de imagen de entrada (recomendado usar el mismo imgsz que en entrenamiento)
gpu_device = 0  # Dispositivo GPU a usar (0 para la primera GPU). Usar 'cpu' para CPU

# --- Módulo 1: Configuración de evaluación del modelo ---
eval_project_name = 'experimento14'  # Nombre de la carpeta para resultados de evaluación
eval_run_name = 'experimento14_eval'  # Nombre de esta ejecución de evaluación (YOLO manejará automáticamente duplicados como eval_run_name2)
eval_batch_size = 16  # Tamaño del batch para evaluación

# --- Módulo 2: Configuración de predicción individual ---
predict_project_name = 'my_yolo_predictions'  # --- Módulo 2: Configuración de predicción individual ---
predict_run_base_name = 'individual_image_detections'  # Nombre base para esta ejecución (YOLO manejará duplicados)
predict_confidence_threshold = 0.25  # Umbral de confianza para mostrar resultados (0.0 a 1.0)

if __name__ == '__main__':
    print("==========================================================")
    print("     Script combinado de evaluación y predicción YOLOv8     ")
    print("==========================================================")

    # --- Verificación de rutas ---
    print(f"\n--- Verificando rutas de entrada ---")
    print(f"Ruta del modelo: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo del modelo '{model_path}'，Verifica la ruta. El script terminará.")
        exit()

    # --- Carga del modelo (solo una vez, para evaluación y predicción) ---
    print(f"\n--- Paso 0: Cargando modelo pre-entrenado ---")
    try:
        model = YOLO(model_path)
        print(f"Modelo '{model_path}' cargado exitosamente.")
    except Exception as e:
        print(f"Error: Falló la carga del modelo: {e}。script terminará.")
        exit()

    # --- Módulo 1: Evaluación del modelo (tu funcionalidad original) ---
    print(f"\n\n--- Módulo 1: Iniciando evaluación de rendimiento en conjunto de prueba ---")
    print(f"Archivo de configuración del dataset: {data_yaml_path}")
    if not os.path.exists(data_yaml_path):
        print(f"Advertencia: No se encontró el archivo de configuración '{data_yaml_path}'. Se omitirá el módulo de evaluación.")
    else:
        print(f"Parámetros de evaluación: imgsz={image_size}, batch_size={eval_batch_size}, device='{gpu_device}'")
        # YOLO的 val 方法在 project/name 冲突时会自动创建 name2, name3... (如果 exist_ok=False)
        print(f"Resultados se guardarán en'{eval_project_name}' bajo la carpeta '{eval_run_name}' (o con sufijo numérico).")

        try:
            metrics = model.val(
                data=data_yaml_path,
                split='test',  # Especifica usar el conjunto de prueba
                imgsz=image_size,
                batch=eval_batch_size,
                device=gpu_device,
                project=eval_project_name,
                name=eval_run_name,
                save_json=True,  # Guarda resultados en formato JSON para análisis detallado
                exist_ok=False  # Crea nueva carpeta con sufijo si ya existe
            )
            print("--- Evaluación del modelo completada ---")

            # Mostrar métricas clave
            if metrics and hasattr(metrics, 'box'):  # 确保 metrics 和 metrics.box 存在
                print("\n--- Métricas principales de evaluación ---")
                # mAP (mean Average Precision) metric
                print(f"mAP50-95 (métrica general): {metrics.box.map:.4f}")
                print(f"mAP50 (IoU@0.50): {metrics.box.map50:.4f}")
                print(f"mAP75 (IoU@0.75): {metrics.box.map75:.4f}")

                # Precision, Recall, F1-score
                precision = metrics.box.mp  # Precisión promedio
                recall = metrics.box.mr  # Recall promedio

                print(f"Precisión promedio (Precision): {precision:.4f}")
                print(f"Recall promedio (Recall): {recall:.4f}")

                # Cálculo de F1-score
                if (precision + recall) == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                print(f"F1 Score: {f1_score:.4f}")

                # metrics.save_dir 包含实际保存结果的目录路径
                actual_eval_save_dir = metrics.save_dir
                print(f"\nResultados detallados (gráficos, matriz de confusión, JSON) guardados en:")
                print(f"{os.path.abspath(actual_eval_save_dir)}")
            else:
                print("La evaluación no devolvió métricas válidas. Verifica la salida del proceso.")

        except Exception as e:
            print(f"Error durante la evaluación del modelo:  {e}")

    # --- Módulo 2: Detección individual y guardado de resultados (nueva funcionalidad) ---
    print(f"\n\n--- Módulo 2: Iniciando detección y guardado individual de imágenes ---")
    print(f"Carpeta con imágenes a predecir:{predict_input_images_folder}")
    if not os.path.exists(predict_input_images_folder) or not os.path.isdir(predict_input_images_folder):
        print(f"Error: No se encontró la carpeta '{predict_input_images_folder}'  o no es válida.")
        print("Se omitirá el módulo de predicción individual.")
    else:
        print(
            f"Parámetros de predicción: imgsz={image_size}, device='{gpu_device}', confidence_threshold={predict_confidence_threshold}")
        # YOLO的 predict 方法同样会自动处理重名文件夹 (如果 exist_ok=False)
        print(
            f"Imágenes resultantes se guardarán en '{predict_project_name}' bajo la carpeta '{predict_run_base_name}' (o con sufijo).")

        try:
            # Llamar al método predict para realizar inferencia y guardar resultados automáticamente
            prediction_results = model.predict(
                source=predict_input_images_folder,  # Puede ser una ruta de carpeta
                imgsz=image_size,
                device=gpu_device,
                conf=predict_confidence_threshold,  # Umbral de confianza
                save=True,   # Clave: Establecer en True para guardar imágenes anotadas automáticamente
                project=predict_project_name,  # Carpeta principal del proyecto para guardar resultados
                name=predict_run_base_name,  # Nombre de esta ejecución de predicción (YOLO manejará conflictos automáticamente)
                exist_ok=False,  # Si el directorio existe, crear uno nuevo con sufijo
                save_txt=False,  # Opcional: Guardar archivos de anotación en formato txt (YOLO)
                save_conf=False,  # Opcional: Si save_txt=True, incluir confianza en el archivo txt
                show_labels=True, # Mostrar etiquetas de clase en imágenes resultantes (por defecto True)
                show_conf=True,  # Mostrar puntuaciones de confianza en imágenes resultantes (por defecto True)

            )
            print("--- Predicción individual completada ---")

            # prediction_results es una lista donde cada elemento corresponde a los resultados de una fuente de entrada
            # Para entradas de carpeta, todas las salidas de imágenes están en el mismo save_dir
            if prediction_results and len(prediction_results) > 0:
                # Normalmente, todos los objetos Results en la lista tienen el atributo .save_dir
                # apuntando al mismo directorio de esta ejecución
                actual_predict_save_dir = prediction_results[0].save_dir
                print(f"Todas las imágenes con predicciones guardadas en:")
                print(f"{os.path.abspath(actual_predict_save_dir)}")
            elif os.path.exists(predict_input_images_folder) and os.listdir(
                    predict_input_images_folder):
                print(f"Predicción ejecutada，Pero desde 'prediction_results' pero no se obtuvo ruta de guardado.")
                print(
                    f"Revisa la subcarpeta '{predict_project_name}' (o con sufijo) en '{predict_run_base_name}'.")
            else:
                print(f"La carpeta '{predict_input_images_folder}' está vacía o no contiene imágenes válidas.")


        except Exception as e:
            print(f"Error durante la predicción individual: {e}")

    print("\n\n==========================================================")
    print("                 Todas las tareas completadas                   ")
    print("==========================================================")