import os
import torch
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import matplotlib.pyplot as plt

# Cargar el modelo y el extractor de características
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
model.eval()

def estimate_depth(image_path):
    """
    Calcula el mapa de profundidad de una imagen y lo normaliza.
    """
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Estimación de profundidad
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()

    # Normalizar el mapa de profundidad para visualización
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)

    return depth_normalized

def process_depth_maps(input_folder, output_folder):
    """
    Procesa todas las imágenes en una carpeta, calcula el mapa de profundidad para cada una,
    y guarda los resultados en otra carpeta.
    """
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterar sobre todas las imágenes en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            
            # Calcular el mapa de profundidad
            depth_map = estimate_depth(image_path)
            
            # Guardar el mapa de profundidad como una imagen
            output_path = os.path.join(output_folder, f"depth_{filename}")
            plt.imsave(output_path, depth_map, cmap='plasma')
            print(f"Mapa de profundidad guardado en {output_path}")

# Carpetas de entrada y salida
input_folder = 'Vistas-Tangenciales'      # Cambia a la ruta de la carpeta de entrada
output_folder = 'Profundidad'      # Cambia a la ruta de la carpeta de salida

# Procesar y guardar mapas de profundidad
process_depth_maps(input_folder, output_folder)
