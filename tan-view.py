import os
import numpy as np
from PIL import Image

def get_tangential_views(panorama_path, output_folder, num_views=12, view_size=(256, 256)):
    """
    Divide una imagen panorámica en varias vistas tangenciales pequeñas y las guarda en una carpeta.
    
    Args:
        panorama_path (str): Ruta de la imagen panorámica.
        output_folder (str): Ruta de la carpeta donde se guardarán las vistas.
        num_views (int): Número de vistas a generar (división equitativa en esferas).
        view_size (tuple): Tamaño de cada vista (ancho, alto).
    """
    panorama = Image.open(panorama_path)
    panorama_width, panorama_height = panorama.size

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Coordenadas de la esfera
    view_count = 0
    for i in range(num_views):
        theta = np.pi * (i / num_views)  # Latitud en radianes
        for j in range(num_views):
            phi = 2 * np.pi * (j / num_views)  # Longitud en radianes
            
            # Convertir coordenadas esféricas a coordenadas en la imagen
            x = int((phi / (2 * np.pi)) * panorama_width)
            y = int((theta / np.pi) * panorama_height)
            
            # Extraer la subimagen
            left = max(0, x - view_size[0] // 2)
            upper = max(0, y - view_size[1] // 2)
            right = min(panorama_width, x + view_size[0] // 2)
            lower = min(panorama_height, y + view_size[1] // 2)
            
            view = panorama.crop((left, upper, right, lower))
            
            # Guardar la subimagen en la carpeta de salida
            view_filename = os.path.join(output_folder, f"vista_{view_count + 1}.jpg")
            view.save(view_filename)
            view_count += 1
            print(f"Guardada {view_filename}")

# Ruta de la imagen panorámica y carpeta de salida
panorama_path = 'panorama.png'  # Cambia esto a la ruta de tu imagen
output_folder = 'Vistas-Tangenciales/'    # Cambia esto a la ruta de tu carpeta de salida

# Generar y guardar las vistas tangenciales
get_tangential_views(panorama_path, output_folder, num_views=6, view_size=(256, 256))

