import requests
import json
import base64

# Define la URL de la API
url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

# Pide al usuario que ingrese el prompt desde la terminal
user_prompt = input("Introduce el prompt: ")

# Crea el payload con el prompt y otros parámetros
payload = {
    "prompt": f"<lora:EQP_project:1>, {user_prompt}, High Quality",
    "negative_prompt": "<lora:easynegative:1>, worst quality, low quality, zombie, logo, text, watermark, username, monochrome, illustration, cartoon, anime",
    "steps": 20,
    "enable_hr": True,
    "denoising_strength": 0.7,
    "hr_upscaler": "Latent",
    "hr_resize_x": 2048,
    "hr_resize_y": 1024,
    "hr_sampler_name": "Euler",
    "hr_second_pass_steps": 10,
    "width": 1024,
    "height": 512,
    "alwayson_scripts": {
        "Asymmetric tiling": {
            "args": [True, True, False, 0, -1]
        }
    }
}

# Envía la solicitud POST
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Verifica si la solicitud fue exitosa
if response.status_code == 200:
    result = response.json()
    
    # Verifica que la respuesta contenga la clave esperada
    if "images" in result:
        # Extrae la imagen en base64 y decodifícala
        image_data = result["images"][0]
        image_bytes = base64.b64decode(image_data)

        # Guarda la imagen en un archivo
        with open("output.png", "wb") as f:
            f.write(image_bytes)
        print("Imagen guardada como output.png")
    else:
        print("No se encontró la clave 'images' en la respuesta.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

