from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "runwayml/stable-diffusion-v1-5"
save_path = "./stable-diffusion-v1-5"

if not os.path.exists(save_path):
    print(f"Descargando Stable Diffusion v1.5 en {save_path}...")
    # Descargamos el modelo completo
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Lo guardamos en la carpeta local para que DiffusionPen lo encuentre
    pipe.save_pretrained(save_path)
    print("¡Descarga completada!")
else:
    print("El modelo ya existe en la carpeta.")