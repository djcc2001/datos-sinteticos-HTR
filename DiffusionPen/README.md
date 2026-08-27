# DiffusionPen: Generación Difusionada de Manuscritos en Español

**Paper**: Diffusionpen: towards controlling the style of handwritten text generation  
**GitHub**: https://github.com/koninik/DiffusionPen

## 📋 Introducción
Modelo diffusion-based de pocos ejemplos para generar texto manuscrito estilizado. Con un número pequeño de muestras de referencia (5 por defecto), aprende el estilo único de un escritor y genera nuevo texto que imita ese estilo. 

**Aclaración importante**: El modelo acepta 5 referencias de estilo como mínimo, pero en nuestros experimentos generamos **15 muestras de salida** por autor usando esas mismas 5 referencias de estilo como condición. Esto permite evaluar la consistencia estilística al generar múltiples variaciones.

## 🚀 Flujo de Trabajo Venv (Paso a Paso)
1. **Crear entorno virtual**: `python -m venv .venv`
2. **Activar entorno**: `source .venv/bin/activate`
3. **Instalar dependencias**: `pip install -r requirements.txt`
4. **Generar muestras**: `python3 scripts/generate_from_txt.py ...`

## ▶️ Uso
```bash
python3 scripts/generate_from_txt.py \
  --checkpoint ./runs/global_bilingual/models/best_ema_unet.pt \
  --stable_dif_path ./models/base/stable-diffusion-v1-5 \
  --style_path ./style_models/mixed_spanish_mobilenetv2_100.pth \
  --dataset_root ./dataset \
  --writer_map ./writers_dict_train.json \
  --writer 1539 \
  --style_manifest ./runs/style_1539_k15/writer_1539_refs_k15.txt \
  --num_style_samples 15 \
  --input ./generar.txt \
  --output ./output \
  --sampling_steps 50 \
  --cfg_scale 2.5 \
  --max_text_len 64 \
  --seed 42
```

**Parámetros clave explicados:**
- `--num_style_samples 15`: Número de **muestras de salida** a generar (15 palabras/texto)
- Las 5 referencias de estilo (condicionado por `--style_manifest`) son el "seed" de escritura del autor
- Se generan 15 variaciones del mismo texto condicionando al mismo estilo

## 📥 Pesos del Modelo
- `diffusionpen.pt`: Modelo global - [Google Drive](https://drive.google.com/file/d/1B4sB8gRTTwKKcFKJEQ6aaeM6vdgENyvf/view?usp=sharing)
- `diffusionpen_finetuning.pt`: Fine-tuning autor específico - [Google Drive](https://drive.google.com/file/d/1xYtds33dL1TwKiGhzFbW2bLrZ7xUCJlt/view?usp=sharing)

## 📊 Resultados
- Charset completo: a-z, A-Z, áéíóúÑüÜ, dígitos, signos `¿¡`
- **15 muestras generadas por autor** usando 5 referencias de estilo
- Mejor para: control de estilo, pocos ejemplos

## 📄 Cita
```bibtex
@inproceedings{nikolaidou2024diffusionpen,
  title={Diffusionpen: towards controlling the style of handwritten text generation},
  author={Nikolaidou, Konstantina and Retsinas, George and Sfikas, Giorgos and Liwicki, Marcus},
  booktitle={European Conference on Computer Vision},
  pages={417--434},
  year={2024},
  organization={Springer}
}
```