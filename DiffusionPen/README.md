# DiffusionPen

Modelo de difusión de pocos ejemplos ("few-shot") para generar texto manuscrito estilizado. A partir de un pequeño número de muestras de referencia (5 por defecto), aprende el estilo de un escritor y genera nuevo texto que lo imita.

> **Nota de este experimento**: el modelo acepta 5 referencias de estilo como mínimo, pero en nuestros experimentos generamos **15 muestras de salida** por autor condicionadas a esas mismas 5 referencias, para evaluar la consistencia estilística entre variaciones.

## 📎 Referencia original

- **Paper**: *DiffusionPen: Towards Controlling the Style of Handwritten Text Generation*
- **Repositorio original**: [koninik/DiffusionPen](https://github.com/koninik/DiffusionPen)

Este repo adapta el modelo original para español, con foco en caracteres de baja frecuencia (ñ). Para detalles de arquitectura y entrenamiento, consulta el repositorio original.

## ⚙️ Instalación

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

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

**Parámetros clave:**
- `--num_style_samples 15`: número de muestras de salida a generar.
- `--style_manifest`: define las 5 referencias de estilo del autor (el "seed" de escritura).
- Se generan 15 variaciones del mismo texto condicionadas al mismo estilo.

## 📥 Pesos del modelo

| Archivo | Descripción | Enlace |
|---|---|---|
| `diffusionpen.pt` | Modelo global | [Google Drive](https://drive.google.com/file/d/1B4sB8gRTTwKKcFKJEQ6aaeM6vdgENyvf/view?usp=sharing) |
| `diffusionpen_finetuning.pt` | Fine-tuning con autor específico | [Google Drive](https://drive.google.com/file/d/1xYtds33dL1TwKiGhzFbW2bLrZ7xUCJlt/view?usp=sharing) |

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