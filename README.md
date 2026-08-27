# Evaluación Comparativa de Modelos de Síntesis de Escritura en Español

![Comparative Sample](img/fig_muestras.png)

## 📋 Descripción

Este repositorio contiene la evaluación comparativa de tres modelos generativos para síntesis de texto manuscrito en español, con enfoque en caracteres de baja frecuencia (como la ñ) y adaptación de estilo con datos limitados.

Cada modelo vive en su propia carpeta, con su propio README explicando instalación y uso:

- [`DiffusionPen/`](DiffusionPen/) — modelo basado en difusión
- [`FW-GAN/`](FW-GAN/) — GAN guiada por frecuencia
- [`VATr-pp/`](VATr-pp/) — modelo transformer con atención visual

## 📊 Dataset

El proyecto utiliza dos fuentes de datos principales:

- **IAM Corpus**: base de datos pública de oraciones para reconocimiento de escritura manuscrita, con 1,550 autores y 111,495 palabras.
- **Causas Criminales**: archivo regional de Cusco, Perú — manuscritos históricos en español del siglo XVIII-XIX. El único documento transcrito disponible fue usado para evaluación.

La combinación permite evaluar los modelos tanto en vocabulario general como en español con caracteres especiales (ñ, acentos).

## 📥 Modelos entrenados (Google Drive)

| Modelo | Archivo | Descripción |
|---|---|---|
| DiffusionPen | `diffusionpen.pt` | Modelo global |
| DiffusionPen (finetuning) | `diffusionpen_finetuning.pt` | Fine-tuning con autor específico |
| FW-GAN | `fw_gan.pt` | Modelo global |
| FW-GAN (finetuning) | `fw_gan_finetuning.pt` | Fine-tuning con autor específico |
| VATr-pp | `vatr-pp.pt` | Modelo few-shot (sin fine-tuning) |

[Enlaces completos de Google Drive](https://drive.google.com/drive/folders/1b8iwhMG-BqQJ218CAc14d4hM2giEdUWN?usp=drive_link)

## 📊 Resultados comparativos

| Modelo | CER (global) ↓ | CER (ñ) ↓ | Confusión ñ→n ↓ | Base |
|---|---|---|---|---|
| FW-GAN (global) | 0.513 | 0.609 | 91.0% | GAN |
| FW-GAN (adapt.) | 0.393 | 0.679 | 45.0% | GAN + Fine-tuning |
| DiffusionPen (global) | 0.696 | 0.901 | 63.0% | Diffusion |
| DiffusionPen (adapt.) | 0.703 | 0.837 | 62.0% | Diffusion + Fine-tuning |
| VATr-pp | 0.197 | 0.448 | 57.0% | Transformer + Few-shot |

> VATr-pp usa 15 referencias reales en inferencia (sin fine-tuning). Ver el README de cada modelo para el detalle de su configuración de evaluación.

## 🚀 Quick start por proyecto

Cada modelo usa su propio entorno virtual. Pasos generales (el comando de generación varía por modelo — ver el README de cada carpeta):

```bash
# 1. Entrar a la carpeta del modelo
cd DiffusionPen   # o FW-GAN, o VATr-pp

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Generar muestras (ver comando específico en el README del modelo)
```

## ⚙️ Requisitos generales

- **Python**: ≥3.8
- **PyTorch**: ≥2.0.0 (recomendado con CUDA 11.7+ para GPU)
- **VRAM**: mínimo 8GB para generación, 24GB recomendado para entrenamiento
- **Dependencias específicas**: ver `requirements.txt` de cada proyecto

## 📄 Citas

Referencias a los papers originales de cada modelo:

- **DiffusionPen**: Nikolaidou et al. 2024 (ECCV)
- **FW-GAN**: Tong et al. 2025 (Expert Systems with Applications)
- **VATr-pp**: Vanherle et al. 2024 (IEEE TPAMI)

Ver el README de cada carpeta para el BibTeX completo.