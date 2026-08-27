# FW-GAN: Síntesis de Escritura Manuscript Frecuencia-Driven

**Paper**: FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator  
**GitHub**: https://github.com/DAIR-Group/FW-GAN

## 📋 Introducción
GAN para síntesis de escritura manuscrito con módulos Wave-MLP modulados por ondas. Incorpora discriminador guided by frequency para fidelidad estilística mejorada. Evaluado en español con foco en el carácter ñ y adaptación via fine-tuning.

## 📊 Resultados y Métricas en Español
- **CER$_g$ (global)**: 0.513 - Tasa de error de caracteres a nivel de palabra
- **CER$_{\tilde{n}}$**: 0.609 - Error específico en el carácter ñ
- **Confusión ñ→n**: 91% - Proporción de errores al modelar la tilde
- **11K+ palabras** evaluadas en español desde `data/spanish_words.txt`

**Parámetros de evaluación:**
- Se generó 1 imagen por muestra del test set (5,000 samples)
- Métricas computadas usando TrOCR sin fine-tuning adicional
- Resultados comparables con DiffusionPen y VATr-pp bajo mismo protocolo

## 📋 Introducción (detalle)
GAN para síntesis de escritura manuscrito con módulos Wave-MLP modulados por ondas. Incorpora discriminador guided by frequency para fidelidad estilística mejorada. Evaluado en español con foco en el carácter ñ y adaptación via fine-tuning.

## 🚀 Flujo de Trabajo Venv (Paso a Paso)
1. **Crear entorno virtual**: `python -m venv .venv`
2. **Activar entorno**: `source .venv/bin/activate`
3. **Instalar dependencias**: `pip install -r requirements.txt`
4. **Generar muestras**: `python3 generate.py --config configs/fw_gan_global_es.yml ...`

## ▶️ Uso
```bash
python3 generate.py \
  --config configs/fw_gan_global_es.yml \
  --ckpt runs/global_80/ckpts/best.pth \
  --input_txt generar.txt \
  --output output/global \
  --writer_id 1539 \
  --style_split train
```

## 📥 Pesos del Modelo
- `fw_gan.pt`: Modelo global - [Google Drive](https://drive.google.com/file/d/1_eMjG5VRTlVxRAhfZs70qc7pjk8F8Yhh/view?usp=sharing)
- `fw_gan_finetuning.pt`: Fine-tuning autor específico - [Google Drive](https://drive.google.com/file/d/1yEqir161AOof1Te3zTVHcrqgRp1Sn2K2/view?usp=sharing)

## 📊 Resultados
- 11K+ palabras en `data/spanish_words.txt`
- n_class: 95 clases de caracteres
- **Charset completo**: a-z, A-Z, **áéíóúÑüÜ**, dígitos, signos **¿¡**
- Mejor para: vocabulario grande, generación one-shot

## 📄 Cita
```bibtex
@article{tong2025fw,
  title={FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator},
  author={Tong, Huynh Tong Dang and Nam, Dang Hoai and Le Duy, Vo Nguyen},
  journal={Expert Systems with Applications},
  pages={130175},
  year={2025},
  publisher={Elsevier}
}
```