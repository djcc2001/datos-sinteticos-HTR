# VATr-pp: Handwritten Text Generation from Visual Archetypes ++

**Paper**: Vatr++: Choose your words wisely for handwritten text generation  
**GitHub**: https://github.com/EDM-Research/VATr-pp

## 📋 Introducción
Transformer-based modelo para generación de texto manuscrito estilizado. Usa atención visual para modelar la relación contenido-estilo. Adaptación few-shot con muestras de referencia de autor (15 imágenes por autor). Evaluado en español con enfoque en caracteres de baja frecuencia como el ñ.

## 🚀 Flujo de Trabajo Venv (Paso a Paso)
1. **Crear entorno virtual**: `python -m venv .venv`
2. **Activar entorno**: `source .venv/bin/activate`
3. **Instalar dependencias**: `pip install -r requirements.txt`
4. **Generar muestras**: `python3 generate.py text --checkpoint files/vatrpp.pth --style-folder files/style_samples/00 ...`

## ▶️ Uso
```bash
python3 generate.py text \
  --checkpoint files/vatrpp.pth \
  --style-folder files/style_samples/00 \
  --num_examples 15 \
  --text-path generar.txt \
  --output output/
```

## 📥 Pesos del Modelo
- `vatr-pp.pt`: [Google Drive](https://drive.google.com/file/d/1vTJhW2SHE9uiq-Fn3VziyN1H_DpoJidx/view?usp=sharing)

## 📊 Resultados y Métricas en Español
- **CER (global)**: 0.197 - Tasa de error de caracteres a nivel de palabra
- **CER$_{\tilde{n}}$**: 0.448 - Error específico en el carácter ñ
- **Confusión ñ→n**: 57% - Proporción de errores al modelar la tilde
- **15 referencias few-shot**: No requiere fine-tuning, usa 15 imágenes reales por autor al inferencia
- **Characters**: ñ, á, é, í, ó, ú, Ü, Ñ

**Resultados completos de la tabla comparativa:**
| Modelo | CER$_{g}$ ↓ | CER$_{\tilde{n}}$ ↓ | ñ→n ↓ |
|--------|-------------|---------------------|-------|
| FW-GAN (global) | 0.513 | 0.609 | 91.0% |
| FW-GAN (adapt.) | 0.393 | 0.679 | 45.0% |
| DiffusionPen (global) | 0.696 | 0.901 | 63.0% |
| DiffusionPen (adapt.) | 0.703 | 0.837 | 62.0% |
| VATr-pp$^{\dagger}$ | 0.197 | 0.448 | 57.0% |

$^{\dagger}$Usa 15 referencias reales en inferencia; ver Sec. de modelos.

## 📄 Cita
```bibtex
@article{vanherle2024vatr++,
  title={Vatr++: Choose your words wisely for handwritten text generation},
  author={Vanherle, Bram and Pippi, Vittorio and Cascianelli, Silvia and Michiels, Nick and Van Reeth, Frank and Cucchiara, Rita},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={47},
  number={2},
  pages={934--948},
  year={2024},
  publisher={IEEE}
}
```