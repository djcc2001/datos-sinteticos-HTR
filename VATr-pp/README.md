# VATr-pp

Modelo transformer para generación de texto manuscrito estilizado, que usa atención visual para modelar la relación entre contenido y estilo. Adaptación few-shot con muestras de referencia de autor (15 imágenes por autor).

## 📎 Referencia original

- **Paper**: *VATr++: Choose Your Words Wisely for Handwritten Text Generation*
- **Repositorio original**: [EDM-Research/VATr-pp](https://github.com/EDM-Research/VATr-pp)

Este repo adapta el modelo original para español, con foco en caracteres de baja frecuencia (ñ, acentos). Para detalles de arquitectura y entrenamiento, consulta el repositorio original.

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
python3 generate.py text \
  --checkpoint files/vatrpp.pth \
  --style-folder files/style_samples/00 \
  --num_examples 15 \
  --text-path generar.txt \
  --output output/
```

**Nota**: VATr-pp no requiere fine-tuning — usa 15 imágenes reales por autor directamente en inferencia como referencia de estilo.

## 📥 Pesos del modelo

| Archivo | Descripción | Enlace |
|---|---|---|
| `vatr-pp.pt` | Modelo few-shot | [Google Drive](https://drive.google.com/file/d/1vTJhW2SHE9uiq-Fn3VziyN1H_DpoJidx/view?usp=sharing) |

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