# FW-GAN

GAN para síntesis de escritura manuscrita con módulos Wave-MLP modulados por ondas, e incorporando un discriminador guiado por frecuencia para mejorar la fidelidad estilística.

## 📎 Referencia original

- **Paper**: *FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator*
- **Repositorio original**: [DAIR-Group/FW-GAN](https://github.com/DAIR-Group/FW-GAN)

Este repo adapta el modelo original para español, con foco en el carácter ñ y adaptación vía fine-tuning. Para detalles de arquitectura y entrenamiento, consulta el repositorio original.

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
python3 generate.py \
  --config configs/fw_gan_global_es.yml \
  --ckpt runs/global_80/ckpts/best.pth \
  --input_txt generar.txt \
  --output output/global \
  --writer_id 1539 \
  --style_split train
```

## 📥 Pesos del modelo

| Archivo | Descripción | Enlace |
|---|---|---|
| `fw_gan.pt` | Modelo global | [Google Drive](https://drive.google.com/file/d/1_eMjG5VRTlVxRAhfZs70qc7pjk8F8Yhh/view?usp=sharing) |
| `fw_gan_finetuning.pt` | Fine-tuning con autor específico | [Google Drive](https://drive.google.com/file/d/1yEqir161AOof1Te3zTVHcrqgRp1Sn2K2/view?usp=sharing) |

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