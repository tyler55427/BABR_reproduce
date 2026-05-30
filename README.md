# BABR Reproduction — BRAR Severity Classification

Reproduction code for the **BRAR-anchored multimodal dataset of panoramic radiographs
for periodontal bone resorption grading**.

> 📌 **Original Dataset DOI**: [10.6084/m9.figshare.30155974](https://doi.org/10.6084/m9.figshare.30155974)
> 🌐 **Figshare**: [https://figshare.com/s/cc2db8648788c7aa175c](https://figshare.com/s/cc2db8648788c7aa175c)

---

## 📁 Dataset Preparation

### 1. Download from Figshare

Download the BRAR dataset from Figshare. After extraction, you should have:

```
datas/
├── meta_data.csv                 # renamed from brar_grades.csv (or metadata CSV)
├── BRAR_codebook_v1.0.xlsx       # optional, for reference
├── level_1/                      # grade-1 images (copy from images/)
├── level_2/                      # grade-2 images
└── level_3/                      # grade-3 images
```

### 2. Organize into `datas/`

Place the data into the repository's `datas/` directory with this structure:

```
BRAR-dataset/
├── images/                       # Anonymized panoramic radiographs (.jpg)
├── annotations/
│   ├── demographics.csv
│   ├── tooth_level_labels.csv
│   └── brar_grades.csv
├── BRAR_codebook_v1.0.xlsx
└── BRAR_annotation_protocol_v1.0.pdf
```

If your data is organized differently by grade, adjust the `--data_root` and `--meta_csv` arguments accordingly.

### 3. Run Preprocessing

```bash
# Default — uses datas/ structure described above
python datapreprocess.py

# Custom paths
python datapreprocess.py \
  --data_root ./datas \
  --meta_csv ./datas/meta_data.csv \
  --output_labels ./datas/labels.csv \
  --output_images ./datas/images
```

This produces:
- `datas/labels.csv` — columns: `新文件名`, `等级` (1–3)
- `datas/images/` — flat directory with all 988 images

---

## 🧪 Training

```bash
# Default paths (after preprocessing)
python train.py

# Custom paths
python train.py \
  --label_csv ./datas/labels.csv \
  --image_dir ./datas/images \
  --output_dir ./results
```

The model (EfficientNet-B0 with strong regularization) is saved to `results/optimized_bone_classifier.pth`.

---

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `datapreprocess.py` | Data preprocessing pipeline — organizes raw images and generates labels CSV |
| `train.py` | Main training script — 3-class BRAR severity classifier (EfficientNet-B0) |
| `train01.py` | Alternative training configuration (e.g., different backbone) |
| `.gitignore` | Excludes `datas/`, model checkpoints, generated plots |
| `optimized_bone_classifier.pth` | Pretrained checkpoint for inference |
| `*.png` | Training curves and analysis plots |

### `datapreprocess.py` Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `datas` | Root dir with `level_1/`, `level_2/`, `level_3/` subdirectories |
| `--meta_csv` | `{data_root}/meta_data.csv` | Path to metadata CSV |
| `--output_labels` | `{data_root}/labels.csv` | Output labels CSV path |
| `--output_images` | `{data_root}/images` | Output flat image directory |
| `--no-copy` | `False` | Skip image copy, only generate labels CSV |

### `train.py` Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--label_csv` | `./datas/labels.csv` | Labels CSV/Excel (columns: `新文件名`, `等级`) |
| `--image_dir` | `./datas/images` | Flat image directory |
| `--output_dir` | `./results` | Output directory for model and plots |

---

## 🧰 Dependencies

```
Python ≥ 3.8
PyTorch ≥ 2.0
timm
pandas, numpy, scikit-learn
Pillow, matplotlib, seaborn
```

```bash
pip install torch torchvision timm pandas numpy scikit-learn pillow matplotlib seaborn
```

---

## 🔬 Inference Example

```python
import torch
from PIL import Image
from torchvision import transforms

checkpoint = torch.load('results/optimized_bone_classifier.pth', map_location='cpu')
model = timm.create_model(checkpoint['model_name'], pretrained=False, num_classes=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open('panoramic_radiograph.jpg').convert('RGB')
with torch.no_grad():
    pred = model(transform(img).unsqueeze(0)).argmax(dim=1).item()
print(f'Predicted BRAR grade: {pred + 1}')  # 1, 2, or 3
```

---

## 📚 Citation

If you use this dataset or code, please cite:

> Li, Z. et al. BRAR-anchored multimodal dataset of panoramic radiographs for
> periodontal bone resorption grading. *Sci Data* (2025).
> DOI: [10.6084/m9.figshare.30155974](https://doi.org/10.6084/m9.figshare.30155974)

## 📜 License

- **Dataset**: CC-BY 4.0
- **Code**: MIT
