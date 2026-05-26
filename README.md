# Brain Anomaly CV — Multi-Task Brain MRI / CT Pipeline

A computer vision research pipeline for brain imaging built around three
complementary capabilities:

1. **Anomaly segmentation** — pixel-level masking of brain anomalies
   (glioma, hemorrhage, and other lesions) with YOLOv11-seg.
2. **Grading & severity** — per-anomaly clinical-grade classification and
   volume / shape severity metrics via a CNN head fed masked ROI crops.
3. **Neuroimaging correlates** — *probabilistic* condition scoring for ADHD
   and ASD (and optionally OCD / schizophrenia) from skull-stripped,
   MNI-registered structural MRI, with Grad-CAM saliency overlays.

> **Research project, not a clinical tool.** Nothing produced by this
> pipeline is suitable for diagnosis or any medical decision. The
> psychological-correlate module always reports probability distributions
> across conditions, never a single predicted label.

---

## Project status

**Phase 1 — Foundation & Setup** (in progress).

Shipped so far:

- Repo scaffold (folders, requirements, gitignore)
- `src/preprocessing/dicom_to_png.py` — DICOM series → normalised 2D PNG slices
- `src/preprocessing/nifti_to_yolo.py` — BraTS NIfTI volumes → YOLOv11-seg
  polygon dataset
- `src/preprocessing/hemorrhage_to_yolo.py` — PhysioNet Hssayeni CT slices +
  hemorrhage masks → YOLOv11-seg polygon dataset (class id 1, alongside glioma=0)
- `src/utils/visualization.py` — YOLO-seg polygon overlay tool for QA
- `data/processed/anomaly/dataset.yaml` — unified class taxonomy (glioma=0, hemorrhage=1)
- Phase 1 datasets downloaded and **fully converted**:
  - BraTS 2024 → `data/processed/anomaly/` (glioma, class 0)
  - PhysioNet Hssayeni hemorrhage CT → `data/processed/anomaly/` (318 positive slices, class 1)

See [Roadmap](#roadmap) for what's next.

---

## Folder structure

```
brain-cv-project/
├── BRIEFING.md                  Project brief — do not modify
├── CLAUDE.md                    Working context for Claude Cowork sessions
├── README.md                    This file
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                     Original downloads (gitignored)
│   ├── processed/
│   │   ├── anomaly/             PNG slices + YOLO-seg annotations
│   │   └── psych/               Skull-stripped, MNI-registered slices
│   └── splits/                  Train/val/test manifests (committed)
│
├── src/
│   ├── preprocessing/
│   │   ├── dicom_to_png.py
│   │   ├── nifti_to_yolo.py
│   │   ├── hemorrhage_to_yolo.py
│   │   └── mri_register.py      (Phase 4)
│   ├── models/
│   │   ├── segmentation/        (Phase 2)
│   │   ├── grading/             (Phase 3)
│   │   └── psych/               (Phase 4)
│   ├── pipeline/
│   │   └── inference.py         (Phase 5)
│   └── utils/
│       ├── metrics.py           (Phase 2+)
│       └── visualization.py
│
├── notebooks/
├── app/
│   └── dashboard.py             (Phase 5)
├── configs/
│   ├── anomaly_train.yaml
│   └── psych_train.yaml
├── experiments/                 Weights & W&B logs (gitignored)
└── reports/
    └── model_cards/
```

`data/raw/`, `data/processed/`, and `experiments/` are gitignored — they are
regenerated locally from the source datasets and training runs.

---

## Prerequisites

- **Python 3.10+** with `pip` and `venv`
- **CUDA-capable GPU** strongly recommended for training (CPU works for the
  preprocessing scripts but is slow)
- **Git** (the project is designed to be developed across multiple machines —
  see [Multi-device workflow](#multi-device-workflow))
- *(Phase 4 only)* **FSL** and **ANTs** for skull-stripping and MNI
  registration. These are standalone tools, not pip packages.
  - FSL: <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>
  - ANTs: <https://stnava.github.io/ANTs/>

---

## Installation

Clone and set up a virtual environment:

```
git clone <your-fork-url> brain-cv-project
cd brain-cv-project

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the install:

```
python -c "import torch, ultralytics, nibabel, cv2; print('OK')"
```

---

## Datasets

All datasets live under `data/raw/` (gitignored). You'll need to obtain them
yourself — most require free registration or a data-use agreement.

| # | Dataset | What it provides | Source | Approx. size |
|---|---------|------------------|--------|--------------|
| 1 | **BraTS 2024** (Glioma) | NIfTI volumes (T1, T1c, T2, FLAIR) + multi-region segmentation masks; supports WHO grade derivation | <https://www.synapse.org/brats> | ~30 GB |
| 2 | **Kaggle Brain Tumor MRI** (Masoudnickparvar) | 4-class JPG dataset: glioma / meningioma / pituitary / notumor — classification only, no masks | <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset> | ~150 MB |
| 3 | **Intracranial Hemorrhage CT** (PhysioNet Hssayeni v1.0.0) | 2D CT slices with per-slice subtype labels and pixel-level hemorrhage masks | <https://physionet.org/content/ct-ich/1.3.1/> | ~2 GB |
| 4 | *(Phase 4)* **ADHD-200** | rs-fMRI / structural MRI for ADHD vs control | <http://fcon_1000.projects.nitrc.org/indi/adhd200/> | varies |
| 5 | *(Phase 4)* **ABIDE I & II** | Structural MRI for ASD vs control | <http://fcon_1000.projects.nitrc.org/indi/abide/> | varies |
| 6 | *(Phase 4)* **OpenNeuro ds000030** | ADHD, bipolar, schizophrenia, control | <https://openneuro.org/datasets/ds000030> | varies |

Place each dataset under `data/raw/` using these exact folder names so the
preprocessing scripts find them without modification:

```
data/raw/
├── training_data1_v2/                   ← BraTS 2024 (one subfolder per case)
│   ├── BraTS-GLI-02251-100/
│   │   ├── BraTS-GLI-02251-100-seg.nii.gz
│   │   ├── BraTS-GLI-02251-100-t1c.nii.gz
│   │   ├── BraTS-GLI-02251-100-t1n.nii.gz
│   │   ├── BraTS-GLI-02251-100-t2f.nii.gz
│   │   └── BraTS-GLI-02251-100-t2w.nii.gz
│   └── ...
│
├── brain-tumor-mri/                     ← Kaggle Masoudnickparvar
│   ├── Training/{glioma,meningioma,pituitary,notumor}/*.jpg
│   └── Testing/{glioma,meningioma,pituitary,notumor}/*.jpg
│
└── hemorrhage-ct/                       ← PhysioNet Hssayeni
    └── computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/
        ├── hemorrhage_diagnosis.csv
        ├── patient_demographics.csv
        ├── README.txt
        └── Patients_CT/<patient_id>/{bone,brain}/<slice>.jpg
```

---

## Scripts shipped so far

All scripts use `pathlib`, type hints, docstrings, and an `argparse` CLI.
Run them as modules from the repo root so package-relative imports resolve.

### `src/preprocessing/dicom_to_png.py`

Converts a folder of DICOM files into z-score-normalised 2D axial PNG slices,
one PNG per slice, grouped by `SeriesInstanceUID` and sorted by
`ImagePositionPatient`.

```
python -m src.preprocessing.dicom_to_png --input <dicom_folder> --output data/processed/anomaly/images
```

Useful flags:

- `--series-prefix` — use the full `SeriesInstanceUID` in output filenames
  instead of the last 12 characters.

### `src/preprocessing/nifti_to_yolo.py`

Converts BraTS-style NIfTI segmentation volumes into a YOLOv11-seg training
dataset: per-slice PNG images plus polygon label files in YOLO-seg format
(`<class_id> x1 y1 x2 y2 ... xN yN`, coordinates normalised to `[0, 1]`).

```
python -m src.preprocessing.nifti_to_yolo --input data/raw/training_data1_v2 --modality t1c --label-mode combined
```

Output lands in `data/processed/anomaly/{images,labels}/` plus a
`dataset.yaml` stub for Ultralytics.

Useful flags:

- `--modality {t1c,t1n,t2f,t2w}` — which modality is saved as the image
  (default `t1c`).
- `--label-mode {combined,subregions}` — `combined` collapses all four BraTS
  labels (1–4) into one `glioma` class; `subregions` keeps NETC / SNFH / ET /
  RC as separate classes.
- `--axis {0,1,2}` — slicing axis (default `2` = axial Z).
- `--include-empty` — also emit images and 0-byte label files for slices with
  no tumor (YOLO uses them as negatives).
- `--min-polygon-area`, `--approx-epsilon-frac` — polygon filtering and
  simplification knobs.

Safe to `Ctrl+C` between cases — conversion is atomic per case.

### `src/utils/visualization.py`

Overlays YOLO-seg polygon labels on their matching PNG slices and saves the
annotated copies to an output directory. Use this to sanity-check converter
output before kicking off a training run.

```
python -m src.utils.visualization --images data/processed/anomaly/images --labels data/processed/anomaly/labels --output data/processed/anomaly/overlays --limit 30
```

Useful flags:

- `--limit N` — only process the first N images.
- `--thickness N` — outline thickness in pixels (default 1).
- `--alpha 0.3` — also blend a translucent fill over each polygon.
- `--skip-empty` — skip slices whose label file is empty.

Class colours cycle through a fixed palette (BGR): yellow, blue, green, red,
magenta, cyan — so in `--label-mode subregions` each tumor sub-region is
visible as a distinct colour.

### `src/preprocessing/hemorrhage_to_yolo.py`

Converts the PhysioNet Hssayeni intracranial hemorrhage CT dataset into the
same YOLO-seg format as `nifti_to_yolo.py`. Reads `hemorrhage_diagnosis.csv`
for per-slice multi-label annotations and pairs each brain CT slice with its
`_HGE_Seg.jpg` binary mask.

```
python -m src.preprocessing.hemorrhage_to_yolo --input "data/raw/hemorrhage-ct/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0" --output data/processed/anomaly --label-mode combined --combined-class-id 1
```

Output lands in the same `data/processed/anomaly/{images,labels}/` directory
as the BraTS output — filenames never collide (BraTS stems start with
`BraTS-GLI-…`, hemorrhage stems are `NNNN_sliceNNNN`).

Useful flags:

- `--label-mode {combined,subtypes}` — `combined` maps all hemorrhage to a
  single class; `subtypes` uses per-subtype class ids
  (Intraventricular=0 … Subdural=4).
- `--combined-class-id N` — YOLO class id for hemorrhage in combined mode
  (default `1` to sit alongside glioma=0).
- `--include-empty` — also write negative (no-hemorrhage) slices.
- `--min-polygon-area`, `--approx-epsilon-frac` — same polygon knobs as
  `nifti_to_yolo.py`.

### End-to-end Phase 1 walkthrough

```
# 1. Convert BraTS NIfTI volumes → YOLO-seg (glioma, class 0)
python -m src.preprocessing.nifti_to_yolo --input data/raw/training_data1_v2 --modality t1c --label-mode combined

# 2. Convert Hssayeni hemorrhage CT → YOLO-seg (hemorrhage, class 1)
python -m src.preprocessing.hemorrhage_to_yolo --input "data/raw/hemorrhage-ct/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0" --output data/processed/anomaly --label-mode combined --combined-class-id 1

# 3. Spot-check a few overlays
python -m src.utils.visualization --images data/processed/anomaly/images --labels data/processed/anomaly/labels --output data/processed/anomaly/overlays --limit 30

# 4. (Next — train/val/test split manifests, augmentation pipeline, training)
```

> **PowerShell note:** these are single-line commands so they paste cleanly
> on Windows. If you want to break them across lines, use a backtick `` ` ``
> for continuation in PowerShell (not the bash backslash `\`).

---

## Roadmap

| Phase | Focus | Headline deliverable |
|------:|-------|----------------------|
| **1** | Foundation & Setup | Clean normalised dataset with YOLO-seg compatible annotations, documented class splits |
| 2 | Core Segmentation | Trained YOLOv11-seg with per-class metrics (Dice, mIoU) |
| 3 | Grading & Severity | Anomaly pipeline ending in grade + severity score (volume, shape irregularity) |
| 4 | Psychological Correlates | Probabilistic condition classifier (ADHD / ASD / Control) with Grad-CAM saliency |
| 5 | Integration & Presentation | Unified inference pipeline + Streamlit / Gradio demo + technical report |

See `BRIEFING.md` for full per-phase task lists, time estimates, and design
decisions.

---

## Tech stack

- **Python 3.10+**, **PyTorch 2.x** with CUDA
- **Ultralytics YOLOv11** for segmentation
- **SimpleITK**, **nibabel**, **pydicom** for medical-image I/O
- **OpenCV**, **scikit-image** for contour / image processing
- **captum** for Grad-CAM explainability
- **Weights & Biases** for experiment tracking
- **Streamlit** or **Gradio** for the Phase 5 dashboard
- **FSL BET**, **ANTs** (external tools, Phase 4) for skull stripping and
  MNI152 registration

Full pin list in `requirements.txt`.

---

## Multi-device workflow

This project is developed across a Mac and a Windows PC. The full state
lives in Git — `data/raw/`, `data/processed/`, and `experiments/` are
gitignored and regenerated per machine.

**Before switching machines:**

```
git add .
git commit -m "session checkpoint: <what changed>"
git push
```

**When resuming on the other machine:**

```
git pull
```

Then re-run any preprocessing scripts as needed to regenerate `data/processed/`.

---

## Coding conventions

- Type hints on all function signatures.
- Docstrings on every public function (`Args`, `Returns`, `Raises`).
- No hardcoded paths — use `pathlib.Path` and load configuration from YAML.
- Preprocessing must be reproducible (set random seeds, log parameters).
- Metrics logged to Weights & Biases at the end of every training run.
- Model weights are never committed; they live under `experiments/`.

---

## Ethical notes

- The psychological-correlate module reports probabilities across conditions,
  not a single predicted label. Single-label "diagnoses" are explicitly out
  of scope.
- This repository must not be used for clinical decision-making.
- Dataset usage is bound by the licences of each source — respect the BraTS,
  ADHD-200, ABIDE, OpenNeuro, and PhysioNet data-use agreements.

---

## Acknowledgments

Datasets courtesy of the BraTS challenge organisers, Masoud Nickparvar's
Kaggle release, Hssayeni et al. (PhysioNet), the ADHD-200 Consortium,
ABIDE, and OpenNeuro. Tooling stands on the shoulders of PyTorch, the
Ultralytics team, the broader nibabel / pydicom / SimpleITK communities,
and Captum's explainability research.

---

## License

TBD — license file to be added before any public release. Until then,
treat the contents of this repository as "all rights reserved" pending
clarification from the author.
