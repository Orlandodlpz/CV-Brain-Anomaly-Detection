# Brain Anomaly Detection — Cowork Briefing & Project Handoff

> Read this entire document before starting any work.
> It contains the full project context, architectural decisions, 5-phase roadmap,
> dataset sources, multi-device workflow rules, and your working CLAUDE.md.

---

## Part 1 — Setting Up Claude Cowork

### What is Cowork?
Cowork is Anthropic's AI desktop agent built into the Claude Desktop app. It operates
directly on your local files and folders, executes multi-step tasks autonomously, and
works without a terminal or coding knowledge. It uses the same agentic architecture as
Claude Code but is designed for knowledge work beyond just coding.

### Requirements
- Claude Desktop app (macOS or Windows) — download at **claude.com/download**
- A paid Claude plan (Pro at $20/mo minimum — Cowork is included)
- Windows users: use the MSIX installer, not the older .exe installer, for full Cowork
  support including the required VM service

### Starting Cowork
1. Open Claude Desktop
2. Sign in with your Claude account
3. Switch to the **Cowork** tab (or Tasks mode)
4. Click **Work in a Folder** and select your project folder
5. Grant the requested file read/write permissions
6. Describe your task in plain language — Cowork plans and executes autonomously

### Important runtime rules
- The desktop app must stay open and your computer must stay awake while tasks run
- Cowork operates on local files only — files must exist on the machine you are using
- Long-running tasks (training runs, large preprocessing jobs) should be started on the
  machine you can leave running

---

## Part 2 — Multi-Device Workflow (Mac + Windows)

You will be working on this project from two machines. Cowork local sessions stay
on-device, so your project files do not automatically follow you between machines.
The solution is Git.

### Setup on both machines
1. Create a GitHub repository for this project
2. Place `CLAUDE.md` and `BRIEFING.md` in the repo root and commit them
3. Clone the repo on both your Mac and your Windows PC
4. When starting a Cowork session, always point it to the local clone

### Before switching machines — always do this
```
git add .
git commit -m "session checkpoint: [brief description of what was done]"
git push
```

### When picking up on the other machine
```
git pull
```
Then open Cowork, point it to the local clone, and start with:
> "Read CLAUDE.md and BRIEFING.md before starting. Summarize where we left off."

### What Git should and should not track

**Commit these:**
- All source code (`src/`)
- Configs (`configs/`)
- CLAUDE.md (updated every session — see Part 3)
- BRIEFING.md
- Split manifests (`data/splits/`)
- Requirements and documentation

**Do NOT commit these (add to .gitignore):**
- `data/raw/` — too large, re-download from source
- `data/processed/` — regenerate via preprocessing scripts
- `experiments/` — model weights and W&B logs
- `__pycache__/`, `.pyc` files
- Any API keys or credentials

---

## Part 3 — CLAUDE.md Update Rules (Critical)

CLAUDE.md is the only persistent memory Cowork has between sessions. Since you are
working across two devices and multiple sessions, keeping it current is what makes
the project coherent over time.

### Auto-update rules for Cowork to follow

**Rule 1 — Update after every significant event, without being asked:**
- A phase is completed or started
- An architectural decision is made or changed
- A dataset is downloaded, preprocessed, or found to be unusable
- A model is trained and evaluated (log the key metrics)
- A bug is found that changes how something is built
- A tool or library is swapped out for another

**Rule 2 — Ask to update after every 5 prompts:**
Every 5 user prompts within a session, pause and say:
> "We've covered a few things — should I update CLAUDE.md before we continue?"

If the user says yes, update immediately. If no, reset the counter and ask again after
5 more prompts.

**Rule 3 — Always update at end of session:**
When the user signals they are done (e.g. "that's it for today", "let's stop here",
"I'll continue later"), update CLAUDE.md before closing out. Include:
- Current phase and status
- What was completed this session
- What is next
- Any decisions or blockers encountered

**Rule 4 — Commit CLAUDE.md after updating:**
After every CLAUDE.md update, stage and commit it:
```
git add CLAUDE.md
git commit -m "claude.md: update after [brief description]"
git push
```
This ensures the other device gets the latest context on the next `git pull`.

---

## Part 4 — Project Context (Full Summary)

### What we are building
A Computer Vision pipeline for brain MRI analysis with three capability layers:

1. **Anomaly segmentation** — pixel-level detection and masking of brain anomalies
2. **Grading and severity** — per-anomaly classification into clinical grade/severity
3. **Psychological correlates** — probabilistic neuroimaging correlate detection for
   conditions like ADHD and ASD, framed explicitly as research correlates, not diagnosis

### Core architectural decisions

**Segmentation model: YOLOv11-seg**
Chosen over object detection because bounding boxes are too imprecise for medical
imaging. Segmentation gives pixel-level masks which carry shape, boundary, and volume
information. YOLOv11-seg produces the mask AND the class label in a single forward pass.

**Classification head for grading**
A secondary CNN (ResNet50 or EfficientNet-B2) fed masked ROI crops from YOLOv11-seg
output. Responsible for fine-grained grading (e.g. WHO Grade I/II vs III/IV for glioma)
and derived severity metrics: lesion volume from mask area, shape irregularity from
contour circularity index. This is a separate model from YOLO, not part of its head.

**Psychological correlates module — different approach entirely**
This module does NOT use YOLO. Psychological traits have no localizable bounding box.
Uses structural MRI preprocessing (skull stripping via FSL BET, MNI space registration).
Trains a CNN classifier on volumetric features or end-to-end on registered MRI slices.
Outputs a probability distribution across conditions — never a single predicted label.
Grad-CAM saliency maps show which brain regions contributed to each prediction.

**Critical framing rule — never violate this:**
All psychological correlate outputs must use probabilistic language.
Correct:   `{"adhd_probability": 0.72, "asd_probability": 0.18, "control": 0.10}`
Incorrect: `{"predicted_condition": "ADHD"}`

### Full inference pipeline (end state)

```
Single MRI input (NIfTI or DICOM)
          │
          ├── Branch A: Anomaly detection
          │     Preprocessing: DICOM/NIfTI → 2D PNG slices → normalize
          │     └── YOLOv11-seg → mask + coarse class
          │           └── Masked ROI crop → CNN head → grade + severity
          │
          └── Branch B: Psychological correlates
                Preprocessing: skull strip (FSL BET) → MNI registration → slices
                └── CNN classifier → condition probabilities + Grad-CAM overlay
```

### Target anomaly classes (Phase 2)
Glioma, Meningioma, Pituitary adenoma, Hemorrhage, Stroke lesion

### Target grading (Phase 3)
- Glioma: WHO Grade I/II (LGG) vs WHO Grade III/IV (HGG) — from BraTS labels
- Other anomalies: volume-based severity (mild/moderate/severe) derived from mask area
- Shape irregularity score (circularity index) as additional severity indicator

### Target psychological correlates (Phase 4)
Control, ADHD, ASD — OCD/schizophrenia only if dataset size permits.
Output: probability score per class, not a single predicted class.

---

## Part 5 — 5-Phase Roadmap

### Phase 1 — Foundation & Setup (2–3 weeks)
**Goal:** Working environment + clean annotated dataset ready for training

Tasks:
- Set up Python environment: PyTorch, CUDA, Ultralytics, SimpleITK, nibabel, OpenCV
- Download all datasets (see Part 6)
- Build DICOM/NIfTI → 2D PNG slice preprocessing pipeline
- Design and lock the full class taxonomy before writing any model code
- Implement augmentation: horizontal flip, rotation, intensity jitter
- Create stratified train/val/test splits across all classes

Deliverable: Clean normalized dataset with YOLO-seg compatible annotations and
documented class splits.

---

### Phase 2 — Core Segmentation Model (4–5 weeks)
**Goal:** YOLOv11-seg producing accurate masks + coarse class labels

Tasks:
- Convert NIfTI ground-truth masks to YOLO-seg polygon annotation format
- Train YOLOv11-seg on multi-class anomaly dataset
- Handle class imbalance via weighted loss or targeted oversampling
- Evaluate: Dice coefficient, mIoU, per-class precision/recall
- Iterate: error analysis on validation set → corrections → retrain

Tools: Ultralytics YOLOv11, Weights & Biases, CVAT (annotation review), torchmetrics

Deliverable: Trained YOLOv11-seg model with documented per-class metrics.

---

### Phase 3 — Grading & Severity (3–4 weeks)
**Goal:** Per-anomaly grade label and severity score appended to pipeline output

Tasks:
- Build CNN classification head (ResNet50 or EfficientNet-B2) fed masked ROI crops
- Train glioma grading using BraTS HGG/LGG split (WHO Grade I-II vs III-IV)
- Compute volume-based severity from mask pixel area → estimated volume in mm³
- Compute circularity index from contour for shape irregularity score
- Wire full pipeline: YOLO mask → ROI crop → classification head → grade + severity
- Evaluate grading accuracy and confidence calibration on held-out BraTS test set

Tools: PyTorch (custom CNN head), torchvision, scikit-image, OpenCV contour analysis

Deliverable: Full anomaly pipeline (segment → classify → grade) with per-class metrics.

---

### Phase 4 — Psychological Correlates (4–5 weeks)
**Goal:** Condition probability scores + Grad-CAM saliency maps from structural MRI

Tasks:
- Preprocess structural MRI: skull stripping (FSL BET), registration to MNI152 space
- Train multi-class CNN: Control, ADHD, ASD (+ OCD/schizophrenia if data allows)
- Generate Grad-CAM saliency maps per predicted class using captum
- Validate predicted regions against neuroimaging literature (PFC for ADHD, etc.)
- All outputs must use explicit probabilistic framing — no diagnostic claims

Tools: FSL / ANTs (external install), PyTorch, captum (Grad-CAM), nilearn

Deliverable: Condition classifier with probability outputs + saliency overlay pipeline.

---

### Phase 5 — Integration & Presentation (2–3 weeks)
**Goal:** Unified demo app + technical documentation

Tasks:
- Build unified inference pipeline: single MRI input → all module outputs in one call
- Develop Streamlit or Gradio dashboard
- Display: segmentation overlays, grade labels, severity bars, condition probability charts
- Explainability panel: Grad-CAM heatmaps for both anomaly and psych modules
- Add prominent research disclaimer in the UI
- Write technical report with model cards, dataset disclosures, and limitations

Tools: Streamlit or Gradio, matplotlib, Plotly, PIL/OpenCV, ONNX (optional export)

Deliverable: Working demo app + written technical report.

---

## Part 6 — Datasets

### Anomaly detection & grading (Phases 2 & 3)

| Dataset | What it provides | Source |
|---|---|---|
| BraTS 2024 | Glioma masks + HGG/LGG grade labels | synapse.org (free registration) |
| RSNA Brain Tumor | Multi-type MRI with tumor annotations | kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification |
| Kaggle Hemorrhage CT | CT scans with hemorrhage segmentation | kaggle.com — search "brain hemorrhage segmentation" |
| Pituitary/Meningioma | Multi-class brain tumor MRI | kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |

### Psychological correlates (Phase 4)

| Dataset | Conditions | Source |
|---|---|---|
| ADHD-200 | ADHD vs Control | fcon_1000.projects.nitrc.org/indi/adhd200 |
| ABIDE I & II | ASD vs Control | fcon_1000.projects.nitrc.org/indi/abide |
| OpenNeuro ds000030 | ADHD, bipolar, schizophrenia, Control | openneuro.org/datasets/ds000030 |

> BraTS requires free registration at synapse.org.
> ADHD-200 and ABIDE require a data use agreement — allow a few days for approval.

---

## Part 7 — Suggested Folder Structure

```
brain-cv-project/
│
├── CLAUDE.md                    ← Updated every session by Cowork
├── BRIEFING.md                  ← This document — do not modify
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/                     ← Original downloads (gitignored)
│   ├── processed/
│   │   ├── anomaly/             ← PNG slices + YOLO-seg annotations
│   │   └── psych/               ← Skull-stripped, MNI-registered slices
│   └── splits/                  ← Train/val/test manifests (committed)
│
├── src/
│   ├── preprocessing/
│   │   ├── dicom_to_png.py
│   │   ├── nifti_to_yolo.py
│   │   └── mri_register.py
│   ├── models/
│   │   ├── segmentation/
│   │   ├── grading/
│   │   └── psych/
│   ├── pipeline/
│   │   └── inference.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
│
├── notebooks/
├── app/
│   └── dashboard.py
├── configs/
│   ├── anomaly_train.yaml
│   └── psych_train.yaml
├── experiments/                 ← Weights & checkpoints (gitignored)
└── reports/
    └── model_cards/
```

---

## Part 8 — requirements.txt

```
# Core
torch>=2.2.0
torchvision>=0.17.0

# YOLO
ultralytics>=8.3.0

# Medical imaging
SimpleITK>=2.3.0
nibabel>=5.2.0
pydicom>=2.4.0
nilearn>=0.10.0

# Computer vision
opencv-python>=4.9.0
Pillow>=10.2.0
scikit-image>=0.22.0

# ML utilities
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0

# Explainability
captum>=0.7.0

# Experiment tracking
wandb>=0.16.0

# Visualization & app
matplotlib>=3.8.0
plotly>=5.19.0
streamlit>=1.32.0

# Metrics
torchmetrics>=1.3.0
```

> FSL and ANTs are standalone tools, not pip packages.
> Install FSL from fsl.fmrib.ox.ac.uk and ANTs from stnava.github.io/ANTs.
> These are only needed for Phase 4.

---

## Part 9 — CLAUDE.md (copy this into your project root as a separate file)

```markdown
# Brain Anomaly CV Project — Cowork Context

## What this project is
A Computer Vision pipeline for brain MRI analysis with three output layers:
1. Pixel-level segmentation of brain anomalies using YOLOv11-seg
2. Per-anomaly grading and severity via a CNN classification head
3. Neuroimaging correlate detection for ADHD and ASD via a structural MRI classifier
This is a research/portfolio project — not a clinical diagnostic tool.

## Current phase
Phase 1 — Foundation & Setup (not started)

## Last session summary
No sessions completed yet.

## Completed milestones
None yet.

## Next task
Set up the folder structure, requirements.txt, and the DICOM → PNG preprocessing script.

## Key decisions (do not change without discussing first)
- Segmentation: YOLOv11-seg only — not object detection mode
- Grading head: ResNet50 or EfficientNet-B2 fed masked ROI crops from YOLO output
- Psych module: separate CNN classifier — NOT YOLO — on skull-stripped MNI-registered MRI
- Explainability: Grad-CAM via captum for grading head and psych classifier
- Experiment tracking: Weights & Biases (wandb)
- Dashboard: Streamlit or Gradio (decide in Phase 5)

## Critical framing rule — never violate
Psychological correlate outputs must always be probability distributions.
CORRECT:   {"adhd_probability": 0.72, "asd_probability": 0.18, "control": 0.10}
INCORRECT: {"predicted_condition": "ADHD"}

## Tech stack
- Python 3.10+, PyTorch 2.x with CUDA, Ultralytics YOLOv11
- SimpleITK, nibabel, pydicom for medical imaging I/O
- FSL BET + ANTs for Phase 4 preprocessing (external tools, not pip)
- captum for Grad-CAM, wandb for experiment tracking
- Streamlit or Gradio for Phase 5 dashboard

## Coding conventions
- Type hints on all function signatures
- Docstrings on every public function (Args, Returns, Raises)
- No hardcoded paths — use pathlib.Path and load from YAML configs
- All preprocessing must be reproducible (set random seeds, log parameters)
- Metrics logged to Weights & Biases at end of every training run
- No model weights committed to Git — use experiments/ which is gitignored

## Multi-device workflow
This project is worked on from two machines: a Mac and a Windows PC.
After every CLAUDE.md update, always run:
  git add CLAUDE.md
  git commit -m "claude.md: [brief reason for update]"
  git push
When resuming on the other machine, always run git pull before starting.

## CLAUDE.md update rules
1. Update automatically after any significant event (phase change, decision, training
   result, dataset issue, tool change)
2. Ask "Should I update CLAUDE.md?" after every 5 user prompts
3. Always update at end of session when the user signals they are done
4. Always commit and push CLAUDE.md immediately after updating it
```

---

## Part 10 — First Prompt to Give Cowork

Once Cowork is open and pointed at your project folder, start with this:

```
Read CLAUDE.md and BRIEFING.md fully before doing anything else.

Then set up the project:
1. Create the full folder structure from BRIEFING.md Part 7
2. Create requirements.txt from BRIEFING.md Part 8
3. Create a .gitignore that excludes data/raw/, data/processed/,
   experiments/, __pycache__/, *.pyc, and *.pth files
4. Build src/preprocessing/dicom_to_png.py — a script that takes a
   folder of DICOM files, converts each series to 2D axial PNG slices,
   applies z-score normalization per slice, and saves to
   data/processed/anomaly/images/
   Use pathlib for all paths, add type hints and docstrings, and make
   it runnable from the command line with argparse.

When done, update CLAUDE.md with what was completed and what is next,
then commit and push everything.
```

---

## Useful References

- Cowork help: https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork
- Ultralytics YOLOv11: https://docs.ultralytics.com
- BraTS challenge: https://www.synapse.org/brats
- ADHD-200 dataset: http://fcon_1000.projects.nitrc.org/indi/adhd200
- ABIDE dataset: http://fcon_1000.projects.nitrc.org/indi/abide
- FSL install: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
- captum (Grad-CAM): https://captum.ai
- Weights & Biases: https://wandb.ai

---

*Generated from project planning session — May 2026*
```
