# Brain Anomaly CV Project — Cowork Context

## What this project is
A Computer Vision pipeline for brain MRI analysis with three output layers:
1. Pixel-level segmentation of brain anomalies using YOLOv11-seg
2. Per-anomaly grading and severity via a CNN classification head
3. Neuroimaging correlate detection for ADHD and ASD via a structural MRI classifier
This is a research/portfolio project — not a clinical diagnostic tool.

## Current phase
Phase 1 — Foundation & Setup (in progress)

## Last session summary
Session 1 (2026-05-20):
- Created full project folder structure (data/, src/, notebooks/, app/, configs/,
  experiments/, reports/) with .gitkeep placeholders so all dirs are tracked by Git.
- Created requirements.txt with all Phase 1–5 Python dependencies and version pins.
- Created .gitignore excluding data/raw/, data/processed/, experiments/,
  __pycache__/, *.pyc, *.pth, and other generated/sensitive files.
- Built src/preprocessing/dicom_to_png.py — full DICOM→PNG converter with:
    - Recursive DICOM discovery (find_dicom_files)
    - Series grouping by SeriesInstanceUID (group_by_series)
    - Anatomical z-sort via ImagePositionPatient
    - RescaleSlope/RescaleIntercept application (extract_pixel_array)
    - Per-slice z-score normalisation clipped to ±3σ, rescaled to uint8 (zscore_normalise)
    - PNG output to data/processed/anomaly/images/ with <uid>_slice<NNNN>.png naming
    - Full type hints, docstrings (Args/Returns/Raises), argparse CLI

## Completed milestones
- [2026-05-20] Project scaffolded: folder structure, requirements.txt, .gitignore
- [2026-05-20] src/preprocessing/dicom_to_png.py complete and committed

## Next task
Phase 1 remaining work:
1. Download datasets (see BRIEFING.md Part 6):
   - BraTS 2024 from synapse.org (register first)
   - Kaggle brain tumor MRI dataset (pituitary/meningioma multi-class)
   - Kaggle brain hemorrhage segmentation dataset
2. Build src/preprocessing/nifti_to_yolo.py — convert NIfTI ground-truth masks
   to YOLO-seg polygon annotation format
3. Build src/preprocessing/mri_register.py — skull strip + MNI registration stub
   (full implementation deferred to Phase 4 when FSL/ANTs are installed)
4. Design and lock the class taxonomy (anomaly classes + grade labels)
5. Create stratified train/val/test split manifests → data/splits/
6. Implement augmentation pipeline (horizontal flip, rotation, intensity jitter)

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
