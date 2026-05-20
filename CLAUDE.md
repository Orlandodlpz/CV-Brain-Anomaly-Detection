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