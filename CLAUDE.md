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
Session 3 (2026-05-20):
- Inspected data/raw/ and recorded the exact on-disk layout of all three Phase 1
  datasets (see "Verified data/raw layout" section below).
- Corrected dataset attribution: the "Kaggle brain hemorrhage segmentation"
  dataset is actually the PhysioNet Hssayeni v1.0.0 release
  ("computed-tomography-images-for-intracranial-hemorrhage-detection-and-
  segmentation-1.0.0"). Same data, different distribution channel.
- Confirmed scope of nifti_to_yolo.py: BraTS only. Hemorrhage CT is already 2D
  JPG with paired masks and needs a separate converter. Brain-tumor-mri is
  classification-only (no masks) and feeds the Phase 3 grading head.
- Built src/preprocessing/nifti_to_yolo.py — full BraTS → YOLOv11-seg converter:
    - find_brats_cases / resolve_case_files (discover + pair seg with modality)
    - load_nifti_volume + iter_axial_slices (nibabel + np.take per axis)
    - mask_to_polygons (cv2.findContours RETR_EXTERNAL + approxPolyDP)
    - polygon_to_yolo_line (normalised polygon vertices)
    - convert_case / convert_dataset (top-level batch entry points)
    - _write_dataset_yaml (emits dataset stub for Ultralytics)
    - Two label-mode taxonomies:
        "combined"   → all BraTS labels collapse to class 0 (glioma)
        "subregions" → NETC=0, SNFH=1, ET=2, RC=3
    - CLI flags: --modality, --label-mode, --axis, --include-empty,
                 --min-polygon-area, --approx-epsilon-frac
- Bash sandbox unavailable this session (Windows EXDEV mount error) so
  py_compile / a sample-case dry-run could NOT be executed remotely. User
  ran the smoke test locally instead.
- Built src/utils/visualization.py — YOLO-seg polygon overlay tool for QA:
    - parse_yolo_seg_label, denormalise_polygon, overlay_yolo_seg,
      overlay_directory + CLI (--images, --labels, --output, --limit,
      --thickness, --alpha, --skip-empty).
    - Per-class colour palette (BGR): 0=yellow, 1=blue, 2=green, 3=red,
      4=magenta, 5=cyan. Reusable for hemorrhage converter QA and later
      for visualising YOLO model predictions.
- Smoke-tested nifti_to_yolo.py on a subset of BraTS cases:
    - Polygons in --label-mode combined correctly trace tumor regions on T1c.
    - Multiple disjoint polygons per slice render as expected for tumors with
      separate necrotic / enhancing / edema components.
    - --include-empty produces 0-byte label files; YOLO treats these as
      negatives during training — keep them unless cleaning up.
- Documented a PowerShell gotcha: multi-line python commands need backtick `
  for line continuation, not bash-style backslash \. Stick to single-line
  invocations in CLAUDE.md examples to avoid the issue across Mac + PC.

Session 2 (2026-05-20):
- All Phase 1 datasets downloaded into data/raw/ (gitignored — not committed).
  Includes BraTS 2024, Kaggle brain tumor MRI (pituitary/meningioma multi-class),
  and Kaggle brain hemorrhage segmentation dataset. Folder layout under data/raw/
  should be confirmed at the start of next session before writing the NIfTI→YOLO
  converter, since the converter's path logic depends on the actual folder names.

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
- [2026-05-20] All Phase 1 datasets downloaded to data/raw/ (BraTS 2024,
  Kaggle brain tumor MRI, PhysioNet Hssayeni hemorrhage CT v1.0.0)
- [2026-05-20] data/raw/ layout verified and documented (see section below)
- [2026-05-20] src/preprocessing/nifti_to_yolo.py complete (BraTS → YOLO-seg)
- [2026-05-20] src/utils/visualization.py complete (YOLO-seg overlay viz)
- [2026-05-20] nifti_to_yolo.py smoke test passed — polygon outlines visually
  verified on representative BraTS cases via visualization.py overlay
- [2026-05-21] README.md added at repo root — public-facing overview, install
  instructions, dataset acquisition guide, and usage examples for all
  scripts shipped to date (dicom_to_png, nifti_to_yolo, visualization)

## Verified data/raw layout
All paths relative to repo root. data/raw/ is gitignored.

1. BraTS 2024 (glioma segmentation + multi-label grade via seg classes)
   - Root: data/raw/training_data1_v2/
   - One folder per case: BraTS-GLI-<id>-<timepoint>/
     e.g. BraTS-GLI-02251-100/
   - Files per case (5):
     <case>-seg.nii.gz   ← ground-truth segmentation (label values 1/2/3/4)
     <case>-t1c.nii.gz   ← T1-weighted contrast-enhanced
     <case>-t1n.nii.gz   ← T1-weighted native
     <case>-t2f.nii.gz   ← T2 FLAIR
     <case>-t2w.nii.gz   ← T2-weighted
   - Use for: YOLOv11-seg segmentation training + WHO grade derivation.

2. Kaggle brain tumor MRI (Masoudnickparvar — 4-class classification, no masks)
   - Root: data/raw/brain-tumor-mri/
   - Layout: {Training,Testing}/{glioma,meningioma,pituitary,notumor}/*.jpg
   - File-name prefixes: Tr-gl_*, Tr-pi_*, Tr-no_*, Tr-me_*, plus
     Tr-aug-me_* (meningioma augmented to balance classes), and Te-*_ for test.
   - Use for: Phase 3 grading/classification head only. Cannot feed YOLO-seg
     (no pixel masks available).

3. PhysioNet Hssayeni intracranial hemorrhage CT (v1.0.0)
   - Root: data/raw/hemorrhage-ct/
     computed-tomography-images-for-intracranial-hemorrhage-detection-and-
     segmentation-1.0.0/
   - Files at root: README.txt, LICENSE.txt, SHA256SUMS.txt,
     hemorrhage_diagnosis.csv, patient_demographics.csv
   - hemorrhage_diagnosis.csv columns: PatientNumber, SliceNumber,
     Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural,
     No_Hemorrhage, Fracture_Yes_No  (per-slice multi-label)
   - Slice images: Patients_CT/<patient_id>/{bone,brain}/<slice_num>.jpg
   - Masks: Patients_CT/<patient_id>/brain/<slice_num>_HGE_Seg.jpg
     (present only for slices with hemorrhage)
   - Use for: YOLOv11-seg (hemorrhage class) AND per-slice subtype labels.
     Needs a different converter from nifti_to_yolo.py — already 2D JPG.

## Next task
Phase 1 remaining work:
1. Run the full BraTS → YOLO-seg conversion against the real output dir.
   Single-line PowerShell-friendly invocation:
     python -m src.preprocessing.nifti_to_yolo --input data/raw/training_data1_v2 --modality t1c --label-mode combined
   Output lands in data/processed/anomaly/{images,labels} + dataset.yaml.
   Expect ~50k–100k PNG/label pairs and several GB. Safe to Ctrl+C between
   cases (atomic per case). After completion, spot-check overlays from later
   cases with src/utils/visualization.py.
2. Build src/preprocessing/hemorrhage_to_yolo.py — convert the Hssayeni
   <slice>.jpg + <slice>_HGE_Seg.jpg pairs to YOLO-seg polygons. Class label
   from hemorrhage_diagnosis.csv (per-slice subtype).
3. Build src/preprocessing/mri_register.py — skull strip + MNI registration stub
   (full implementation deferred to Phase 4 when FSL/ANTs are installed)
4. Design and lock the class taxonomy (anomaly classes + grade labels) —
   reconcile combined vs subregions label-mode with the hemorrhage subtypes
   and the brain-tumor-mri 4-class set.
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
