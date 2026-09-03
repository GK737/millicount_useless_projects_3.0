# Millipede Segment & Ring Dataset for YOLO

A comprehensive dataset and computer vision pipeline for detecting, identifying, and sequentially counting the number of body rings/segments (annuli/diplosegments) in millipedes using Ultralytics YOLO (YOLOv8 / YOLO11).

---

## Table of Contents
1. [Overview & Problem Formulation](#overview--problem-formulation)
2. [Why YOLO-OBB (Oriented Bounding Boxes)?](#why-yolo-obb-oriented-bounding-boxes)
3. [Dataset Architecture & Directory Layout](#dataset-architecture--directory-layout)
4. [Quick Start Guide](#quick-start-guide)
5. [Tools & Modules](#tools--modules)
   - [1. Procedural Synthetic Generator](#1-procedural-synthetic-generator)
   - [2. iNaturalist Real Macro Image Fetcher](#2-inaturalist-real-macro-image-fetcher)
   - [3. Assisted Auto-Annotator](#3-assisted-auto-annotator)
   - [4. Dataset Visualizer & QC](#4-dataset-visualizer--qc)
   - [5. Segment Counting & Gap Detection Engine](#5-segment-counting--gap-detection-engine)
6. [Training YOLO](#training-yolo)
7. [Classes Specification](#classes-specification)

---

## Overview & Problem Formulation

Millipedes (*Class Diplopoda*) are characterized by repetitive segmented body rings. Counting these segments is crucial for biological taxonomy, developmental stage tracking, and phenotypic analysis.

However, counting segments using standard computer vision object detection is challenging:
- Millipedes have between **25 and 100+ narrow segments** tightly packed in sequence.
- When the animal is curved, coiled, or oriented diagonally, standard horizontal/vertical axis-aligned bounding boxes (AABB) heavily overlap.
- Traditional Non-Maximum Suppression (NMS) treats overlapping boxes as duplicates and erroneously deletes neighboring segments.

This dataset solves this by utilizing **YOLO-OBB (Oriented Bounding Boxes)** where each ring is represented as a rotated rectangle aligned perpendicular to the longitudinal spine of the millipede.

---

## Why YOLO-OBB (Oriented Bounding Boxes)?

| Format | Coordinate Format | Behavior on Curved Millipede | NMS Overlap Risk |
|---|---|---|---|
| **YOLO-OBB (Recommended)** | `class x1 y1 x2 y2 x3 y4 x4 y4` | Rotates with the body curve | **None**: Boxes remain disjoint |
| **YOLO-Seg** | `class x1 y1 ... xn yn` | Follows exact ring polygon | **None**: Highly accurate |
| **YOLO-Detect (Standard)** | `class cx cy w h` | Horizontal boxes overlap on diagonals | **High**: Requires low IoU thresholds |

---

## Dataset Architecture & Directory Layout

```
millipede_dataset/
├── data.yaml                          # YOLO dataset configuration
├── train/
│   ├── images/                        # Training images (.jpg / .png)
│   └── labels/                        # YOLO label files (.txt)
├── val/
│   ├── images/                        # Validation images
│   └── labels/                        # Validation labels
├── test/
│   ├── images/                        # Test images
│   └── labels/                        # Test labels
├── raw_images/                        # Unannotated harvested macro photos
├── previews/                          # Visual QC & counting report exports
├── tools/
│   ├── generate_synthetic_dataset.py  # Procedural millipede generator
│   ├── fetch_inaturalist.py           # Real CC-licensed macro downloader
│   ├── assisted_annotator.py          # Semi-automated segment labeling tool
│   ├── visualize_dataset.py           # Visual inspection of annotations
│   └── count_segments.py              # Inference + head-to-tail counting engine
├── train_model.py                     # One-click YOLO training script
└── README.md                          # Documentation
```

---

## Quick Start Guide

### 1. Inspect & Verify Existing Dataset
Visualize current samples with sequential segment numbers:
```bash
python tools/visualize_dataset.py --split train --samples 3
```
Output previews will be saved to `previews/`.

### 2. Generate More Synthetic Annotated Data
Generate 50 more synthetic samples with 25–65 rings, varied backgrounds (soil, bark, stone), and realistic metachronal legs:
```bash
python tools/generate_synthetic_dataset.py --count 50 --format obb
```

### 3. Fetch Real High-Resolution Millipede Photos
Harvest research-grade macro images from iNaturalist:
```bash
python tools/fetch_inaturalist.py --limit 15
```

### 4. Label Real Photos with the Assisted Annotator
Auto-detect rings and generate candidate YOLO labels:
```bash
python tools/assisted_annotator.py --image raw_images/your_image.jpg --label train/labels/your_image.txt --preview previews/your_image.jpg --format obb
```

### 5. Train YOLO Model
```bash
python train_model.py --model yolov8n-obb.pt --epochs 50 --batch 8
```

### 6. Count Segments on a New Image
```bash
python tools/count_segments.py --image test.jpg --model runs/millipede/milli_segment_exp/weights/best.pt
```

---

## Tools & Modules

### 1. Procedural Synthetic Generator (`tools/generate_synthetic_dataset.py`)
Generates 100% pixel-perfect annotations:
- **Parameters**:
  - `--count`: Number of images to generate (default: 50).
  - `--format`: Label format (`obb`, `bbox`, `seg`).
  - `--val-ratio`: Validation split ratio (default: 0.2).
- **Features**:
  - Randomizes curve trajectories (S-curves, arcs, C-coils, meandering).
  - Realistic chitin textures, dorsal specular sheen, and intersegmental sutures.
  - Generates two pairs of articulated legs per diplosegment with metachronal wave oscillation.
  - Procedural backgrounds: forest soil, paver stones, tree bark, leaf litter.

### 2. iNaturalist Real Macro Image Fetcher (`tools/fetch_inaturalist.py`)
- Queries the iNaturalist REST API for Taxon ID 47735 (*Diplopoda*).
- Downloads high-resolution CC-licensed images.
- Saves attribution metadata (`raw_images/metadata.json`) preserving photographer and observation IDs.

### 3. Assisted Auto-Annotator (`tools/assisted_annotator.py`)
Assists in labeling real photographs to eliminate manual click fatigue:
- Slices along the principal body spine to compute median caliber.
- Performs transverse cross-sectional contrast analysis to detect segment sutures.
- Options:
  - `--sensitivity`: Tuning parameter (higher = denser segment peaks).
  - `--target-segments`: Target approximate segment count if known.

### 4. Dataset Visualizer & QC (`tools/visualize_dataset.py`)
- Draws colored bounding boxes (Orange for Head, Green for Body Segments, Magenta for Telson).
- Orders segments sequentially and renders index badges `[1, 2, ..., N]`.
- Displays top statistical dashboard banner with total segment count.

### 5. Segment Counting & Gap Detection Engine (`tools/count_segments.py`)
Runs YOLO inference and produces the final scientific report:
- **Head-to-tail spatial ordering**: Automatically traces the spine curve from anterior to posterior.
- **Occlusion & Gap Analysis**: Analyzes inter-segment distance. If a segment is occluded or missed, it flags the gap and estimates the missing count.
- **Report Image**: Outputs an annotated graphic report showing detected rings and overall stats.

---

## Training YOLO

Run `train_model.py` with custom hyperparameters:
```bash
python train_model.py \
    --data data.yaml \
    --model yolov8n-obb.pt \
    --epochs 50 \
    --batch 8 \
    --imgsz 640
```

Augmentations tailored for dense invertebrates:
- `degrees=180.0`: Rotational invariance (millipedes can crawl in any orientation).
- `flipud=0.5`, `fliplr=0.5`: Horizontal and vertical reflection.
- `scale=0.3`: Scale jittering.
- `mosaic=0.5`: Multi-image composition.

---

## Classes Specification

| Class ID | Name | Description |
|---|---|---|
| `0` | `segment` | Intermediate body ring / diplosegment (annulus) |
| `1` | `head` | Anterior cranial capsule and collum |
| `2` | `telson` | Posterior pre-anal ring, anal valves, and tail tip |

Total Rings in a Millipede = $\text{Head} (1) + \text{Body Segments} (N) + \text{Telson} (1) = N + 2$.
