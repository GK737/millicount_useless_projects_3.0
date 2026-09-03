"""
Train YOLO on the Millipede Segment/Ring Dataset.
Supports YOLOv8 / YOLO11 for:
- Oriented Bounding Boxes (YOLO-OBB, e.g. yolov8n-obb.pt) - RECOMMENDED
- Standard Bounding Boxes (YOLO-Detect, e.g. yolov8n.pt)
- Instance Segmentation (YOLO-Seg, e.g. yolov8n-seg.pt)
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def train(
    data_yaml="data.yaml",
    model_name="yolov8n-obb.pt",
    epochs=35,
    imgsz=416,
    batch=8,
    device=None,
    project="runs/millipede",
    name="milli_segment_exp",
    export_dir="e:/Useless_Project/Vision_Based_Millicount/models"
):
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at '{yaml_path}'")

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"\n=======================================================")
    print(f" TRAINING MILLIPEDE SEGMENT COUNTER WITH YOLO-OBB")
    print(f"=======================================================")
    print(f"  Model Architecture : {model_name}")
    print(f"  Dataset Config     : {yaml_path}")
    print(f"  Epochs             : {epochs}")
    print(f"  Image Size         : {imgsz}x{imgsz}")
    print(f"  Batch Size         : {batch}")
    print(f"  Compute Device     : {device}")
    print(f"=======================================================\n")

    # Initialize YOLO model
    model = YOLO(model_name)

    # Train with augmentations tailored for articulated invertebrates
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        # Augmentation settings for dense repeating structures
        degrees=180.0,       # Full rotational invariance (millipedes lie at any angle)
        flipud=0.5,          # Vertical flip
        fliplr=0.5,          # Horizontal flip
        scale=0.3,           # Zoom in/out variation
        mosaic=0.5,          # Mosaic augmentation
        verbose=True
    )

    best_weight = Path(project) / name / "weights" / "best.pt"
    print(f"\n[+] Training complete!")
    if best_weight.exists():
        print(f"[+] Best model saved at: {best_weight}")
        
        # Export to Vision_Based_Millicount models folder
        if export_dir:
            out_dir = Path(export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            target_file = out_dir / "millipede_yolov8n_obb.pt"
            import shutil
            shutil.copy2(best_weight, target_file)
            print(f"[+] Model weights copied to: {target_file}")

        print(f"\n[*] To count segments on a new image using this model, run:")
        print(f"    python tools/count_segments.py --image your_millipede.jpg --model {best_weight}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on Millipede Segment Dataset.")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-obb.pt", help="Base model weights (yolov8n-obb.pt, yolov8n.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, 0, 0,1)")
    parser.add_argument("--export", type=str, default="e:/Useless_Project/Vision_Based_Millicount/models", help="Export directory")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        export_dir=args.export
    )
