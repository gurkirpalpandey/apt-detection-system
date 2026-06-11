"""
train.py — Run this ONCE to train and save all ML models before starting the app.
Usage:  python train.py
        python train.py --data ./cicids_data   (if you have CICIDS CSV files)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.apt_model import train_all

def main():
    parser = argparse.ArgumentParser(description='Train APT Detection Models')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to directory containing CICIDS CSV files. '
                             'If not provided, synthetic data will be used.')
    parser.add_argument('--models', type=str, default='models',
                        help='Directory to save trained models (default: ./models)')
    parser.add_argument('--samples', type=int, default=8000,
                        help='Number of synthetic samples if no real data (default: 8000)')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║   APT Detection System — Model Trainer       ║")
    print("║   Amity University Punjab                    ║")
    print("╚══════════════════════════════════════════════╝\n")

    if args.data:
        if not os.path.isdir(args.data):
            print(f"[ERROR] Data directory not found: {args.data}")
            sys.exit(1)
        print(f"[INFO] Using real CICIDS data from: {args.data}")
    else:
        print(f"[INFO] No data directory given → using synthetic data ({args.samples} samples)")
        print("[INFO] To use real data: python train.py --data ./cicids_data\n")

    results = train_all(data_dir=args.data, model_save_dir=args.models)

    print("\n╔══════════════════════════════════════════════╗")
    print("║          TRAINING SUMMARY                    ║")
    print("╠══════════════════════════════════════════════╣")
    for r in results:
        print(f"║  {r['name']:<22} Acc: {r['accuracy']*100:5.1f}%  F1: {r['f1']*100:5.1f}%  ║")
    print("╚══════════════════════════════════════════════╝")
    print("\n✅ Training complete. You can now run: python app.py")


if __name__ == '__main__':
    main()
