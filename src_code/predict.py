import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DogCatTestDataset
from model import build_model


@torch.no_grad()
def predict_and_save(model, test_loader, device, save_path="submission.csv"):
    model.eval()
    all_ids, all_preds = [], []

    for imgs, img_ids in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_ids.extend(img_ids.tolist())
        all_preds.extend(preds)

    df = pd.DataFrame({"id": all_ids, "label": all_preds})
    df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} predictions to {save_path}")
    print(df["label"].value_counts().rename({1: "dog", 0: "cat"}))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = DogCatTestDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    predict_and_save(model, test_loader, device, save_path=args.save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50_pretrained",
                        choices=["simplecnn", "resnet50_scratch", "resnet50_pretrained"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--save_path", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
