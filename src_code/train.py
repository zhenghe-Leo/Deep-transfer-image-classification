import argparse
import json
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import DogCatDataset
from model import build_model
from utils import ensure_dir, save_confusion_matrix, save_training_curves, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        pred = outputs.argmax(1)
        total_loss += loss.item() * len(labels)
        correct += (pred == labels).sum().item()
        total += len(labels)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


def run_training(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform_full = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform_none = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform_flip = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.augmentation == "none":
        train_transform = train_transform_none
    elif args.augmentation == "flip_only":
        train_transform = train_transform_flip
    else:
        train_transform = train_transform_full

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_ds = DogCatDataset(train_dir, transform=train_transform)
    val_ds = DogCatDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ensure_dir(os.path.dirname(args.save_path) or ".")
    ensure_dir(args.log_dir)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    start = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    elapsed_min = (time.time() - start) / 60.0

    with open(os.path.join(args.log_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.log_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"best_val_acc={best_val_acc:.6f}\n")
        f.write(f"train_time_min={elapsed_min:.2f}\n")

    # 用最佳模型重新评估以便导出报告/混淆矩阵
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    report_txt = classification_report(y_true, y_pred, target_names=["cat", "dog"], digits=4)
    with open(os.path.join(args.log_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    save_training_curves(history, os.path.join(args.log_dir, "training_curve.png"), title_prefix=args.model)
    save_confusion_matrix(
        y_true,
        y_pred,
        labels=["cat", "dog"],
        save_path=os.path.join(args.log_dir, "confusion_matrix.png"),
        title=f"{args.model} Validation Confusion Matrix",
    )

    print(f"Best val acc: {best_val_acc:.6f}")
    print(f"Saved checkpoint to: {args.save_path}")
    print(f"Logs saved to: {args.log_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50_pretrained",
                        choices=["simplecnn", "resnet50_scratch", "resnet50_pretrained"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_path", type=str, default="checkpoints/resnet50_pretrained_best.pth")
    parser.add_argument("--log_dir", type=str, default="experiments/exp3_resnet50_pretrained")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--augmentation", type=str, default="full",
                        choices=["none", "flip_only", "full"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
