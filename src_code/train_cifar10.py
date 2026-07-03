import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

from utils import ensure_dir, save_confusion_matrix, save_training_curves, set_seed


def make_imbalanced_subset(dataset, minority_classes=(0, 1, 2, 3, 4), keep_ratio=0.1, seed=42):
    rng = random.Random(seed)
    cls_to_indices = {i: [] for i in range(10)}
    for idx, target in enumerate(dataset.targets):
        cls_to_indices[target].append(idx)

    selected = []
    for cls, indices in cls_to_indices.items():
        if cls in minority_classes:
            k = max(1, int(len(indices) * keep_ratio))
            indices = indices.copy()
            rng.shuffle(indices)
            selected.extend(indices[:k])
        else:
            selected.extend(indices)

    selected.sort()
    return Subset(dataset, selected)


def build_model(num_classes=10):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    except AttributeError:
        weights = "IMAGENET1K_V1"
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


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
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
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


def per_class_accuracy(y_true, y_pred, num_classes=10):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cls_acc = []
    for i in range(num_classes):
        denom = cm[i].sum()
        acc = (cm[i, i] / denom) if denom > 0 else 0.0
        cls_acc.append(acc)
    return cls_acc


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.log_dir)
    ensure_dir(os.path.dirname(args.save_path) or ".")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set_full = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)

    if args.imbalanced:
        train_set = make_imbalanced_subset(train_set_full, seed=args.seed)
    else:
        train_set = train_set_full

    sampler = None
    class_weights_tensor = None

    if args.imbalanced and args.method == "weighted_sampler":
        if isinstance(train_set, Subset):
            targets = np.array(train_set_full.targets)[train_set.indices]
        else:
            targets = np.array(train_set.targets)

        class_counts = np.bincount(targets, minlength=10)
        weights_per_class = 1.0 / np.maximum(class_counts, 1)
        sample_weights = weights_per_class[targets]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    if args.imbalanced and args.method == "weighted_loss":
        if isinstance(train_set, Subset):
            targets = np.array(train_set_full.targets)[train_set.indices]
        else:
            targets = np.array(train_set.targets)

        class_counts = np.bincount(targets, minlength=10)
        total_samples = class_counts.sum()
        class_weights = total_samples / (10 * np.maximum(class_counts, 1))
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    start = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(te_loss)
        history["val_acc"].append(te_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
            f"Test Loss={te_loss:.4f} Acc={te_acc:.4f}"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), args.save_path)

    elapsed_min = (time.time() - start) / 60.0

    with open(os.path.join(args.log_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.log_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_test_acc={best_acc:.6f}\n")
        f.write(f"train_time_min={elapsed_min:.2f}\n")
        f.write(f"imbalanced={args.imbalanced}\n")
        f.write(f"method={args.method}\n")

    model.load_state_dict(torch.load(args.save_path, map_location=device))
    _, overall_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    save_training_curves(history, os.path.join(args.log_dir, "training_curve.png"), title_prefix="CIFAR10")
    save_confusion_matrix(
        y_true,
        y_pred,
        labels=[str(i) for i in range(10)],
        save_path=os.path.join(args.log_dir, "confusion_matrix.png"),
        title="CIFAR10 Confusion Matrix",
    )

    cls_acc = per_class_accuracy(y_true, y_pred, num_classes=10)
    with open(os.path.join(args.log_dir, "per_class_accuracy.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy"])
        for i, a in enumerate(cls_acc):
            writer.writerow([i, f"{a:.6f}"])

    with open(os.path.join(args.log_dir, "overall_accuracy.txt"), "w", encoding="utf-8") as f:
        f.write(f"{overall_acc:.6f}\n")

    print(f"Best test acc: {best_acc:.6f}")
    print(f"Saved checkpoint to: {args.save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--imbalanced", action="store_true")
    parser.add_argument("--method", type=str, default="no_handling",
                        choices=["no_handling", "weighted_sampler", "weighted_loss"])

    parser.add_argument("--save_path", type=str, default="checkpoints/cifar10_best.pth")
    parser.add_argument("--log_dir", type=str, default="experiments/exp6_cifar10")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
