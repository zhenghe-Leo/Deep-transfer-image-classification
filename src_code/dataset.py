import os
from PIL import Image
from torch.utils.data import Dataset


class DogCatDataset(Dataset):
    """用于 train/val 有标签集: cat=0, dog=1"""

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label, cls in enumerate(["cat", "dog"]):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class DogCatTestDataset(Dataset):
    """用于 test 无标签集，返回图片和 id"""

    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.samples = []

        fnames = [
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        fnames = sorted(fnames, key=lambda x: int(os.path.splitext(x)[0]))

        for fname in fnames:
            img_id = int(os.path.splitext(fname)[0])
            self.samples.append((os.path.join(test_dir, fname), img_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, img_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_id
