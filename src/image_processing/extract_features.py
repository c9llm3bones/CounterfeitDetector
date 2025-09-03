import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import h5py
from torch.cuda.amp import autocast

IMAGE_DIR="./data/ml_ozon_сounterfeit_train_images"      # папка с {ItemID}.png
OUT_H5 = "./data/image_features.h5"       # куда сохранять фичи
BATCH_SIZE = 32                    # можно увеличить/уменьшить
NUM_WORKERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SHORTER_SIDE = 256             # для Resize(256) -> CenterCrop(224)
CROP_SIZE = 224
BACKBONE = "resnet50"              # можно поменять на efficientnet_b3/clip позже
DTYPE = np.float32                 # dtype для features

class ItemImageDataset(Dataset):
    def __init__(self, items: list, image_dir: str, transform=None):
        """
        items: list of tuples (item_id, filename) OR list of filenames where basename is ItemID
        image_dir: root dir
        transform: torchvision transforms
        """
        self.image_dir = Path(image_dir)
        # normalize items into list of (id, path)
        normalized = []
        for it in items:
            if isinstance(it, tuple) and len(it) >= 2:
                item_id, fname = it[0], it[1]
            else:
                p = Path(it)
                fname = p.name
                item_id = p.stem
            normalized.append((item_id, str(self.image_dir / fname)))
        self.items = normalized
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_id, fp = self.items[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return item_id, img

# Получаем список файлов: если у тебя csv с id -> filename, загрузить и собрать список.
# Простой fallback: взять все файлы в папке
image_dir = Path(IMAGE_DIR)
files = [p.name for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}]
items = [(Path(fn).stem, fn) for fn in files]   # (ItemID, filename)
print("Total images:", len(items))

# Трансформы: Resize(256) + CenterCrop(224) — не искажает пропорции
transform = transforms.Compose([
    transforms.Resize(IMG_SHORTER_SIDE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
dataset = ItemImageDataset(items, IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Загружаем resnet50 и отрезаем классификатор
model_full = torchvision.models.resnet50(pretrained=True)
# убрать последний fc, оставить avgpool -> [B,2048,1,1]
backbone = torch.nn.Sequential(*list(model_full.children())[:-1])
backbone.to(DEVICE)
backbone.eval()

# сколько элементов и размер фичи
N = len(dataset)
with torch.no_grad():
    # небольшой прогон одного батча, чтобы убедиться в размерности (без grad)
    item0, img0 = dataset[0]
    tmp = img0.unsqueeze(0).to(DEVICE)
    with autocast():
        feat0 = backbone(tmp)
    feat_shape = feat0.shape  # (1, 2048, 1, 1) обычно
    feat_dim = int(np.prod(feat_shape[1:]))  # 2048
print("features dim:", feat_dim)

# подготовка HDF5 (создаем datasets)
h5f = h5py.File(OUT_H5, "w")
h5_feats = h5f.create_dataset("features", shape=(N, feat_dim), dtype=DTYPE)
# Сохраним ids — попытка привести к int, иначе сохраняем как fixed-length ascii
try:
    ids_np = np.array([int(t[0]) for t in items], dtype=np.int64)
    h5_ids = h5f.create_dataset("ids", data=ids_np, dtype=np.int64)
    ids_stored_as_int = True
except Exception:
    # fallback to bytes
    ids_bytes = np.array([str(t[0]).encode("utf-8") for t in items], dtype='S')
    h5_ids = h5f.create_dataset("ids", data=ids_bytes, dtype='S')
    ids_stored_as_int = False

print("HDF5 created:", OUT_H5)


# Ячейка D: извлечение фичей и запись в HDF5
start_idx = 0
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Feature extraction"):
        item_ids, imgs = batch  # item_ids: list of ids as strings, imgs: tensor [B,C,H,W]
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast():  # ускорение и уменьшение consumption VRAM
            feats = backbone(imgs)  # [B, 2048, 1, 1]
            feats = feats.view(feats.size(0), -1)  # [B, feat_dim]
        feats_cpu = feats.cpu().numpy().astype(DTYPE)
        b = feats_cpu.shape[0]
        h5_feats[start_idx:start_idx+b] = feats_cpu
        start_idx += b

# ensure flush & close
h5f.flush()
h5f.close()
print("Done. Saved features for", start_idx, "images.")

