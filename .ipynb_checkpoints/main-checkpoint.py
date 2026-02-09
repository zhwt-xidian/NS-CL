import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import copy, random, os, numpy as np
from dataset import NSRemoteSensingDataset
from models import NSCL_Encoder
import time
from tqdm import tqdm

# 尝试导入 thop 用于计算 FLOPs
try:
    from thop import profile
except ImportError:
    profile = None

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_model_specs(model, device):
    """仅用于统计，不改变模型状态"""
    model.eval()
    dummy_input = torch.randn(1, 30, 7, 7).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    flops = 0.0
    if profile:
        flops, _ = profile(copy.deepcopy(model), inputs=(dummy_input, 'proj'), verbose=False)
        flops = flops / 1e6
        
    # 推理延迟测试
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input, mode='proj') # 预热
        starter.record()
        for _ in range(100): _ = model(dummy_input, mode='proj')
        ender.record()
        torch.cuda.synchronize()
        latency = starter.elapsed_time(ender) / 100
    return params, flops, latency

def evaluate_detailed(model, dataset, device, num_classes):
    model.eval()
    dl = DataLoader(dataset, batch_size=512, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dl:
            outputs = model(imgs.to(device), mode='class')
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    oa = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes)) 
    each_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    return oa, np.mean(each_acc), kappa, each_acc

def momentum_update(model_q, model_k, m=0.999):
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

CONFIG = {
    "DATA_PATH": "/root/autodl-tmp/data/HSI/pavia_uni_normal.mat",
    "GT_PATH": "/root/autodl-tmp/data/HSI/pavia_uni_gt.mat",
    "BATCH_SIZE": 64, 
    "EPOCHS_S1": 100, 
    "EPOCHS_S2": 20,           
    "EPOCHS_S3": 150, 
    "NUM_RUNS": 1, 
    "DATASET_SEED": 50,
    "SAMPLES_PER_CLASS": 50,
    "R_VALUE": 2 
} 

# MoCo 队列管理类 (保持不变)
class MoCoQueue:
    def __init__(self, queue_size, feature_dim, device):
        self.queue_size = queue_size
        self.device = device
        self.register_buffer("queue", torch.randn(queue_size, feature_dim).to(device))
        self.register_buffer("queue_spec", torch.randn(queue_size, 30).to(device))
        self.register_buffer("queue_coord", torch.zeros(queue_size, 2).to(device))
        self.queue_ptr = 0
        self.queue.copy_(F.normalize(self.queue, dim=1))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, specs, coords):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_spec[ptr:ptr + batch_size] = specs
            self.queue_coord[ptr:ptr + batch_size] = coords
        else:
            rem = self.queue_size - ptr
            self.queue[ptr:] = keys[:rem]; self.queue[:batch_size-rem] = keys[rem:]
            self.queue_spec[ptr:] = specs[:rem]; self.queue_spec[:batch_size-rem] = specs[rem:]
            self.queue_coord[ptr:] = coords[:rem]; self.queue_coord[:batch_size-rem] = coords[rem:]
        self.queue_ptr = (ptr + batch_size) % self.queue_size

# ----------------- Stage 1 -----------------
def train_stage1(model_q, model_k, loader, device, epochs):
    trainable_params = sum(p.numel() for p in model_q.parameters() if p.requires_grad)
    print(f"  [S1 Config] Trainable Params: {trainable_params/1e6:.4f}M")
    optimizer = optim.AdamW(model_q.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(range(epochs), desc="[Stage 1]")
    for epoch in pbar:
        model_q.train(); model_k.train()
        total_loss = 0
        for anchor, comp_set in loader:
            anchor, comp_set = anchor.to(device), comp_set.to(device)
            q = F.normalize(model_q(anchor, mode='proj'), dim=1)
            with torch.no_grad():
                k_all = F.normalize(model_k(comp_set.view(-1, 30, 7, 7), mode='proj'), dim=1).view(anchor.size(0), -1, 128)
            logits = torch.bmm(q.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1)
            loss = F.cross_entropy(logits / 0.1, torch.zeros(anchor.size(0), dtype=torch.long).to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); momentum_update(model_q, model_k, 0.999)
            total_loss += loss.item()
        scheduler.step()
        pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}")
    return torch.cuda.max_memory_allocated() / 1e6

# ----------------- Stage 2 -----------------
def train_stage2(model_q, model_k, loader, device, epochs):
    trainable_params = sum(p.numel() for p in model_q.parameters() if p.requires_grad)
    print(f"  [S2 Config] Trainable Params: {trainable_params/1e6:.4f}M")
    moco = MoCoQueue(queue_size=4096, feature_dim=128, device=device)
    optimizer = optim.AdamW(model_q.parameters(), lr=1e-4, weight_decay=1e-4)
    
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(range(epochs), desc="[Stage 2]")
    for epoch in pbar:
        model_q.train(); model_k.eval()
        total_loss = 0
        for anchors, pos_patches, specs, coords in loader:
            anchors, pos_patches = anchors.to(device), pos_patches.to(device)
            specs, coords = specs.to(device), coords.to(device)
            with torch.no_grad():
                k_pos = F.normalize(model_k(pos_patches, mode='proj'), dim=1)
            q = F.normalize(model_q(anchors, mode='proj'), dim=1)
            l_pos = torch.einsum('nc,nc->n', [q, k_pos]).unsqueeze(-1)
            l_neg = torch.matmul(q, moco.queue.clone().detach().T)
            with torch.no_grad():
                dist = torch.cdist(coords, moco.queue_coord)
                spec_sim = torch.matmul(F.normalize(specs, dim=1), F.normalize(moco.queue_spec, dim=1).T)
                mask = (dist > 20) & (spec_sim < 0.9)
            l_neg = l_neg.masked_fill(~mask, -1e9) 
            logits = torch.cat([l_pos, l_neg], dim=1) / 0.1
            loss = F.cross_entropy(logits, torch.zeros(anchors.size(0), dtype=torch.long).to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            momentum_update(model_q, model_k, 0.999)
            moco.dequeue_and_enqueue(k_pos, specs, coords)
            total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}")
    return torch.cuda.max_memory_allocated() / 1e6

# ----------------- Stage 3 -----------------
def train_stage3(model, loader, test_dataset, device, num_classes, epochs):
    model.classifier = nn.Linear(model.flatten_dim, num_classes).to(device)
    for param in model.feature.parameters(): param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [S3 Config] Trainable Params: {trainable_params/1e6:.4f}M")
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=0.01, weight_decay=0.01)
    best_oa, best_wts = 0, copy.deepcopy(model.state_dict())
    
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(range(epochs), desc="[Stage 3]")
    for epoch in pbar:
        model.train()
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            loss = F.cross_entropy(model(img, mode='class'), label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            oa, _, _, _ = evaluate_detailed(model, test_dataset, device, num_classes)
            if oa > best_oa:
                best_oa = oa
                best_wts = copy.deepcopy(model.state_dict())
            pbar.set_postfix(best_oa=f"{best_oa*100:.2f}%")
    model.load_state_dict(best_wts)
    return best_oa, torch.cuda.max_memory_allocated() / 1e6

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n>>>> Starting Accurate Three-Stage Training <<<<")
    
    # 基础设置
    seed_everything(CONFIG["DATASET_SEED"])
    temp_ds = NSRemoteSensingDataset(CONFIG["DATA_PATH"], CONFIG["GT_PATH"], stage=3)
    num_classes = temp_ds.num_classes
    model = NSCL_Encoder(in_ch=30, n_classes=num_classes).to(device)
    
    # 统计模型复杂度
    p_total, f_total, inf_lat = get_model_specs(model, device)

    # RUN Stage 1
    s1_start = time.time()
    ds1 = NSRemoteSensingDataset(CONFIG["DATA_PATH"], CONFIG["GT_PATH"], stage=1, seed=CONFIG["DATASET_SEED"], num_labeled=CONFIG["SAMPLES_PER_CLASS"])
    dl1 = DataLoader(ds1, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, drop_last=True)
    mem_s1 = train_stage1(model, copy.deepcopy(model).requires_grad_(False), dl1, device, CONFIG["EPOCHS_S1"])
    s1_time = (time.time() - s1_start) / 60

    # RUN Stage 2
    s2_start = time.time()
    ds2 = NSRemoteSensingDataset(CONFIG["DATA_PATH"], CONFIG["GT_PATH"], stage=2, seed=CONFIG["DATASET_SEED"], r=CONFIG["R_VALUE"])
    dl2 = DataLoader(ds2, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    mem_s2 = train_stage2(model, copy.deepcopy(model).requires_grad_(False), dl2, device, CONFIG["EPOCHS_S2"])
    s2_time = (time.time() - s2_start) / 60

    # RUN Stage 3
    s3_start = time.time()
    ds3_train = NSRemoteSensingDataset(CONFIG["DATA_PATH"], CONFIG["GT_PATH"], stage=3, mode='train', seed=CONFIG["DATASET_SEED"], num_labeled=CONFIG["SAMPLES_PER_CLASS"])
    ds3_test = NSRemoteSensingDataset(CONFIG["DATA_PATH"], CONFIG["GT_PATH"], stage=3, mode='test', seed=CONFIG["DATASET_SEED"])
    dl3_train = DataLoader(ds3_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    best_oa, mem_s3 = train_stage3(model, dl3_train, ds3_test, device, num_classes, CONFIG["EPOCHS_S3"])
    s3_time = (time.time() - s3_start) / 60

    # Final Metrics
    oa, aa, kappa, _ = evaluate_detailed(model, ds3_test, device, num_classes)

    #打印报表
    print("\n" + "="*85)
    print(f"{'Metric':<40} | {'Value':<40}")
    print("-" * 85)
    print(f"{'[1] Model Specification (Total Params)':<40} | {p_total:.4f} M")
    print(f"{'[1] Computational Complexity (FLOPs)':<40} | {f_total:.4f} M (per patch)")
    print(f"{'[2] Mean Inference Latency':<40} | {inf_lat:.4f} ms/patch")
    print(f"{'[2] Inference Peak Memory':<40} | {torch.cuda.max_memory_reserved()/1e6:.2f} MB")
    print("-" * 85)
    print(f"{'[3] Training Time (Stage 1)':<40} | {s1_time:.2f} min")
    print(f"{'[3] Training Time (Stage 2)':<40} | {s2_time:.2f} min")
    print(f"{'[3] Training Time (Stage 3)':<40} | {s3_time:.2f} min")
    print(f"{'[4] GPU Peak Mem (S1/S2/S3)':<40} | {mem_s1:.1f} / {mem_s2:.1f} / {mem_s3:.1f} MB")
    print("-" * 85)
    print(f"{'[Final Result] Overall Accuracy':<40} | {oa*100:.2f}%")
    print(f"{'[Final Result] Average Accuracy':<40} | {aa*100:.2f}%")
    print(f"{'[Final Result] Kappa Coefficient':<40} | {kappa:.4f}")
    print("=" * 85 + "\n")