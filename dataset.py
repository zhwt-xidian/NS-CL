 
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn.decomposition import PCA
from tqdm import tqdm # 方便查看初始化进度

class NSRemoteSensingDataset(Dataset):
    def __init__(self, data_path, gt_path, stage=1, patch_size=7, r=2, num_labeled=100, seed=42, mode='train'):
        # 1. 自动识别数据 Key
        mat_data = sio.loadmat(data_path)
        data_keys = [k for k in mat_data if not k.startswith('_')]
        data_key = [k for k in data_keys if 'gt' not in k.lower() and 'label' not in k.lower()][0]
        self.raw_data = mat_data[data_key].astype(np.float32)
        
        # 2. 自动识别 GT Key
        mat_gt = sio.loadmat(gt_path)
        gt_keys = [k for k in mat_gt if not k.startswith('_')]
        gt_candidates = [k for k in gt_keys if any(x in k.lower() for x in ['gt', 'ts', 'label'])]
        gt_key = gt_candidates[0] if gt_candidates else gt_keys[0]
        self.gt = np.squeeze(mat_gt[gt_key]).astype(np.int32)
        
        # 3. PCA 降维
        h, w, c = self.raw_data.shape
        data_reshaped = self.raw_data.reshape(-1, c)
        data_norm = (data_reshaped - np.mean(data_reshaped, axis=0)) / (np.std(data_reshaped, axis=0) + 1e-8)
        self.pca = PCA(n_components=30)
        self.data = self.pca.fit_transform(data_norm).reshape(h, w, 30)
        self.data = (self.data - np.mean(self.data)) / (np.std(self.data) + 1e-8)

        self.stage, self.p, self.r, self.mode = stage, patch_size, r, mode
        self.margin = patch_size // 2
        self.padded_data = np.pad(self.data, ((self.margin, self.margin), (self.margin, self.margin), (0, 0)), mode='edge')

        # 4. 样本划分
        all_indices = np.argwhere(self.gt > 0)
        labels_all = self.gt[all_indices[:,0], all_indices[:,1]] - 1
        self.unique_labels = np.unique(labels_all)
        self.num_classes = len(self.unique_labels)
        
        self.class_indices_train = {}
        self.train_pool, self.test_pool, self.unlabeled_pool = [], [], []
        np.random.seed(seed)
        for c in self.unique_labels:
            c_idx = all_indices[labels_all == c]
            np.random.shuffle(c_idx)
            n_lab = min(len(c_idx), num_labeled)
            self.train_pool.extend(c_idx[:n_lab])
            self.class_indices_train[c] = c_idx[:n_lab]
            remaining = c_idx[n_lab:]
            split = int(len(remaining) * 0.5)
            self.unlabeled_pool.extend(remaining[:split])
            self.test_pool.extend(remaining[split:])
            
        self.train_pool = np.array(self.train_pool)
        self.unlabeled_pool = np.array(self.unlabeled_pool)
        self.test_pool = np.array(self.test_pool)

        # --- 优化核心：Stage 2 预计算最佳邻居 ---
        if self.stage == 2:
            self.best_pos_coords = self._precompute_best_neighbors()

    def _precompute_best_neighbors(self):
        print(f"Pre-computing best neighbors for Stage 2 (r={self.r})...")
        h, w, c = self.data.shape
        # 预归一化用于算余弦相似度
        norm_data = self.data / (np.linalg.norm(self.data, axis=2, keepdims=True) + 1e-8)
        
        best_coords = []
        # 遍历 unlabeled_pool
        for i in tqdm(range(len(self.unlabeled_pool))):
            x, y = self.unlabeled_pool[i]
            
            # 确定邻域范围
            x1, x2 = max(0, x - self.r), min(h, x + self.r + 1)
            y1, y2 = max(0, y - self.r), min(w, y + self.r + 1)
            
            # 提取邻域特征并展平
            neighborhood = norm_data[x1:x2, y1:y2, :] # [rows, cols, 30]
            anchor = norm_data[x, y] # [30]
            
            # 矩阵计算余弦相似度
            sims = np.dot(neighborhood, anchor) # [rows, cols]
            
            # 掩盖中心点自身，防止选到自己
            local_x, local_y = x - x1, y - y1
            sims[local_x, local_y] = -1e9
            
            # 找到相似度最高的坐标
            max_idx = np.argmax(sims)
            rel_x, rel_y = np.unravel_index(max_idx, sims.shape)
            
            best_coords.append((x1 + rel_x, y1 + rel_y))
        return np.array(best_coords)

    def get_patch(self, x, y):
        x_p, y_p = x + self.margin, y + self.margin
        patch = self.padded_data[x_p - self.margin : x_p + self.margin + 1, 
                                 y_p - self.margin : y_p + self.margin + 1, :]
        return torch.from_numpy(patch).permute(2, 0, 1)

    def __getitem__(self, index):
        if self.stage == 1:
            c_anchor = np.random.choice(self.unique_labels)
            pool = self.class_indices_train[c_anchor]
            idx_anchor, idx_pos = pool[np.random.choice(len(pool), 2, replace=True)]
            
            comp_list = [self.get_patch(*idx_pos)]
            for c_neg in self.unique_labels:
                if c_neg == c_anchor: continue
                n_p = self.class_indices_train[c_neg]
                comp_list.append(self.get_patch(*n_p[np.random.randint(len(n_p))]))
            return self.get_patch(*idx_anchor), torch.stack(comp_list)
        
        elif self.stage == 2:
            # 优化后：直接查表，不再有循环
            x, y = self.unlabeled_pool[index]
            nx, ny = self.best_pos_coords[index]
            
            anchor_patch = self.get_patch(x, y)
            pos_patch = self.get_patch(nx, ny)
            anchor_spec = self.data[x, y]
            
            return anchor_patch, pos_patch, torch.tensor(anchor_spec), torch.tensor([x, y], dtype=torch.float)
        else:
            p = self.train_pool if self.mode == 'train' else self.test_pool
            x, y = p[index]
            return self.get_patch(x, y), torch.tensor(int(self.gt[x, y]) - 1, dtype=torch.long)

    def __len__(self):
        if self.stage == 1: return 2000 
        if self.stage == 2: return len(self.unlabeled_pool)
        return len(self.train_pool) if self.mode == 'train' else len(self.test_pool)