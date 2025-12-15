from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

# from tvm.auto_scheduler.feature import get_per_store_features_from_measure_pairs

# 시드 고정
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class FeatureDataset:
    """SegmentDataLoader 패턴을 따르는 데이터셋"""
    def __init__(self, segment_sizes, features, batch_size, device, shuffle=False):
        self.device = device
        self.shuffle = shuffle
        self.number = len(segment_sizes)  # 샘플 개수
        self.batch_size = batch_size
        
        self.segment_sizes = torch.tensor(segment_sizes, dtype=torch.int32)
        self.features = torch.tensor(features, dtype=torch.float32)
        
        self.normalize()
        
        self.feature_offsets = (
            torch.cumsum(self.segment_sizes, 0, dtype=torch.int32) - self.segment_sizes
        ).cpu().numpy()
        
        self.iter_order = self.pointer = None
        self.rng = np.random.RandomState(SEED)
    
    def normalize(self, norm_vector=None):
        if norm_vector is None:
            norm_vector = torch.ones((self.features.shape[1],))
            for i in range(self.features.shape[1]):
                max_val = self.features[:, i].max().item()
                if max_val > 0:
                    norm_vector[i] = max_val
        self.features /= norm_vector
        return norm_vector
    

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.from_numpy(self.rng.permutation(self.number))
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0
        return self
    
    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration
        
        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)
    
    def _fetch_indices(self, indices):
        segment_sizes = self.segment_sizes[indices]
        
        feature_offsets = self.feature_offsets[indices]
        feature_indices = np.empty((segment_sizes.sum().item(),), dtype=np.int32)
        ct = 0
        for offset, seg_size in zip(feature_offsets, segment_sizes.numpy()):
            feature_indices[ct: ct + seg_size] = np.arange(offset, offset + seg_size, 1)
            ct += seg_size
        
        features = self.features[feature_indices]
        # VAE용이므로 labels 대신 dummy tensor 반환
        return (x.to(self.device) for x in (segment_sizes, features, torch.zeros(len(indices))))
    
    def __len__(self):
        return self.number


class SegmentRegressionDataset:
    """SegmentDataLoader 패턴을 따르는 데이터셋"""
    def __init__(self, segment_sizes, features, labels, batch_size, device, shuffle=False):
        self.device = device
        self.shuffle = shuffle
        self.number = len(labels)
        self.batch_size = batch_size
        
        self.segment_sizes = torch.tensor(segment_sizes, dtype=torch.int32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.features = torch.tensor(features, dtype=torch.float32)
        
        self.normalize()
        
        self.feature_offsets = (
            torch.cumsum(self.segment_sizes, 0, dtype=torch.int32) - self.segment_sizes
        ).cpu().numpy()
        
        self.iter_order = self.pointer = None
        self.rng = np.random.RandomState(SEED)
    
    def normalize(self, norm_vector=None):
        if norm_vector is None:
            norm_vector = torch.ones((self.features.shape[1],))
            for i in range(self.features.shape[1]):
                max_val = self.features[:, i].max().item()
                if max_val > 0:
                    norm_vector[i] = max_val
        self.features /= norm_vector
        return norm_vector
    
    def set_results(self, used_indices, new_labels):
        # - log값으로 설정
        new_labels = -np.log(new_labels)
        self.labels[used_indices] = torch.tensor(new_labels, dtype=torch.float32).to(self.device)
    

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.from_numpy(self.rng.permutation(self.number))
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0
        return self
    
    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration
        
        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)
    
    def _fetch_indices(self, indices):
        segment_sizes = self.segment_sizes[indices]
        
        feature_offsets = self.feature_offsets[indices]
        feature_indices = np.empty((segment_sizes.sum().item(),), dtype=np.int32)
        ct = 0
        for offset, seg_size in zip(feature_offsets, segment_sizes.numpy()):
            feature_indices[ct: ct + seg_size] = np.arange(offset, offset + seg_size, 1)
            ct += seg_size
        
        features = self.features[feature_indices]
        labels = self.labels[indices]
        return (x.to(self.device) for x in (segment_sizes, features, labels))
    
    def __len__(self):
        return self.number




def load_dataset(features, results, type='regression', test_size=0.0, device='cuda'):
    # features가 3D 배열인 경우 (N, max_n_buf, feature_len)
    if isinstance(features, np.ndarray) and features.ndim == 3:
        n_samples, max_n_buf, feature_len = features.shape
        feature_list = []
        costs = []
        segment_sizes = []
        for i in range(n_samples):
            feature = features[i]  # (max_n_buf, feature_len)
            # 유효한 row만 선택 (all-zero row 제외)
            valid_mask = ~np.all(feature == 0, axis=1)
            valid_feature = feature[valid_mask]
            if valid_feature.shape[0] == 0:
                continue
            feature_list.append(valid_feature)
            cost = np.mean([cost.value for cost in results[i].costs])
            costs.append(cost)
            segment_sizes.append(valid_feature.shape[0])
    else:
        # 기존 로직: features가 리스트인 경우
        feature_list = []
        costs = []
        segment_sizes = []
        for feature, result in zip(features, results):
            # 빈 feature 또는 1차원 feature 스킵
            if feature.ndim != 2 or feature.shape[0] == 0:
                continue
            feature_list.append(feature)
            cost = np.mean([cost.value for cost in result.costs])
            costs.append(cost)
            segment_sizes.append(feature.shape[0])


    features_array = np.array(feature_list, dtype=object)
    costs = np.array(costs, dtype=np.float32)
    segment_sizes = np.array(segment_sizes, dtype=np.int32)


    # Train/Val 분할
    n_samples = len(costs)
    indices = np.arange(n_samples)
    train_indices, val_indices = train_test_split(indices, test_size=test_size, shuffle=False)

    train_segment_sizes = segment_sizes[train_indices]
    val_segment_sizes = segment_sizes[val_indices]
    train_labels = costs[train_indices]
    val_labels = costs[val_indices]

    train_feature_list = [features_array[i] for i in train_indices]
    val_feature_list = [features_array[i] for i in val_indices]

    train_flatten_features = np.concatenate(train_feature_list, axis=0).astype(np.float32)
    val_flatten_features = np.concatenate(val_feature_list, axis=0).astype(np.float32)

    if type == 'vae':
        train_loader = FeatureDataset(
            train_segment_sizes, train_flatten_features,
            batch_size=256, device=device, shuffle=True
        )
        val_loader = FeatureDataset(
            val_segment_sizes, val_flatten_features,
            batch_size=256, device=device, shuffle=False
        )
        return train_loader, val_loader, features_array

    else:  # regression
        print(f"훈련 샘플 수: {len(train_labels)}, 검증 샘플 수: {len(val_labels)}")
        train_loader = SegmentRegressionDataset(
            train_segment_sizes, train_flatten_features, train_labels,
            batch_size=256, shuffle=False
        )
        val_loader = SegmentRegressionDataset(
            val_segment_sizes, val_flatten_features, val_labels,
            batch_size=256, shuffle=False
        )
        return train_loader, val_loader
