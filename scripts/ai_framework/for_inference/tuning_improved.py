import os
import tvm
from tvm import auto_scheduler
import time
import random
import torch
import numpy as np



def select_features(model, features, used_indices, fea_norm_vec, num_select=64, T_mc=20, 
                    w_cost=0.5, w_unc=0.3, w_dist=0.2, topk_factor=5):
    """
    Active learning 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        features: 전체 feature 리스트 (각 element는 [seg_size, input_dim] 형태)
        used_indices: 이미 측정된 인덱스 집합 (set)
        num_select: 선택할 샘플 수
        T_mc: MC Dropout 샘플 수
        w_cost: 예측 cost 가중치 (낮을수록 좋음)
        w_unc: epistemic 불확실성 가중치 (클수록 좋음)
        w_dist: latent 거리 가중치 (클수록 덜 탐색된 영역)
        topk_factor: 상위 후보 선택 배수
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 리스트
    """

    device = next(model.parameters()).device
    if fea_norm_vec is not None:
        fea_norm_vec = fea_norm_vec.to(device)

    def normalize_feature(f):
        if fea_norm_vec is None:
            return f
        # f: (seg_len, input_dim)
        if isinstance(f, np.ndarray):
            return f / fea_norm_vec.cpu().numpy()
        else:
            return f / fea_norm_vec.to(f.device)

    all_indices = list(range(len(features)))
    
    # labeled / unlabeled 분리
    labeled_idx = sorted(list(used_indices))
    unlabeled_idx = [i for i in all_indices if i not in used_indices]
    
    if len(unlabeled_idx) == 0:
        return []
    
    if len(unlabeled_idx) <= num_select:
        return unlabeled_idx
    
    # ========== 1. Latent 임베딩 계산 ==========
    def get_latent_embeddings(indices):
        """주어진 인덱스들에 대해 latent embedding 계산"""
        if len(indices) == 0:
            return None
        
        # features를 하나의 tensor로 합치기
        feats_list = [normalize_feature(features[i]) for i in indices]
        segment_sizes = np.array([f.shape[0] for f in feats_list], dtype=np.int32)
        
        # segment_sizes와 features를 tensor로 변환
        segment_sizes_t = torch.tensor(segment_sizes, dtype=torch.long, device=device)
        
        # features를 float32로 확실하게 변환
        features_concat = np.vstack([np.asarray(f, dtype=np.float32) for f in feats_list])
        features_t = torch.tensor(features_concat, dtype=torch.float32, device=device)
        
        # Latent embedding 계산
        z = model.get_latent_embedding(segment_sizes_t, features_t, use_mean=True)
        return z
    
    # labeled / unlabeled에 대해 latent 임베딩 계산
    z_labeled = get_latent_embeddings(labeled_idx)  # [N_l, D] or None
    z_unlabeled = get_latent_embeddings(unlabeled_idx)  # [N_u, D]
    
    # ========== 2. MC Dropout 기반 불확실성 추정 ==========
    feats_unlabeled = [features[i] for i in unlabeled_idx]
    seg_sizes_unlabeled = np.array([f.shape[0] for f in feats_unlabeled], dtype=np.int32)
    seg_sizes_t = torch.tensor(seg_sizes_unlabeled, dtype=torch.long, device=device)
    
    # features를 float32로 확실하게 변환
    feats_concat_unlabeled = np.vstack([np.asarray(f, dtype=np.float32) for f in feats_unlabeled])
    feats_t_unlabeled = torch.tensor(feats_concat_unlabeled, dtype=torch.float32, device=device)
    
    # MC prediction
    mu_u, var_u = model.mc_predict(seg_sizes_t, feats_t_unlabeled, T=T_mc, use_mean=True)
    mu_u = mu_u.cpu().numpy()  # [N_u]
    var_u = var_u.cpu().numpy()  # [N_u]
    
    # ========== 3. Latent 거리 계산 ==========
    z_unlabeled_np = z_unlabeled.cpu().numpy()  # [N_u, D]
    
    if z_labeled is not None and len(labeled_idx) > 0:
        z_labeled_np = z_labeled.cpu().numpy()  # [N_l, D]
        
        # Pairwise L2 거리 계산: [N_u, N_l]
        # 효율적인 계산: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        z_u_sq = np.sum(z_unlabeled_np ** 2, axis=1, keepdims=True)  # [N_u, 1]
        z_l_sq = np.sum(z_labeled_np ** 2, axis=1, keepdims=True).T  # [1, N_l]
        dist_sq = z_u_sq + z_l_sq - 2 * np.dot(z_unlabeled_np, z_labeled_np.T)  # [N_u, N_l]
        dist_sq = np.maximum(dist_sq, 0)  # numerical stability
        
        # 각 unlabeled 샘플에서 가장 가까운 labeled까지의 최소 거리
        d_min_u = np.sqrt(np.min(dist_sq, axis=1))  # [N_u]
    else:
        # labeled가 없으면 모든 거리를 1로 설정
        d_min_u = np.ones(len(unlabeled_idx))
    
    # ========== 4. Acquisition Score 계산 ==========
    eps = 1e-8
    
    def normalize(x):
        """0~1 범위로 정규화"""
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min + eps)
    
    # cost는 낮을수록 좋음 -> -mu_u를 정규화
    norm_cost = normalize(-mu_u)
    # 불확실성은 클수록 좋음
    norm_unc = normalize(var_u)
    # 거리는 클수록 좋음 (덜 탐색된 영역)
    norm_dist = normalize(d_min_u)
    
    # Acquisition score
    scores = w_cost * norm_cost + w_unc * norm_unc + w_dist * norm_dist
    
    # ========== 5. 상위 K개 + Diversity 선택 ==========
    K = min(num_select * topk_factor, len(unlabeled_idx))
    
    # score 상위 K개 인덱스 (unlabeled_idx 내 인덱스)
    topk_local_idx = np.argsort(scores)[::-1][:K]
    
    # Greedy farthest-point sampling
    z_topk = z_unlabeled_np[topk_local_idx]  # [K, D]
    scores_topk = scores[topk_local_idx]
    
    selected_local = []  # topk_local_idx 내에서의 인덱스
    
    # 첫 번째: score가 가장 높은 샘플
    first_idx = 0  # 이미 정렬되어 있음
    selected_local.append(first_idx)
    
    # 이후: farthest-point sampling
    while len(selected_local) < num_select and len(selected_local) < K:
        selected_z = z_topk[selected_local]  # [selected, D]
        
        # 남은 후보들
        remaining = [i for i in range(K) if i not in selected_local]
        if len(remaining) == 0:
            break
        
        # 각 remaining에 대해 selected와의 최소 거리 계산
        best_idx = None
        best_min_dist = -1
        
        for r_idx in remaining:
            r_z = z_topk[r_idx]
            # selected와의 거리
            dists = np.sqrt(np.sum((selected_z - r_z) ** 2, axis=1))
            min_dist = np.min(dists)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = r_idx
        
        if best_idx is not None:
            selected_local.append(best_idx)
    
    # 선택된 인덱스를 원래 pool 인덱스로 변환
    selected_global_idx = [unlabeled_idx[topk_local_idx[i]] for i in selected_local]

    print("Feature 선택 완료")
    
    return selected_global_idx