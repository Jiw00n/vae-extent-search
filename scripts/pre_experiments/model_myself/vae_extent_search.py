#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import re
import time

project_root = "/root/work/tenset"
os.environ["TVM_HOME"] = f"{project_root}"
os.environ["TVM_LIBRARY_PATH"] = f"{project_root}/build"
if f"{project_root}/python" not in sys.path:
    sys.path.insert(0, f"{project_root}/python")
    

sys.path = [p for p in sys.path if not p.startswith(f"{project_root}/build")]
sys.path.append(f"{project_root}/build")
os.environ["LD_LIBRARY_PATH"] = f"{project_root}/build:" + os.environ.get("LD_LIBRARY_PATH", "")


# In[ ]:


import numpy as np
sys.path.append("/root/work/tenset/scripts")
from print_programs import return_all_states
from make_dataset import load_and_register_tasks
from tvm import auto_scheduler
from tvm.auto_scheduler.dataset import Dataset, make_dataset_from_log_file
# json_file = "/root/work/tenset/dataset/measure_records_tenset/k80/([0bcb8746286db050cd088f375c85372d,1,64,64,128,6,6,32,128,1,64,64,32],cuda).json"
json_file = "/root/work/tenset/dataset/measure_records_tenset/k80/([0c9a5ba46ffc5e1a9e5641018527117f,4,7,7,160,1,1,160,960,1,1,1,960,4,7,7,960],cuda).json"
# json_file = "/root/work/tenset/dataset/measure_records_tenset/k80/([3eb184d18885126bd13d564ef260c820,4,16,16,256,6,6,256,256,1,1,1,256,4,16,16,256,4,16,16,256],cuda).json"
# json_file = "/root/work/tenset/dataset/measure_records_tenset/k80/([8c674f26f66543069d1e1c56cda249f9,4,60,60,256,1,1,256,512,1,1,1,512,4,30,30,512],cuda).json"
load_and_register_tasks()
print("Loading dataset from", json_file)


# In[ ]:


states, costs = return_all_states(json_file)
records_raw = list(map(lambda x: str(x).strip(), states))

records = {"schedules": [], "extents": [], "costs": [], "unroll" : [], "all": []}

for rec, cost in zip(records_raw, costs):
    cost = np.array([c.value for c in cost])
    cost = -np.log(np.mean(cost) + 1e-8)
    schedule = rec.split("Placeholder")[-1][2:]
    
    records["schedules"].append(schedule)
    records["costs"].append(cost)


# In[30]:


for a in records["schedules"][:1]:
    print(a)
    print("---------------------------------------------------")


# In[240]:


import re
import numpy as np

def find_common_for_loops(schedules):
    """
    모든 스케줄에서 공통으로 나타나는 (0,1) for문 변수명을 찾음
    """
    common_vars = None
    
    for schedule in schedules:
        lines = schedule.split('\n')
        vars_in_schedule = set()
        
        for line in lines:
            stripped = line.lstrip()
            match = re.match(r'for\s+(\S+)\s+\(0,\s*1\)', stripped)
            if match:
                vars_in_schedule.add(match.group(1))
        
        if common_vars is None:
            common_vars = vars_in_schedule
        else:
            common_vars &= vars_in_schedule  # 교집합
    
    return common_vars if common_vars is not None else set()


def remove_common_for_loops(schedule, common_vars):
    """
    스케줄 코드에서 공통으로 나타나는 (0,1) for문을 제거하고 들여쓰기를 정리
    """
    lines = schedule.split('\n')
    result_lines = []
    
    # 제거할 for문의 인덱스들을 먼저 찾기
    remove_indices = set()
    for_loop_indents = {}  # 제거될 for문의 인덱스 -> 들여쓰기 레벨
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent_level = len(line) - len(stripped)
        
        # (0,1) for문인지 확인
        match = re.match(r'for\s+(\S+)\s+\(0,\s*1\)', stripped)
        if match and match.group(1) in common_vars:
            remove_indices.add(i)
            for_loop_indents[i] = indent_level
    
    # 각 줄에 대해 들여쓰기를 얼마나 줄여야 하는지 계산
    indent_reduction = [0] * len(lines)
    
    for idx in sorted(remove_indices):
        base_indent = for_loop_indents[idx]
        # 이 for문 다음부터 같거나 작은 들여쓰기가 나올 때까지 2칸씩 줄이기
        for j in range(idx + 1, len(lines)):
            if j in remove_indices:
                continue
            line = lines[j]
            stripped = line.lstrip()
            if not stripped:  # 빈 줄
                continue
            current_indent = len(line) - len(stripped)
            
            # 이 for문의 body인 경우 (들여쓰기가 더 큰 경우)
            if current_indent > base_indent:
                indent_reduction[j] += 2
            else:
                # 같거나 작은 들여쓰기 레벨이 나오면 이 for문 블록 종료
                break
    
    # 제거하지 않는 줄들에 대해 들여쓰기를 조정하여 결과 생성
    for i, line in enumerate(lines):
        if i in remove_indices:
            continue
        
        if not line.strip():  # 빈 줄
            result_lines.append(line)
            continue
        
        stripped = line.lstrip()
        original_indent = len(line) - len(stripped)
        new_indent = max(0, original_indent - indent_reduction[i])
        result_lines.append(' ' * new_indent + stripped)
    
    return '\n'.join(result_lines)


common_for_loops = find_common_for_loops(records["schedules"])
print(f"발견된 공통 (0,1) for문 변수: {common_for_loops}")


# 모든 스케줄에 적용
cleaned_schedules = []
records["extents"] = []
records["unroll"] = []
records["all"] = []
for i, schedule in enumerate(records["schedules"]):
    extents = [float(x) for x in re.findall(r'\(0,\s*(\d+)\)', schedule)]

for i, schedule in enumerate(records["schedules"]):
    extents = [float(x) for x in re.findall(r'\(0,\s*(\d+)\)', schedule)]
    unrolls = [float(x) for x in re.findall(r'auto_unroll:\s*(\d+)', schedule)]
    records["extents"].append(extents)
    if unrolls == []:
        unrolls = [0.0]
    records["unroll"].append(unrolls)
    feature = extents+unrolls
    records["all"].append(np.array(feature, dtype=np.float32))
    
    cleaned = remove_common_for_loops(schedule, common_for_loops)
    cleaned_schedules.append(cleaned)
records["cleaned_schedules"] = cleaned_schedules


total_removed = sum(len(orig.split('\n')) - len(clean.split('\n')) 
                    for orig, clean in zip(records['schedules'], cleaned_schedules))
avg_removed = total_removed / len(cleaned_schedules)
print(f"제거된 줄 수: {avg_removed:.1f}")


# In[241]:


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

class FeatureRegressionDataset(Dataset):
    def __init__(self, X, y, feature=None):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X
        self.y = torch.from_numpy(y).float()
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

        self.feature = feature
        if feature is not None:
            if isinstance(feature, np.ndarray):
                self.feature = torch.from_numpy(feature).float()
            else:
                self.feature = feature
            
            if self.feature.ndim == 1:
                self.feature = self.feature.unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.feature is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.feature[idx]


class FeatureDataset(Dataset):
    def __init__(self, X, feature=None):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X
        
        if isinstance(feature, np.ndarray):
            self.feature = torch.from_numpy(feature).float()
        else:
            self.feature = feature
        # feature shape이 (N,)이면 (N,1)로 바꿔주는 게 편할 때가 많음
        if self.feature is not None and self.feature.ndim == 1:
            self.feature = self.feature.unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.feature is None:
            return self.X[idx]
        return self.X[idx], self.feature[idx]


# In[242]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_feature_head(nn.Module):
    def __init__(self, input_dim, feature_dim=None, latent_dim=16, hidden_dim=128):
        """
        input_dim: 2 * D (v_norm + is_zero concat한 차원)
        latent_dim: latent space 차원
        hidden_dim: MLP hidden 크기
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            
            # 출력은 연속값이니까 activation 없이 그대로
        )

        if feature_dim is None:
            self.use_feature = False
        else:
            self.use_feature = True
            self.feature_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),  # features.shape[1]는 feature 차원
            )

    def encode(self, x, use_mean=False):

        h = self.encoder(x)
        mean = self.fc_mu(h)
        if not use_mean:
            logvar = self.fc_logvar(h)
        else:
            return mean
        
        return mean, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def predict_feature(self, z):
        return self.feature_predictor(z)

    def forward(self, x, use_mean=True):
        mu, logvar = self.encode(x)
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        if self.use_feature:
            feature_pred = self.predict_feature(z)
        else:
            feature_pred = None
        return x_recon, mu, logvar, z, feature_pred

class L3Loss(torch.nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) ** 4)

def vae_feature_loss(x_recon, x, mu, logvar, feature_pred, feature, alpha_recon=0, alpha_feature=0, beta=1.0):
    """
    x, x_recon: (B, input_dim)
    mu, logvar: (B, latent_dim)

    beta: KL 가중치 (β-VAE 스타일로 조절)
    """
    # reconstruction loss: MSE
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    # 
    # recon_loss = L3Loss()(x_recon, x)

    feature_loss = F.mse_loss(feature_pred, feature, reduction="mean") if feature_pred is not None else 0.0

    # KL divergence: D_KL(q(z|x) || N(0, I))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = alpha_recon * recon_loss + beta * kl + alpha_feature * feature_loss
    return loss, recon_loss, kl, feature_loss



# In[243]:


def seed_everything(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


# In[244]:


import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train_seed = 2023
seed_everything(train_seed)


input_data = np.log1p(np.array(records["all"], dtype=np.float32))

scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

X_train, X_val = train_test_split(
    input_data_scaled,  test_size=0.2, random_state=train_seed
)


# feature 없음
train_dataset = FeatureDataset(X_train)
val_dataset   = FeatureDataset(X_val)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)


# In[245]:


from sklearn.metrics import r2_score
import itertools
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




input_dim = X_train.shape[-1]
latent_dim = 64
hidden_dim = 256


hyperparameter = {
    'beta': [0.01],
    'alpha_recon': [1.0],
    'alpha_feature': [0.0],
    'latent_dim': [64],
    'lr': [1e-3],
}

cnt = 0
epochs = 500


for vals in itertools.product(*hyperparameter.values()):
    (beta, alpha_recon, alpha_feature, latent_dim, lr) = vals
    cnt += 1
    print("=============================================")
    print(f"Experiment {cnt}/{len(list(itertools.product(*hyperparameter.values())))}")
    print(f"beta={beta}, alpha_recon={alpha_recon}, alpha_feature={alpha_feature},\nepochs={epochs}, latent_dim={latent_dim}, hidden_dim={hidden_dim}, lr={lr}")

    seed_everything(train_seed)

    vae = VAE_feature_head(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # early stopping
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(1, epochs+1):
        vae.train()
        for x_batch in train_loader:
            if len(x_batch) == 2:
                x_batch, feature_batch = x_batch
                feature_batch = feature_batch.to(device)
            else:
                feature_batch = None
            x_batch = x_batch.to(device)  # (N, D)
            
            

            x_recon, mu, logvar, z, feature_pred = vae(x_batch, use_mean=False)

            loss, recon_loss, kl, feature_loss = vae_feature_loss(x_recon, x_batch, mu, logvar, feature_pred, feature_batch, alpha_recon=alpha_recon, alpha_feature=alpha_feature, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        vae.eval()
        for x_batch in val_loader:
            if len(x_batch) == 2:
                x_batch, feature_batch = x_batch
                feature_batch = feature_batch.to(device)
            else:
                feature_batch = None
            x_batch = x_batch.to(device)
            if feature_batch is not None:
                feature_batch = feature_batch.to(device)
            x_recon, mu, logvar, z, feature_pred = vae(x_batch, use_mean=True)
            val_loss, val_recon_loss, val_kl, val_feature_loss = vae_feature_loss(x_recon, x_batch, mu, logvar, feature_pred, feature_batch, alpha_recon=alpha_recon, alpha_feature=alpha_feature, beta=beta)
            val_recon_r2 = r2_score(x_batch.detach().cpu().numpy(), x_recon.detach().cpu().numpy())
            if feature_batch is not None:
                val_feature_r2 = r2_score(feature_batch.detach().cpu().numpy(), feature_pred.detach().cpu().numpy())
            else:
                val_feature_r2 = None

        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    print(f"epoch {epoch}: loss={loss.item():.4f}, recon={recon_loss.item():.4f}, kl={kl.item():.4f}")
    print(f"epoch {epoch}: val loss={val_loss.item():.4f}, val recon={val_recon_loss.item():.4f}, val kl={val_kl.item():.4f}")

    print(f"Recon R2 : {val_recon_r2}, Feature R2 : {val_feature_r2}")


# In[246]:


class VAECostPredictor(nn.Module):
    """
    VAE 기반 Cost Regression 모델
    
    구조:
    - input → segment_encoder → segment_sum → VAE encoder → z → cost_predictor → cost
    
    특징:
    - Pretrained VAE encoder를 finetune (작은 learning rate)
    - Cost predictor는 더 큰 learning rate로 학습
    - 전체 forward 경로가 완전히 미분 가능 (detach, stop_grad 없음)
    """
    
    def __init__(self, input_dim, feature_dim=None, hidden_dim=256, latent_dim=64, 
                 predictor_hidden=256, predictor_layers=2, dropout=0.1, use_feature=False):
        super(VAECostPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ========== Cost Predictor (새로 학습) ==========
        predictor_modules = []
        current_dim = latent_dim
        for i in range(predictor_layers):
            predictor_modules.extend([
                nn.Linear(current_dim, predictor_hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if i < predictor_layers - 1 else nn.Identity(),
            ])
            current_dim = predictor_hidden
        predictor_modules.append(nn.Linear(predictor_hidden, 1))
        
        self.cost_predictor = nn.Sequential(*predictor_modules)

        self.use_feature = use_feature
        if self.use_feature:
            pass
            self.feature_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),  # feature_dim는 feature 차원
            )
        
    
    def encode(self, input_data, use_mean=False):
        """
        Full encoding path: features → z
        완전히 미분 가능
        """
                
        # VAE Encoder
        h = self.encoder(input_data)
        
        mean = self.fc_mu(h)
        if not use_mean:
            logvar = self.fc_logvar(h)
        else:
            return mean
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick - 미분 가능"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def predict_cost(self, z):
        """z → cost prediction - 완전히 미분 가능"""
        return self.cost_predictor(z).squeeze(-1)
    
    def predict_feature(self, z):
        return self.feature_predictor(z)
    
    def forward(self, input_data, use_mean=True):
        """
        Forward pass: input → z → cost
        
        Args:
            use_mean: True면 reparameterize 대신 mean 사용 (inference용)
        
        Returns:
            cost_pred: 예측된 cost
            mean: latent mean
            logvar: latent log-variance
            z: sampled/mean latent vector
        """
        mean, logvar = self.encode(input_data)
        
        if use_mean:
            z = mean  # Inference시 deterministic
        else:
            z = self.reparameterize(mean, logvar)  # Training시 stochastic
        
        cost_pred = self.predict_cost(z)
        
        return cost_pred, mean, logvar, z
    
    def get_encoder_params(self):
        """Encoder 파라미터 (작은 lr)"""
        encoder_params = []
        encoder_params.extend(self.encoder.parameters())
        encoder_params.extend(self.fc_mu.parameters())
        encoder_params.extend(self.fc_logvar.parameters())
        return encoder_params
    
    def get_cost_predictor_params(self):
        """Predictor 파라미터 (큰 lr)"""
        return self.cost_predictor.parameters()
    
    def get_feature_predictor_params(self):
        """Feature Predictor 파라미터"""
        return self.feature_predictor.parameters()

    def load_pretrained_encoder(self, checkpoint):
        """Pretrained VAE encoder 가중치 로드"""
        

        vae_state = checkpoint
        
        # 매칭되는 키만 로드
        encoder_keys = ['encoder', 'fc_mu', 'fc_logvar']
        own_state = self.state_dict()
        
        loaded_keys = []
        for name, param in vae_state.items():
            if any(name.startswith(k) for k in encoder_keys):
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded_keys.append(name)
        
        # print(f"Loaded {len(loaded_keys)} parameters from pretrained VAE")
        # return loaded_keys

    def _enable_dropout(self):
        """모든 Dropout 모듈을 train 모드로 강제 활성화"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def mc_predict(self, input_tensor, T=20):
        """
        MC Dropout 기반 불확실성 추정
        
        Args:
            input_tensor: 입력 텐서 (shape [N, input_dim])
            T: MC 샘플 수
        
        Returns:
            mean: epistemic 평균 cost (shape [N])
            var: epistemic 분산 (shape [N])
        """

        self.eval()  # 전체 모델을 eval 모드로
        self._enable_dropout()  # Dropout만 train 모드로 활성화
        
        
        with torch.no_grad():
            predictions = []
            
            for _ in range(T):
                # Encode
                z, logvar = self.encode(input_tensor)
                cost_pred = self.predict_cost(z)
                predictions.append(cost_pred)
            
            predictions = torch.stack(predictions, dim=0)
            
            # epistemic mean & variance
            mc_mean = predictions.mean(dim=0)
            mc_var = predictions.var(dim=0)

        return mc_mean, mc_var


# In[247]:


def reg_loss_fn(cost_pred, cost_true, loss_type='mse'):
    """
    기본 회귀 손실 (MSE 또는 MAE)
    """
    if loss_type == 'mse':
        return F.mse_loss(cost_pred, cost_true)
    else:  # mae
        return F.l1_loss(cost_pred, cost_true)


def pair_loss_fn(cost_pred, cost_true, margin=0.1):
    """
    Pairwise ranking loss: 실제 cost 순서를 예측이 유지하도록.
    cost_true[i] < cost_true[j] 이면 cost_pred[i] < cost_pred[j] + margin
    """
    batch_size = cost_pred.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=cost_pred.device)
    
    # 모든 쌍에 대해 ranking loss 계산
    idx = torch.arange(batch_size, device=cost_pred.device)
    i_idx, j_idx = torch.meshgrid(idx, idx, indexing='ij')
    mask = i_idx < j_idx  # upper triangular only
    
    pred_i = cost_pred[i_idx[mask]]
    pred_j = cost_pred[j_idx[mask]]
    true_i = cost_true[i_idx[mask]]
    true_j = cost_true[j_idx[mask]]
    
    # label: 1 if true_i < true_j, -1 otherwise
    labels = torch.sign(true_j - true_i).float()
    
    # Margin ranking loss
    loss = F.margin_ranking_loss(pred_j.view(-1), pred_i.view(-1), labels.view(-1), margin=margin)
    return loss


def smooth_loss_fn(model, z, noise_std=0.1):
    """
    Smoothness loss: z에 작은 노이즈를 더했을 때 예측이 크게 변하지 않도록.
    """
    was_training = model.training
    model.eval()
    
    z_noisy = z + noise_std * torch.randn_like(z)
    
    cost_original = model.predict_cost(z)
    cost_noisy = model.predict_cost(z_noisy)
    
    smooth_loss = F.mse_loss(cost_original, cost_noisy)
    
    if was_training:
        model.train()
    
    return smooth_loss


def kld_loss_fn(mean, logvar):
    """
    KL Divergence: q(z|x) || N(0, I)
    """
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return kld

def feature_loss_fn(use_feature, feature_pred, feature_true, coef=0.1):
    """
    Feature 예측 손실 (MSE)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_feature:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(feature_pred, feature_true) * coef


def compute_total_loss(model, cost_pred, mean, logvar, z, labels, feature, config, return_components=True):
    """
    Total loss 계산 (Segment 기반 데이터용).
    total_loss = reg_loss + λ_pair * pair_loss + γ * smooth_loss + β * kld_loss
    """
    
    # Individual losses
    reg = reg_loss_fn(cost_pred, labels, loss_type=config.get('loss_type', 'mse'))
    pair = pair_loss_fn(cost_pred.view(-1), labels.view(-1), margin=config.get('margin', 0.1))
    smooth = smooth_loss_fn(model, z, noise_std=config.get('noise_std', 0.1))
    kld = kld_loss_fn(mean, logvar)
    feature_loss = feature_loss_fn(model.use_feature, None, feature, coef=0)
    
    # Weighted sum
    total = config['lambda_reg'] * reg + config['lambda_pair'] * pair + config['gamma'] * smooth + config['beta'] * kld + feature_loss
    
    if return_components:
        return total, {
            'reg_loss': reg.item(),
            'pair_loss': pair.item(),
            'smooth_loss': smooth.item(),
            'kld_loss': kld.item(),
            'feature_loss': feature_loss.item(),
        }
    return total


# In[248]:


def pair_accuracy(cost_pred, labels, rng=np.random.default_rng(42)):
    """
    cost_pred, labels: (B,) 텐서
    """
    n_samples = min(1000, len(cost_pred))
    sample_indices = rng.choice(len(cost_pred), n_samples, replace=False)

    correct = 0
    total = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            idx_i = sample_indices[i]
            idx_j = sample_indices[j]
            pred_diff = cost_pred[idx_i] - cost_pred[idx_j]
            true_diff = labels[idx_i] - labels[idx_j]
            if (pred_diff * true_diff) > 0:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def recall_at_k(pred, labels, k=1):
    true_best_idx = torch.argmax(labels)
    topk_pred_idx = torch.topk(pred, k=k, largest=True).indices

    return int((topk_pred_idx == true_best_idx).any())


# In[249]:


def xgb_select_indices(xgb_all_preds, train_indices, test_indices, topk_size, eps_greedy_size, rng):
    """
    랜덤으로 2개, xgb 모델로 상위 62개 선택
    """
    # 남은 인덱스 중에서 무작위로 random_select_size개 선택

    remaining_indices = set(test_indices)

    if topk_size + eps_greedy_size > test_indices.shape[0]:
        remaining_indices.update(train_indices.tolist())
        train_indices = np.array(list(remaining_indices), dtype=np.int64)
        return train_indices, np.array([], dtype=np.int64)


    top_indices, remaining_indices = select_topk_cost(xgb_all_preds, remaining_indices, topk_size)
    random_indices, remaining_indices = random_select_indices(remaining_indices, eps_greedy_size, rng=rng)
    test_indices = np.array(list(remaining_indices), dtype=np.int64)

    selected_indices = np.concatenate([top_indices, random_indices])

    train_indices = np.concatenate([train_indices, selected_indices])

    return train_indices, test_indices



def random_select_indices(remaining_indices, select_size, rng=np.random.default_rng(42)):
    if select_size == 0:
        return np.array([], dtype=np.int64), remaining_indices
    
    random_indices = rng.choice(list(remaining_indices), size=select_size, replace=False)

    remaining_indices = util_update_remaining_indices(remaining_indices, random_indices)

    return random_indices, remaining_indices



def util_update_remaining_indices(remaining_indices, selected_indices):
    """
    남은 인덱스 집합 업데이트
    util_update_remaining_indices에서 selected_indices 제거
    """
    selected_indices = set(selected_indices)
    remaining_indices.difference_update(selected_indices)

    return remaining_indices



def util_select_topk(predictions, remaining_indices, num_select):
    """
    예측값 기반 다음 측정할 샘플 선택
    
    Args:
        predictions: 전체 예측값 리스트 ([N, ] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 numpy 배열
        remaining_indices: 업데이트된 남은 인덱스 집합 (set)
    """
    
    prediction = np.asarray(predictions)  # [N]

    remaining_np = np.array(list(remaining_indices), dtype=np.int64)
    remaining_pred = prediction[remaining_np]

    k = min(num_select, len(remaining_np))

    topk_local = np.argsort(remaining_pred)[-k:]
    selected_indices = remaining_np[topk_local]

    # remaining 업데이트
    remaining_indices.difference_update(selected_indices.tolist())

    return selected_indices, remaining_indices






def select_topk_cost(cost_pred, remaining_indices, num_select):
    """
    예측된 cost 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_data_scaled: 전체 input 리스트 ([N, input_dim] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
    
    """
    if num_select == 0:
        return np.array([], dtype=np.int64), remaining_indices

    if isinstance(cost_pred, torch.Tensor):
        cost_pred = cost_pred.detach().cpu().numpy()  # [N]

    topk_cost_indices, remaining_indices = util_select_topk(cost_pred, remaining_indices, num_select)
    

    return topk_cost_indices, remaining_indices


def select_topk_z_grad(z, cost_pred, remaining_indices, num_select):
    """
    z에 대한 cost gradient 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_tensor: 전체 input numpy 배열 ([N, input_dim] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
    
    """
    if num_select == 0:
        return np.array([], dtype=np.int64), remaining_indices

    candidate_indices = np.array(list(remaining_indices), dtype=np.int64)

    # z-gradient 계산
    z_grad = torch.autograd.grad(
        outputs=cost_pred.sum(),
        inputs=z,
        retain_graph=False,
        create_graph=False
    )[0]  # [N, latent_dim]

    z_grad_norm = torch.norm(z_grad, dim=1).detach().cpu().numpy()  # [N]

    # 후보 중 grad-norm top-k
    candidate_grad = z_grad_norm[candidate_indices]
    k = min(num_select, len(candidate_indices))

    topk_local = np.argsort(candidate_grad)[-k:]
    selected_indices = candidate_indices[topk_local]

    # remaining 업데이트
    remaining_indices = set(remaining_indices)
    remaining_indices.difference_update(selected_indices.tolist())

    return selected_indices, remaining_indices


def select_topk_uncertainty(model, input_tensor, remaining_indices, num_select, T_mc=10):
    """
    MC Dropout 기반 불확실성 추정으로 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_data_scaled: 전체 input 리스트 ([N, input_dim] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
        T_mc: MC Dropout 샘플 수
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 리스트
    """
    if num_select == 0:
        return np.array([], dtype=np.int64), remaining_indices


    was_training = model.training
    model.train()

    with torch.no_grad():
        _, mc_var = model.mc_predict(input_tensor, T=T_mc)

    if not was_training:
        model.eval()  # 원복

    var_np = mc_var.detach().cpu().numpy()  # [N]

    topk_uncertainty_indices, remaining_indices = util_select_topk(var_np, remaining_indices, num_select)

    return topk_uncertainty_indices, remaining_indices


def select_topk_latent_diversity(z, candidate_indices, used_indices, select_n_div, chunk_size=1024, eps=1e-12):
    """
    먼저 candidates 320개를 뽑았다고 치자.
    이후 앞에서 topk_cost, topk_z_grad로 40개 정도를 뽑았다고 치자.
    latent diversity는 40개 + used_indices로부터 가장 멀리 떨어진 24개를 280개에서 뽑는다.

    z를 L2 정규화한 뒤, k-center greedy(farthest-first)로 diversity 선택.
    초기 센터는 used_indices (이미 측정된 점들).
    매 스텝마다 "센터 집합까지의 최소거리"가 최대인 candidate를 하나씩 추가.
    
    Args:
        z: torch.Tensor [N, latent_dim]
        candidate_indices: set(int)
        used_indices: set(int)
        select_n_div: int
        chunk_size: int
    Returns:
        diverse_indices: np.ndarray (int64)
        candidate_indices: set (선택된 인덱스 제거된 상태)
    """
    if select_n_div == 0 or len(candidate_indices) == 0:
        return np.array([], dtype=np.int64), candidate_indices


    device = z.device

    # 1) L2 normalize z  (각 벡터를 단위벡터로)
    with torch.no_grad():
        z_norm = z / (z.norm(dim=1, keepdim=True) + eps)

    cand = np.array(list(candidate_indices), dtype=np.int64)
    k = min(select_n_div, len(cand))

    cand_t = torch.from_numpy(cand).to(device=device)
    z_cand = z_norm[cand_t]  # [M, D], M=len(cand)

    # 초기 센터: used_indices (비어있을 수도 있음)
    used = np.array(list(used_indices), dtype=np.int64)
    selected = []

    # 2) 각 candidate의 "현재 센터 집합까지 최소거리" 벡터 init
    #    used가 비어있으면 +inf로 시작해서 임의 첫 점을 뽑게(가장 큰 값) 만들기
    if len(used) > 0:
        used_t = torch.from_numpy(used).to(device=device)
        z_used = z_norm[used_t]  # [U, D]

        # min_dists[j] = min_{u in used} ||z_cand[j] - z_used[u]||
        min_dists = torch.empty(len(cand), device=device, dtype=torch.float32)

        with torch.no_grad():
            for s in range(0, len(cand), chunk_size):
                e = min(s + chunk_size, len(cand))
                d = torch.cdist(z_cand[s:e], z_used, p=2)  # [B, U]
                min_dists[s:e] = d.min(dim=1).values
    else:
        # 센터가 없으면 모두 동일하게 시작 → 첫 선택은 아래 argmax가 0번째로 갈 수 있음
        # 다양성 목적이면 랜덤/최대 norm 등으로 첫 점을 정할 수도 있지만,
        # 여기서는 "가장 큰 min_dists"를 위해 +inf로 둔다.
        min_dists = torch.full((len(cand),), float("inf"), device=device, dtype=torch.float32)

    # 3) k-center greedy 반복
    #    매번 argmax(min_dists) 하나 선택 -> 그 점을 센터에 추가 -> min_dists 갱신
    with torch.no_grad():
        for _ in range(k):
            j = torch.argmax(min_dists).item()     # cand 내부 위치
            sel_idx = cand[j]                      # 원본 인덱스
            selected.append(sel_idx)

            # 선택된 점을 "센터"로 추가: 모든 candidate에 대해 dist_to_new_center 계산 후 min 갱신
            new_center = z_cand[j:j+1]  # [1, D]

            # 방금 뽑은 점은 다시 뽑히지 않게 min_dists를 -inf로
            min_dists[j] = -float("inf")

            # 나머지 후보들의 min 거리 업데이트
            for s in range(0, len(cand), chunk_size):
                e = min(s + chunk_size, len(cand))
                d_new = torch.cdist(z_cand[s:e], new_center, p=2).squeeze(1)  # [B]
                min_dists[s:e] = torch.minimum(min_dists[s:e], d_new)

    diverse_indices = np.array(selected, dtype=np.int64)

    candidate_indices = set(candidate_indices)
    candidate_indices.difference_update(diverse_indices.tolist())

    return diverse_indices, candidate_indices


def select_init_latent_diversity(model, input_data_scaled, remaining_indices, select_num, device):
    model.eval()

    # remaining indices를 리스트로 고정
    rem_idx = np.array(list(remaining_indices), dtype=np.int64)
    select_num = min(select_num, len(rem_idx))

    input_tensor = torch.tensor(
        input_data_scaled[rem_idx],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        z = model.encode(input_tensor, use_mean=True)  # (M, D)

    z = z.detach()
    M = z.size(0)

    selected_local = []

    # 1) 첫 점 랜덤 (remaining 내부)
    first = torch.randint(0, M, (1,), device=device).item()
    selected_local.append(first)

    # 2) farthest-point greedy (remaining 내부)
    dist = torch.cdist(z, z[[first]])[:, 0]  # (M,)

    for _ in range(1, select_num):
        idx = torch.argmax(dist).item()
        selected_local.append(idx)

        new_dist = torch.cdist(z, z[[idx]])[:, 0]
        dist = torch.minimum(dist, new_dist)

    # local index → global index
    selected_global = rem_idx[selected_local]

    # remaining 업데이트
    remaining_indices.difference_update(selected_global.tolist())

    return selected_global, remaining_indices


def select_representative_kmeans(model, input_data_scaled, remaining_indices, select_num, device, iters=10):
    model.eval()
    x = torch.tensor(input_data_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        z = model.encode(x, use_mean=True)  # (N, D)
    z = z.detach()
    N, D = z.shape
    k = min(select_num, N)

    # --- kmeans++ 초기화 (center는 실제 데이터 포인트로 잡음) ---
    centers_idx = []
    first = torch.randint(0, N, (1,), device=z.device).item()
    centers_idx.append(first)

    dist = torch.cdist(z, z[[first]])[:, 0]  # (N,)
    for _ in range(1, k):
        probs = (dist ** 2)
        probs = probs / probs.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, 1).item()
        centers_idx.append(idx)
        dist = torch.minimum(dist, torch.cdist(z, z[[idx]])[:, 0])

    centers = z[centers_idx].clone()  # (k, D)

    # --- Lloyd iterations ---
    for _ in range(iters):
        d = torch.cdist(z, centers)          # (N, k)
        assign = torch.argmin(d, dim=1)      # (N,)
        new_centers = centers.clone()
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                new_centers[j] = z[mask].mean(dim=0)
        centers = new_centers

    # --- 각 중심에 가장 가까운 실제 데이터 인덱스 선택 ---
    d = torch.cdist(z, centers)  # (N, k)
    selected = []
    used = set()
    for j in range(k):
        # 중심 j에 가장 가까운 점부터 시도하되, 중복 방지
        order = torch.argsort(d[:, j])
        for idx in order.tolist():
            if idx not in used:
                used.add(idx)
                selected.append(idx)
                break

    remaining_indices.difference_update(selected)
    k_means_indices = np.array(selected, dtype=np.int64)

    return k_means_indices, remaining_indices

def select_programs(model, input_data_scaled, used_indices, remaining_indices, num_select=64, T_mc=10, uncertainty_topk=128,
                    w_cost=0.5, w_unc=0.3, w_div=0.2, grad_num=2, rand_num=0, rng=np.random.default_rng(42), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), topk_factor=5):
    """
    Active learning 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_data_scaled: 전체 input 리스트 ([N, input_dim] 형태)
        used_indices: 이미 측정된 인덱스 집합(set)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
        T_mc: MC Dropout 샘플 수
        w_cost: 예측값이 큰 샘플 가중치
        w_unc: epistemic 불확실성이 높은 샘플 가중치
        w_div: latent 다양성이 높은 샘플 가중치
        grad_num: z에 대한 cost의 gradient가 큰 샘플 수
        rand_num: 무작위로 선택할 샘플 수
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 리스트
    """

    # 합쳐서 64개 선택
    total = num_select
    budget = total - grad_num - rand_num

    # 랜덤 선택만 할 경우
    if num_select == 0 and rand_num > 0:
        rand_indices, remaining_indices = random_select_indices(remaining_indices, rand_num, rng=rng)
        return rand_indices, remaining_indices
    

    select_n_cost = int(budget * w_cost)
    select_n_unc  = int(budget * w_unc)
    select_n_div  = int(budget * w_div)
    select_n_grad = grad_num
    s = select_n_cost + select_n_unc + select_n_div
    if s < budget:
        select_n_cost += budget - s

    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32, device=device)
    

    model.eval()
    with torch.no_grad():
        z, _ = model.encode(input_tensor)
    z = z.detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    cost_pred = model.predict_cost(z)
    cost_pred = cost_pred.view(-1)
    cost_np = cost_pred.detach().cpu().numpy()

    remaining_np = np.array(list(remaining_indices), dtype=np.int64)
    remaining_cost = cost_np[remaining_np]

    k_pref = min(len(remaining_np), total * topk_factor)
    top_local = np.argsort(remaining_cost)[-k_pref:]
    candidate_indices = set(remaining_np[top_local].tolist())  # 작업용 remaining

    # print(f"Candidate pool size: {len(candidate_indices)}")


    # 중복 방지용
    currently_used = set()
    topk_cost_indices, candidate_indices = select_topk_cost(cost_pred, candidate_indices, select_n_cost)
    currently_used.update(topk_cost_indices.tolist())
    z_grad_indices, candidate_indices = select_topk_z_grad(z, cost_pred, candidate_indices, select_n_grad)
    currently_used.update(z_grad_indices.tolist())

    # if len(used_indices) / len(input_data_scaled) >= 0.1:
    if len(used_indices) >= uncertainty_topk:
        uncertainty_indices, candidate_indices = select_topk_uncertainty(model, input_tensor, candidate_indices, select_n_unc, T_mc=T_mc)
    else:
        pool_for_uncertainty = set(remaining_indices)
        pool_for_uncertainty.difference_update(currently_used)
        uncertainty_indices, _ = select_topk_uncertainty(model, input_tensor, pool_for_uncertainty, select_n_unc, T_mc=T_mc)
        candidate_indices.difference_update(uncertainty_indices.tolist())


    currently_used.update(uncertainty_indices.tolist())
    used_local = set(used_indices)
    used_local.update(currently_used)

    diverse_indices, _ = select_topk_latent_diversity(z, candidate_indices, used_local, select_n_div)
    currently_used.update(diverse_indices.tolist())


    remaining_indices.difference_update(currently_used)


    rand_indices, remaining_indices = random_select_indices(remaining_indices, rand_num, rng=rng)
    currently_used.update(rand_indices.tolist())

    

    all_selected_indices = np.array(sorted(currently_used), dtype=np.int64)



    return all_selected_indices, remaining_indices


# In[250]:


def make_vae_reg_dataloaders(input_data_scaled, costs, used_indices, remaining_indices):

    train_indices = np.array(list(used_indices), dtype=np.int64)
    val_indices = np.array(list(remaining_indices), dtype=np.int64)

    X_train = input_data_scaled[train_indices]
    X_val = input_data_scaled[val_indices]
    y_train = costs[train_indices]
    y_val = costs[val_indices]

    train_dataset = FeatureRegressionDataset(X_train, y_train)
    val_dataset   = FeatureRegressionDataset(X_val,   y_val)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)



    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8  # 0 나누기 방지용 작은 값 추가
    print(f"y_train mean: {y_mean}, std: {y_std}")

    
    return train_loader, val_loader, y_mean, y_std


def make_xgb_datasets(inputs, results):
    f_inputs = []
    f_results = []
    r_costs = []
    for inp, res in zip(inputs, results):
        cost = np.mean([c.value for c in res.costs])
        if cost < 1e10:
            f_inputs.append(inp)
            f_results.append(res)
            r_costs.append(cost)
    r_costs = np.array(r_costs, dtype=np.float32)
    
    dataset = auto_scheduler.dataset.Dataset()
    dataset.update_from_measure_pairs(f_inputs, f_results)
    return dataset


def split_xgb_datasets(dataset, train_indices, test_indices):

    raw_features = list(dataset.features.values())[0]
    raw_throughputs = list(dataset.throughputs.values())[0]

    
    train_set, test_set = dataset.random_split_within_task(train_set_ratio=0, 
                                                        train_idxs=train_indices.tolist(), 
                                                        test_idxs=test_indices.tolist())
    return train_set, test_set, raw_throughputs


# In[251]:


def make_vae_reg_model(vae, config, input_dim, latent_dim, hidden_dim, y_std, verbose=True):
    
    print(f"lambda_reg={config['lambda_reg']}, lambda_pair={config['lambda_pair']}, margin_scale={config['margin_scale']}, epochs={config['epochs']}, gamma={config['gamma']}, beta={config['beta']}, noise_std={config['noise_std']}",
            f"\nscratch={config['scratch']}, encoder_freeze={config['encoder_freeze']}, encoder_lr={config['encoder_lr']}, cost_predictor_lr={config['cost_predictor_lr']}")

    vae_cost_model = VAECostPredictor(input_dim=input_dim, 
                                latent_dim=latent_dim, 
                                hidden_dim=hidden_dim, 
                                predictor_layers=2,
                                dropout=0.1, use_feature=False).to(device)
    if not config['scratch']:
        vae_cost_model.load_pretrained_encoder(vae.state_dict())
    
    if config['encoder_freeze']:
        for param in vae_cost_model.get_encoder_params():
            param.requires_grad = False
        optimizer = torch.optim.AdamW([
            {'params': vae_cost_model.get_cost_predictor_params(), 'lr': config['cost_predictor_lr']},
        ], weight_decay=1e-5)
    else:
        for param in vae_cost_model.get_encoder_params():
            param.requires_grad = True
        optimizer = torch.optim.AdamW([
            {'params': vae_cost_model.get_encoder_params(), 'lr': config['encoder_lr']},
            {'params': vae_cost_model.get_cost_predictor_params(), 'lr': config['cost_predictor_lr']},
        ], weight_decay=1e-5)
        
    return vae_cost_model, optimizer, config


# In[252]:


def lambda_pair_warmup(epoch: int, warmup_epochs: int, lambda_pair_max: float) -> float:
    if warmup_epochs <= 0:
        return lambda_pair_max
    t = min(max(epoch, 0), warmup_epochs) / warmup_epochs  # 0~1
    return lambda_pair_max * t


# In[253]:


def train_regression(vae_cost_model, optimizer, train_loader, val_loader, input_data_scaled, costs, config, top_k=10, use_rank=True, warmup_epochs=200):

    # print("Train size :", len(train_loader.dataset))

    # all_reg_results = []

    lambda_pair_max = config['lambda_pair']

    for epoch in range(1, config['epochs']+1):
        vae_cost_model.train()
        for x_batch, labels in train_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device).squeeze(-1)
            
        
            cost_pred, mean, logvar, z = vae_cost_model(x_batch, use_mean=True)

            config['lambda_pair'] = lambda_pair_warmup(epoch, warmup_epochs, lambda_pair_max)

            
            train_loss, train_components = compute_total_loss(vae_cost_model, 
                                                    cost_pred, mean, logvar, z, labels, None, config)

            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_cost_model.parameters(), max_norm=1.0)
            optimizer.step()
            
        

        if epoch % config['epochs'] == 0:
            vae_cost_model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for x_batch, labels in val_loader:
                    x_batch = x_batch.to(device)
                    labels = labels.to(device).squeeze(-1)

                    cost_pred, mean, logvar, z = vae_cost_model(x_batch, use_mean=True)
                    all_preds.append(cost_pred)
                    all_labels.append(labels)

                    val_loss, val_components = compute_total_loss(vae_cost_model, cost_pred, mean, logvar, z, labels, None, config)
                val_reg_r2 = r2_score(torch.cat(all_labels).detach().cpu().numpy(), torch.cat(all_preds).detach().cpu().numpy())
                val_reg_r2 = round(val_reg_r2, 4)
                
                print(f"Train loss epoch {epoch} : reg={train_components['reg_loss']: .4f} rank={train_components['pair_loss']: .4f} kl={train_components['kld_loss']: .4f}")
                print(f"Val loss epoch {epoch}: reg={val_components['reg_loss']: .4f} rank={val_components['pair_loss']: .4f} kl={val_components['kld_loss']: .4f}")
                
                print(f"Regression R2 : {val_reg_r2:.4f}, ")
        
        # rank r2 계산
        vae_cost_model.eval()
        with torch.no_grad():
            if epoch % config['epochs'] == 0:
                input_data_tensor = torch.from_numpy(input_data_scaled).float().to(device)
                all_preds = vae_cost_model(input_data_tensor, use_mean=True)[0].detach().cpu().numpy()
                if use_rank:
                    val_rank_r2 = pair_accuracy(all_preds, costs)
                    val_rank_r2 = round(val_rank_r2, 4)
                    print(f"Rank R2 : {val_rank_r2:.4f}")
                else:
                    val_rank_r2 = None
                recall_top_k = recall_at_k(torch.tensor(all_preds), torch.from_numpy(costs), k=top_k)
                
                print(f"Recall@{top_k} : {recall_top_k}")

    return vae_cost_model, recall_top_k, val_reg_r2, val_rank_r2


# In[254]:


def generate_weight_grid(step=0.1):
    m = int(round(1.0 / step))  # step=0.1 -> 10
    weights = []
    for i in range(m + 1):
        for j in range(m + 1):
            k = m - i - j
            if k < 0:
                continue
            weights.append((i/m, j/m, k/m))
    return weights
weights = generate_weight_grid(step=0.1)



# In[255]:


f_weights = []
for w in weights:
    w_cost, w_unc, w_div = w
    if w_cost < 0.3:
        continue
    # if w_unc == 0.0 and w_cost > 0.0 and w_div > 0.0:
    #     f_weights.append(w)
    #     continue
    # if w_div == 0.0 and w_cost > 0.0 and w_unc > 0.0:
    #     f_weights.append(w)
        # continue
    f_weights.append(w)


# In[ ]:


def filter_already_measured(total_csv_path, sampling_hyper):

    if total_csv_path is not None:
        total_csv = pd.read_csv(total_csv_path)

        measured_keys = {
            (
                row.measure_size,

                # row.scratch,
                # row.encoder_freeze,
                
                row.encoder_lr,
                row.cost_predictor_lr,
                row.weights,
                row.sampling_seed,
                row.rank_warmup_epochs,

                # row.uncertainty_topk,
                row.grad_num,
                row.rand_num,
            )
            for row in total_csv.itertuples(index=False)
        }

    else:
        measured_keys = set()
    to_measure_configs = []

    for params in itertools.product(*sampling_hyper.values()):
        hyper_config = dict(zip(sampling_hyper.keys(), params))

        key = (
            hyper_config["measure_size"],
            # hyper_config["scratch"],
            # hyper_config["encoder_freeze"],

            hyper_config["encoder_lr"],
            hyper_config["cost_predictor_lr"],
            str(hyper_config["weight"]),
            hyper_config["sampling_seed"],
            hyper_config["warmup_epochs"],
            
            # hyper_config["uncertainty_topk"],
            hyper_config["grad_num"],
            hyper_config["rand_num"],
        )

        if key in measured_keys:
            continue

        to_measure_configs.append(hyper_config)
    
    return to_measure_configs

def filter_unnecessary_configs(to_measure_configs, filtering):
    filtered_configs = []
    for config in to_measure_configs:
        skip = False
        for f in filtering:
            key1, val1, key2, val2 = f
            if config[key1] == val1 and config[key2] == val2:
                skip = True
                break
        if not skip:
            filtered_configs.append(config)
    return filtered_configs

def add_specific_configs(to_measure_configs, specific_configs, sampling_hyper):
    # 특정 설정 추가
    # 특정 설정 이외에 다른 하이퍼파라미터는 전체 조합으로
    for spec in specific_configs:
        key1, val1, key2, val2 = spec
        other_keys = [k for k in sampling_hyper.keys() if k != key1 and k != key2]
        for other_params in itertools.product(*[sampling_hyper[k] for k in other_keys]):
            hyper_config = {key1: val1, key2: val2}
            hyper_config.update(dict(zip(other_keys, other_params)))
            to_measure_configs.append(hyper_config)
    return to_measure_configs


def save_avg_csv(df_results, filename, top_k):
    group_cols = [
        "measure_size",
        "scratch",
        "encoder_freeze",
        "encoder_lr",
        "cost_predictor_lr",
        "rank_warmup_epochs",
        "weights",
        "uncertainty_topk",
        "T_mc",
        "grad_num",
        "rand_num",
    ]

    df_avg = (
        df_results
        .groupby(group_cols, as_index=False)
        .agg(
            phase=("phase", "mean"),
            train_size=("train_size", "mean"),
            used_time=("used_time", "mean"),
            **{f"top-{top_k}": (f"top-{top_k}", "mean")},
            val_reg_r2=("val_reg_r2", "first"),
            val_rank_r2=("val_rank_r2", "first"),
            seed_n=("sampling_seed", "nunique"),
            sampling_seed=("sampling_seed", list),
        )
    )

    df_avg.to_csv(filename.replace(".csv", "_avg.csv"), index=False)
    df_avg


# In[ ]:


import pandas as pd
import datetime



# 데이터셋 길이만큼의 인덱스 numpy 배열 생성
all_indices = np.arange(len(input_data_scaled))
costs = np.array(records["costs"], dtype=np.float32)

real_optimum_index = np.argmax(costs)

top_k = 1

train_seed = 2023


sampling_hyper = {
    "measure_size": [48],
    "weight" : [
            # (1.0, 0.0, 0.0),
            (0.7, 0.0, 0.3),
            (0.7, 0.3, 0.0),
            (0.6, 0.1, 0.3),
            (0.5, 0.2, 0.3),
            # (0.3, 0.4, 0.3),
            (0.4, 0.3, 0.3),
            (0.3, 0.3, 0.4),
            ],
    
    "uncertainty_topk": [0, 48],    # 몇 개부터 불확실성 기반 선택을 할지
    # "weight" : f_weights,
    "grad_num": [0, 2, 4],
    "rand_num": [0],
    
    "T_mc": [20],
    "encoder_freeze": [False, True],
    "scratch": [False, True],
    "encoder_lr": [1e-5],
    "cost_predictor_lr": [1e-4, 1e-5],
    "warmup_epochs" : [0, 100, 200],

    "sampling_seed" : range(2000, 2020),
    # "seed" : [2023,2025],

}


       
now = datetime.datetime.now().strftime("%m%d_%H%M")
filename = f"result/{os.path.basename(json_file)}/vae_extent_{now}.csv"

total_csv_path = f"result/{os.path.basename(json_file)}/vae_extent_total.csv"
to_measure_configs = filter_already_measured(total_csv_path, sampling_hyper)

filtering_configs = [
    ("measure_size", 64, "uncertainty_topk", 48),
    ("encoder_freeze", True, "scratch", True),
    # ("weight", (1.0, 0.0, 0.0), "uncertainty_topk", 48),
    # ("weight", (1.0, 0.0, 0.0), "uncertainty_topk", 64),
    ("weight", (0.7, 0.0, 0.3), "uncertainty_topk", 48),
    ("weight", (0.7, 0.0, 0.3), "uncertainty_topk", 64),
]
# add_configs = [
#     ("weight", (1.0, 0.0, 0.0), "rand_num", 4),
# ]


to_measure_configs = filter_unnecessary_configs(to_measure_configs, filtering_configs)
# to_measure_configs = add_specific_configs(to_measure_configs, add_configs, sampling_hyper)



print(f"{len(list(itertools.product(*sampling_hyper.values())))} -> {len(to_measure_configs)}개의 실험 남음")

random_indices_list = []
all_results = []

cnt = 0
for hyper_config in to_measure_configs:

    # hyper_config = dict(zip(sampling_hyper.keys(), params))

    cnt += 1
    print(f"########## 실험 {cnt}/{len(to_measure_configs)} ##########")

    tic = time.time()
    # used_indices : 이미 측정된 인덱스 집합. train_indices와 동일
    # remaining_indices : 아직 측정되지 않은 인덱스 집합. val_indices와 동일
    used_indices = set()
    remaining_indices = set(all_indices)
    
    measure_size = hyper_config["measure_size"]
    sampling_seed = hyper_config["sampling_seed"]
    w_cost, w_unc, w_div = hyper_config["weight"]
    print(f"weights: {hyper_config['weight']}")
    print(f"measure_size: {hyper_config['measure_size']}, T_mc: {hyper_config['T_mc']}, sampling_seed: {hyper_config['sampling_seed']}")

    sampling_rng = np.random.default_rng(sampling_seed)

    hyperparameter = {

        'lambda_reg' : 0.01,
        'lambda_pair': 3.0,
        'margin_scale': 0.3,
        'gamma': 0.01,
        'beta': 0.01,
        'noise_std': 0.001,

        'encoder_lr': hyper_config["encoder_lr"],
        'encoder_freeze' : hyper_config["encoder_freeze"],
        'scratch': hyper_config["scratch"],
        'feature_predictor_lr': 0,
        'cost_predictor_lr': hyper_config["cost_predictor_lr"],
        'epochs': 1000,
        
    }



    seed_everything(sampling_seed)

    init_indices, remaining_indices = random_select_indices(remaining_indices, select_size=sampling_hyper["measure_size"][0], rng=sampling_rng)
    random_indices_list.append(init_indices)

    # random_num = int(sampling_hyper["measure_size"][0] * (3/4))
    # diverse_num = int(sampling_hyper["measure_size"][0] - random_num)
    # random_indices, remaining_indices = random_select_indices(remaining_indices, select_size=random_num, rng=sampling_rng)
    # diverse_indices, remaining_indices = select_init_latent_diversity(vae, input_data_scaled, remaining_indices, diverse_num, device)
    # init_indices = np.concatenate([random_indices, diverse_indices])

    # init_indices, remaining_indices = select_representative_kmeans(vae, input_data_scaled, remaining_indices, sampling_hyper["measure_size"][0], device, iters=5)


    print(f"초기 랜덤 선택 샘플 인덱스: {np.sort(init_indices)}")
    used_indices.update(init_indices)
    

    reg_history = []
    rank_history = []

    for phase in range(1, len(input_data_scaled) // measure_size + 1):

        print(f"=============== 측정 Phase {phase} ({len(used_indices)}개) ================")


        # DataLoader 갱신
        seed_everything(train_seed)
        train_loader, val_loader, y_mean, y_std = make_vae_reg_dataloaders(input_data_scaled, costs, used_indices, remaining_indices)

        
        vae_cost_model, optimizer, config = make_vae_reg_model(vae, hyperparameter, input_dim, latent_dim, hidden_dim, y_std, verbose=False)
        
        seed_everything(train_seed)
        vae_cost_model, topk_recall_signal, val_reg_r2, val_rank_r2 = train_regression(vae_cost_model, optimizer, train_loader, val_loader, 
                                                                                       input_data_scaled, costs, config, top_k=top_k, use_rank=False, warmup_epochs=hyper_config["warmup_epochs"])

        reg_history.append(val_reg_r2)
        rank_history.append(val_rank_r2)
        

        


        # 다음 측정할 샘플 선택
        selected_indices, remaining_indices = select_programs(
            model=vae_cost_model,
            input_data_scaled=input_data_scaled,
            remaining_indices=remaining_indices,
            used_indices=used_indices,
            num_select=measure_size,
            
            w_cost=w_cost,
            w_unc=w_unc,
            w_div=w_div,
            # w_cost=0.3,
            # w_unc=0.35,
            # w_div=0.35,
            uncertainty_topk=hyper_config["uncertainty_topk"],
            T_mc=hyper_config["T_mc"],
            grad_num=hyper_config["grad_num"],
            rand_num=hyper_config["rand_num"],
            device=device,
            rng=sampling_rng,
            
            topk_factor=5
        )
        # w_cost += 0.03
        # w_unc -= 0.02
        # w_div -= 0.01

        # selected_indices: numpy 배열
        used_indices.update(selected_indices.tolist())

        measured_optimum = True if real_optimum_index in used_indices else False


        use_topk = False
        

        break_signal = False
        if not use_topk and measured_optimum:
            break_signal = True
        elif use_topk and topk_recall_signal:
            break_signal = True
            filename= filename.replace("result/", "result_topk/")


        if break_signal:
            print("최적화 종료")
            print("학습한 데이터 수 :", len(used_indices)-measure_size)
            used_time = time.time() - tic
            print(f"총 측정 시간: {used_time:.2f} 초")
            print("=============================================")
            all_results.append({
                
                "scratch": hyper_config["scratch"],
                "encoder_freeze": hyper_config["encoder_freeze"],
                "measure_size": measure_size,
                "encoder_lr": hyper_config["encoder_lr"],
                "cost_predictor_lr": hyper_config["cost_predictor_lr"],
                "rank_warmup_epochs": hyper_config["warmup_epochs"],


                
                "weights": hyper_config["weight"],
                "uncertainty_topk": hyper_config["uncertainty_topk"],
                "T_mc": hyper_config["T_mc"],
                "grad_num": hyper_config["grad_num"],
                "rand_num": hyper_config["rand_num"],
                "phase" : phase,
                "used_time": round(used_time, 2),
                "train_size" : len(used_indices)-measure_size,
                f"top-{top_k}" : topk_recall_signal,
                "val_reg_r2": reg_history,
                "val_rank_r2": rank_history,
                "sampling_seed": sampling_seed,
                
                
            })
            if use_topk:
                all_results[-1]["top_k"] = top_k

            df_results = pd.DataFrame(all_results)
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df_results.to_csv(filename, index=False)
            
            break

if len(all_results) > 0:
    save_avg_csv(df_results, filename, top_k)


# In[ ]:


# import pandas as pd
# from glob import glob

# csv_dir = glob("/root/work/tenset/scripts/pre_experiments/model_myself/result/([0c9a5ba46ffc5e1a9e5641018527117f,4,7,7,160,1,1,160,960,1,1,1,960,4,7,7,960],cuda).json/vae_extent_*.csv")
# csv_dir = [p for p in csv_dir if not p.endswith("avg.csv")]

# dfs = []
# for p in csv_dir:
#     sub_df = pd.read_csv(p)

#     if "rank_warmup_epochs" not in sub_df.columns:
#         sub_df["rank_warmup_epochs"] = 0
#     if "measure_size" not in sub_df.columns:
#         sub_df["measure_size"] = 64
    
#     dfs.append(sub_df)

    
# df_total = pd.concat(dfs, ignore_index=True)
# df_total.drop(columns=["scratch", "encoder_freeze", "val_rank_r2"], inplace=True)

# # measure_size 컬럼을 맨 앞으로 이동
# cols = df_total.columns.tolist()
# df_total = df_total[["measure_size"] + [c for c in cols if c != "measure_size"]]

# df_total.to_csv(os.path.dirname(filename)+"/vae_extent_total.csv", index=False)
# df_total


# # In[212]:


# agg_kwargs = {
#     "phase": ("phase", "mean"),
#     "train_size": ("train_size", "mean"),
#     "used_time": ("used_time", "mean"),
#     "val_reg_r2": ("val_reg_r2", "first"),
#     "seed_n": ("sampling_seed", "nunique"),
#     "sampling_seed": ("sampling_seed", list),
# }

# topk_col = f"top-{top_k}"
# if topk_col in df_total.columns:
#     agg_kwargs[topk_col] = (topk_col, "mean")

# group_cols = [
#     "measure_size",
#     # "scratch",
#     # "encoder_freeze",
#     "encoder_lr",
#     "cost_predictor_lr",
#     "rank_warmup_epochs",
#     "weights",
#     "uncertainty_topk",
#     "grad_num",
#     "rand_num",
# ]

# df_total_avg = (
#     df_total
#     .groupby(group_cols, as_index=False)
#     .agg(**agg_kwargs)
# )
# df_total_avg.to_csv(os.path.dirname(filename)+"/vae_extent_total_avg.csv", index=False)
# df_total_avg




# # ## XGB test

# # In[22]:


# import warnings

# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
#     message=".*Old style callback is deprecated.*"
# )

# from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal


# inputs, results = auto_scheduler.RecordReader(json_file).read_lines()


# # In[25]:


# topk_size = int(measure_size * 0.95)
# eps_greedy_size = measure_size - topk_size


# seeds = sampling_hyper["seed"]
# random_indices = random_indices_list[:len(seeds)]

# xgb_results = []

# now = datetime.datetime.now().strftime("%m%d_%H%M")
# xgb_filename = f"result_xgb/{os.path.basename(json_file)}/xgb_search_{now}.csv"

# for i, seed in enumerate(seeds):

#     tic = time.time()
#     sample_rng = np.random.default_rng(seed)

    
    
#     tenset_model = XGBModelInternal(use_workload_embedding=False, seed=train_seed)

#     seed_everything(train_seed)
#     dataset = make_xgb_datasets(inputs, results)

    
#     used_indices = set(random_indices[i])
#     remaining_indices = set(all_indices)
#     remaining_indices.difference_update(used_indices)

#     train_indices = np.array(sorted(used_indices), dtype=np.int64)
#     test_indices = np.array(sorted(remaining_indices), dtype=np.int64)
#     print(train_indices)

#     reg_history = []
#     rank_history = []

#     for phase in range(1,  len(input_data_scaled) // measure_size + 1):

#         print(f"=============== 측정 Phase {phase} ================")

#         seed_everything(train_seed)
#         train_set, test_set, dataset_costs = split_xgb_datasets(dataset, train_indices, test_indices)
#         real_optimum_idx = np.argmax(dataset_costs)
#         seed_everything(train_seed)
#         tenset_model.fit_base(train_set=train_set)
#         xgb_all_preds = tenset_model.predict(dataset)
#         xgb_all_preds = np.array(list(xgb_all_preds.values())[0], dtype=np.float32)
        
        
#         xgb_reg_r2 = r2_score(dataset_costs, xgb_all_preds)
#         reg_history.append(round(xgb_reg_r2, 4))
#         print(f"XGB Reg R2 : {xgb_reg_r2:.4f}")

#         # xgb_rank_r2 = pair_accuracy(xgb_all_preds, dataset_costs)
#         # rank_history.append(round(xgb_rank_r2, 4))
#         # print(f"XGB Rank R2 : {xgb_rank_r2:.4f}")

#         recall_score = recall_at_k(torch.tensor(xgb_all_preds), torch.tensor(dataset_costs), k=10)        
#         print(f"XGB Recall@{top_k} : {recall_score}")
        
        
        
        
        
#         # 다음 측정할 샘플 선택
#         train_indices, test_indices = xgb_select_indices(xgb_all_preds, 
#                             train_indices, test_indices, topk_size=topk_size, eps_greedy_size=eps_greedy_size, rng=sample_rng)
#         measured_optimum = True if real_optimum_idx in train_indices else False

#         use_topk = False
        

#         break_signal = False
#         if not use_topk and measured_optimum:
#             break_signal = True
            
#         elif use_topk and recall_score:
#             break_signal = True
#             xgb_filename= xgb_filename.replace("result_xgb/", "result_xgb_topk/")


#         if break_signal:
#         # if recall_score:
#             print("XGB 최적화 종료 신호 감지")
#             print(f"총 측정 시간: {time.time() - tic:.2f} 초")
#             print("=============================================")
#             xgb_results.append({
#                 "measure_size": measure_size,
#                 "phase" : phase,
#                 "used_time": round(time.time() - tic, 2),
#                 "train_size" : len(train_indices) - measure_size,
#                 "val_reg_r2": reg_history,
#                 "val_rank_r2": rank_history,
#                 "sampling_seed": seed,
                
#             })
#             df_xgb_results = pd.DataFrame(xgb_results)
#             os.makedirs(os.path.dirname(xgb_filename), exist_ok=True)
#             df_xgb_results.to_csv(xgb_filename, index=False)
#             # raise KeyboardInterrupt
#             break
        
#         if test_indices.shape[0] < measure_size:
#             print("측정할 샘플이 더 이상 남아있지 않음")
#             xgb_results.append({
#                 "measure_size": measure_size,
#                 "phase" : "all but not found",
#                 "used_time": round(time.time() - tic, 2),
#                 "train_size" : len(train_indices) - measure_size,
#                 "val_reg_r2": reg_history,
#                 "val_rank_r2": rank_history,
#                 "sampling_seed": seed,
                
#             })
#             df_xgb_results = pd.DataFrame(xgb_results)
#             os.makedirs(os.path.dirname(xgb_filename), exist_ok=True)
#             df_xgb_results.to_csv(xgb_filename, index=False)
#             break
#             # raise KeyboardInterrupt



# # In[27]:


# group_cols = [
#     "measure_size",
# ]

# agg_dict = {
#     # "phase": "mean",
#     "train_size": "mean",
#     "used_time": "mean",
#     "val_reg_r2": "first",
#     "val_rank_r2": "first",
# }

# df_avg = (
#     df_xgb_results
#     .groupby(group_cols, as_index=False)
#     .agg(agg_dict)
# )
# df_avg


# # In[50]:


# import xgboost as xgb
# import multiprocessing

# topk_size = int(measure_size * 0.95)
# eps_greedy_size = measure_size - topk_size


# seeds = sampling_hyper["seed"]
# random_indices = random_indices_list[:len(seeds)]

# xgb_results = []

# # XGBModelInternal과 동일한 xgb_params 설정
# xgb_params = {
#     "max_depth": 6,
#     "gamma": 0.003,
#     "min_child_weight": 2,
#     "eta": 0.2,
#     "n_gpus": 0,
#     "nthread": multiprocessing.cpu_count() // 2,
#     "verbosity": 0,
#     "seed": train_seed or 43,
#     "disable_default_eval_metric": 1,
# }

# for i, seed in enumerate(seeds):

#     tic = time.time()
#     sample_rng = np.random.default_rng(seed)

#     dataset = make_xgb_datasets(inputs, results)
    
#     used_indices = set(random_indices[i])
#     remaining_indices = set(all_indices)
#     remaining_indices.difference_update(used_indices)

#     train_indices = np.array(sorted(used_indices), dtype=np.int64)
#     test_indices = np.array(sorted(remaining_indices), dtype=np.int64)
#     print(train_indices)

#     reg_history = []
#     rank_history = []

#     for phase in range(1,  len(input_data_scaled) // measure_size + 1):

#         print(f"=============== 측정 Phase {phase} ================")

#         seed_everything(train_seed)
#         _, _, dataset_costs = split_xgb_datasets(dataset, train_indices, test_indices)
#         input_train = input_data_scaled[train_indices]
#         label_train = dataset_costs[train_indices]
#         input_test = input_data_scaled[test_indices]
#         label_test = dataset_costs[test_indices]
        
#         real_optimum_idx = np.argmax(dataset_costs)
        
#         # XGB 모델 학습 - input_train, label_train 사용
#         seed_everything(train_seed)
#         dtrain = xgb.DMatrix(input_train, label=label_train)
#         dtest = xgb.DMatrix(input_test, label=label_test)
        
#         # 학습 (XGBModelInternal과 유사하게 num_boost_round=300, early stopping 없이 단순화)
#         bst = xgb.train(
#             params=xgb_params,
#             dtrain=dtrain,
#             num_boost_round=300,
#             evals=[(dtrain, "train"), (dtest, "test")],
#             verbose_eval=50,
#         )
        
#         # input_data_scaled 전체로 predict
#         dmatrix_all = xgb.DMatrix(input_data_scaled)
#         xgb_all_preds = bst.predict(dmatrix_all)
#         xgb_all_preds = np.array(xgb_all_preds, dtype=np.float32)
        
        
#         xgb_reg_r2 = r2_score(dataset_costs, xgb_all_preds)
#         reg_history.append(round(xgb_reg_r2, 4))
#         print(f"XGB Reg R2 : {xgb_reg_r2:.4f}")

#         # xgb_rank_r2 = pair_accuracy(xgb_all_preds, dataset_costs)
#         # rank_history.append(round(xgb_rank_r2, 4))
#         # print(f"XGB Rank R2 : {xgb_rank_r2:.4f}")

#         recall_score = recall_at_k(torch.tensor(xgb_all_preds), torch.tensor(dataset_costs), k=10)        
#         print(f"XGB Recall@{top_k} : {recall_score}")
        
        
        
        
        
#         # 다음 측정할 샘플 선택
#         train_indices, test_indices = xgb_select_indices(xgb_all_preds, 
#                             train_indices, test_indices, topk_size=topk_size, eps_greedy_size=eps_greedy_size, rng=sample_rng)
#         measured_optimum = True if real_optimum_idx in train_indices else False

#         use_topk = False
        

#         break_signal = False
#         if not use_topk and measured_optimum:
#             break_signal = True
            
#         elif use_topk and recall_score:
#             break_signal = True
#             xgb_filename= xgb_filename.replace("result_xgb/", "result_xgb_topk/topk_")


#         if break_signal:
#             print("XGB 최적화 종료 신호 감지")
#             print(f"총 측정 시간: {time.time() - tic:.2f} 초")
#             print("=============================================")
#             xgb_results.append({
#                 "measure_size": measure_size,
#                 "phase" : phase,
#                 "used_time": round(time.time() - tic, 2),
#                 "train_size" : len(train_indices) - measure_size,
#                 "val_reg_r2": reg_history,
#                 "val_rank_r2": rank_history,
#                 "sampling_seed": seed,
                
#             })
#             df_xgb_results = pd.DataFrame(xgb_results)
#             # df_xgb_results.to_csv(xgb_filename, index=False)
#             break
        
#         if test_indices.shape[0] < measure_size:
#             print("측정할 샘플이 더 이상 남아있지 않음")
#             xgb_results.append({
#                 "measure_size": measure_size,
#                 "phase" : "all but not found",
#                 "used_time": round(time.time() - tic, 2),
#                 "train_size" : len(train_indices) - measure_size,
#                 "val_reg_r2": reg_history,
#                 "val_rank_r2": rank_history,
#                 "sampling_seed": seed,
                
#             })
#             df_xgb_results = pd.DataFrame(xgb_results)
#             # df_xgb_results.to_csv(xgb_filename, index=False)
#             break
#             # raise KeyboardInterrupt


# # In[124]:


# df_xgb_results


# # In[52]:


# group_cols = [
#     "measure_size",
# ]

# agg_dict = {
#     # "phase": "mean",
#     "train_size": "mean",
#     "used_time": "mean",
#     "val_reg_r2": "first",
#     "val_rank_r2": "first",
# }

# df_avg = (
#     df_xgb_results
#     .groupby(group_cols, as_index=False)
#     .agg(agg_dict)
# )
# df_avg


# # In[ ]:


# from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal

# for i in range(1000):

#     tenset_model = XGBModelInternal()
#     tenset_model.fit_base(train_set, valid_set=test_set)
#     throughputs = np.array(list(test_set.throughputs.values()))

#     pred = tenset_model.predict(test_set)

#     true_biggest_index = np.argsort(throughputs[0])[-1]
#     biggest_indices_64 = np.argsort(list(pred.values())[0])[-64:]

#     # list(pred.values())[0]
#     if true_biggest_index in biggest_indices_64:
#         print("✓ Tenset 모델이 실제 가장 높은 throughput 정확히 예측했습니다!")
#         break
#     break


# # pred, throughputs rank accuracy
# correct_pairs = 0
# total_pairs = 0
# n_samples = min(2000, throughputs.shape[-1])
# sample_indices = np.random.choice(throughputs.shape[-1], n_samples, replace=False)
# pred_values = list(pred.values())[0]
# throughput_values = throughputs.squeeze()
# rank_accuracy = pair_accuracy(pred_values, throughput_values)
# print(f"Tenset 모델 Rank Accuracy: {rank_accuracy:.4f}")

