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
import numpy as np
sys.path.append("/root/work/tenset/scripts")
from print_programs import return_all_states
from make_dataset import load_and_register_tasks
from tvm import auto_scheduler
from tvm.auto_scheduler.dataset import Dataset, make_dataset_from_log_file

json_file = "/root/work/tenset/dataset/measure_records_tenset/k80/([0c9a5ba46ffc5e1a9e5641018527117f,4,7,7,160,1,1,160,960,1,1,1,960,4,7,7,960],cuda).json"
load_and_register_tasks()

states, costs = return_all_states(json_file)
records_raw = list(map(lambda x: str(x).strip(), states))

records = {"schedules": [], "extents": [], "costs": [], "unroll" : [], "all": []}

for rec, cost in zip(records_raw, costs):
    cost = np.array([c.value for c in cost])
    cost = -np.log(np.mean(cost) + 1e-8)
    schedule = rec.split("Placeholder")[-1][2:]
    
    records["schedules"].append(schedule)
    records["costs"].append(cost)
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

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

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


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

seed_everything(42)

input_data = np.log1p(np.array(records["all"], dtype=np.float32))

scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

X_train, X_val = train_test_split(
    input_data_scaled,  test_size=0.2, random_state=42
)


# feature 없음
train_dataset = FeatureDataset(X_train)
val_dataset   = FeatureDataset(X_val)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)

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
    'alpha_feature': [1.0],
    'latent_dim': [64],
    'lr': [1e-3],
}

cnt = 0
epochs = 300


for vals in itertools.product(*hyperparameter.values()):
    (beta, alpha_recon, alpha_feature, latent_dim, lr) = vals
    cnt += 1
    print("=============================================")
    print(f"Experiment {cnt}/{len(list(itertools.product(*hyperparameter.values())))}")
    print(f"beta={beta}, alpha_recon={alpha_recon}, alpha_feature={alpha_feature},\nepochs={epochs}, latent_dim={latent_dim}, hidden_dim={hidden_dim}, lr={lr}")

    seed_everything(42)

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
        
    
    def encode(self, input_data):
        """
        Full encoding path: features → z
        완전히 미분 가능
        """
                
        # VAE Encoder
        h = self.encoder(input_data)
        
        mean = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
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
    model.eval()
    with torch.no_grad():
        z_noisy = z + noise_std * torch.randn_like(z)
    
    cost_original = model.predict_cost(z)
    cost_noisy = model.predict_cost(z_noisy)
    
    smooth_loss = F.mse_loss(cost_original, cost_noisy)
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
def pair_accuracy(cost_pred, labels):
    """
    cost_pred, labels: (B,) 텐서
    """
    seed_everything(42)
    n_samples = min(2000, len(cost_pred))
    sample_indices = np.random.choice(len(cost_pred), n_samples, replace=False)

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

def xgb_select_indices(xgb_all_preds, train_indices, test_indices, topk_size, eps_greedy_size):
    """
    랜덤으로 2개, xgb 모델로 상위 62개 선택
    """
    # 남은 인덱스 중에서 무작위로 random_select_size개 선택

    remaining_indices = set(test_indices)

    top_indices, remaining_indices = select_topk_cost(xgb_all_preds, remaining_indices, topk_size)
    random_indices, remaining_indices = random_select_indices(remaining_indices, eps_greedy_size)

    test_indices = np.array(list(remaining_indices), dtype=np.int64)

    selected_indices = np.concatenate([top_indices, random_indices])

    train_indices = np.concatenate([train_indices, selected_indices])

    return train_indices, test_indices



def random_select_indices(remaining_indices, select_size):
    
    random_indices = np.random.choice(list(remaining_indices), size=select_size, replace=False)

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

    cost_pred_np = cost_pred.detach().cpu().numpy()  # [N]

    # 1) remaining 중 cost 기준 상위 1/3 후보 만들기
    remaining_np = np.array(list(remaining_indices), dtype=np.int64)
    remaining_cost = cost_pred_np[remaining_np]

    # 상위 1/3 cutoff
    k_cost = max(num_select, len(remaining_np) // 3)
    top_cost_local = np.argsort(remaining_cost)[-k_cost:]
    candidate_indices = remaining_np[top_cost_local]

    # 2) z-gradient 계산 (전체 z 기준)
    z_grad = torch.autograd.grad(
        outputs=cost_pred.sum(),
        inputs=z,
        retain_graph=False,
        create_graph=False
    )[0]  # [N, latent_dim]

    z_grad_norm = torch.norm(z_grad, dim=1).detach().cpu().numpy()  # [N]

    # 3) 후보(candidate) 중에서 grad-norm top-k
    candidate_grad = z_grad_norm[candidate_indices]
    k = min(num_select, len(candidate_indices))

    topk_local = np.argsort(candidate_grad)[-k:]
    selected_indices = candidate_indices[topk_local]

    # 4) remaining 업데이트
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


def select_programs(model, input_data_scaled, remaining_indices, num_select=64, T_mc=10, 
                    w_cost=0.5, w_unc=0.3, w_grad=0.2, rand_num=0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Active learning 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_data_scaled: 전체 input 리스트 ([N, input_dim] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합
        num_select: 선택할 샘플 수
        T_mc: MC Dropout 샘플 수
        w_cost: 예측값이 큰 샘플 가중치
        w_unc: epistemic 불확실성이 높은 샘플 가중치
        w_grad: z에 대한 cost의 gradient가 큰 샘플 가중치
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 리스트
    """

    # 합쳐서 64개 선택
    num_select = num_select - rand_num
    if num_select <= 0 and rand_num > 0:
        rand_indices, remaining_indices = random_select_indices(remaining_indices, rand_num)
        return rand_indices, remaining_indices
    select_n_cost = int(num_select * w_cost)
    select_n_grad = int(num_select * w_grad)
    select_n_unc = int(num_select * w_unc)
    if select_n_cost + select_n_grad + select_n_unc < num_select:
        select_n_cost += num_select - (select_n_cost + select_n_grad + select_n_unc)

    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32, device=device)
    
    if select_n_cost != 0 or select_n_grad != 0:
        model.eval()
        with torch.no_grad():
            z, _ = model.encode(input_tensor)
        z = z.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        cost_pred = model.predict_cost(z)
        cost_pred = cost_pred.view(-1)


    topk_cost_indices, remaining_indices = select_topk_cost(cost_pred, remaining_indices, select_n_cost)
    z_grad_indices, remaining_indices = select_topk_z_grad(z, cost_pred, remaining_indices, select_n_grad)
    uncertainty_indices, remaining_indices = select_topk_uncertainty(model, input_tensor, remaining_indices, select_n_unc, T_mc=T_mc)
    
    if rand_num:
        rand_indices, remaining_indices = random_select_indices(remaining_indices, rand_num)
        uncertainty_indices = np.concatenate([uncertainty_indices, rand_indices])

    all_selected_indices = np.concatenate([topk_cost_indices, z_grad_indices, uncertainty_indices])
    
    return all_selected_indices, remaining_indices
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

def make_xgb_datasets(inputs, results, train_indices, test_indices):
    f_inputs = []
    f_results = []
    for inp, res in zip(inputs, results):
        cost = np.mean([c.value for c in res.costs])
        if cost < 1e10:
            f_inputs.append(inp)
            f_results.append(res)
    dataset = auto_scheduler.dataset.Dataset()
    dataset.update_from_measure_pairs(f_inputs, f_results)


    raw_features = list(dataset.features.values())[0]
    raw_throughputs = list(dataset.throughputs.values())[0]

    # features_list = []  # 각 샘플의 feature (seq_len, feature_dim)
    dataset_costs = []

    for feature, throughput in zip(raw_features, raw_throughputs):

        if feature.shape[0] != 1 and throughput > 1e-8:
            # features_list.append(feature)
            dataset_costs.append(throughput)

    dataset_costs = np.array(dataset_costs, dtype=np.float32)
    train_set, test_set = dataset.random_split_within_task(train_set_ratio=0, 
                                                        train_idxs=train_indices.tolist(), 
                                                        test_idxs=test_indices.tolist())
    return train_set, test_set, dataset, dataset_costs
def make_vae_reg_model(vae, hyperparameter, input_dim, latent_dim, hidden_dim, y_std):

    cnt = 0
    for vals in itertools.product(*hyperparameter.values()):
        (lambda_reg, lambda_pair, margin_scale, gamma, beta, noise_std, 
        encoder_lr, feature_predictor_lr, cost_predictor_lr, seed, epochs) = vals
        cnt += 1
        
        print(f"Experiment {cnt}/{len(list(itertools.product(*hyperparameter.values())))}")
        print(f"lambda_reg={lambda_reg}, lambda_pair={lambda_pair}, margin_scale={margin_scale}, \
              gamma={gamma}, beta={beta}, noise_std={noise_std}\nencoder_lr={encoder_lr}, cost_predictor_lr={cost_predictor_lr}, seed={seed}, epochs={epochs}")
        config = {
                    'encoder_lr': encoder_lr,
                    'feature_predictor_lr': feature_predictor_lr,
                    'cost_predictor_lr': cost_predictor_lr,
                    'lambda_reg' : lambda_reg,
                    'lambda_pair': lambda_pair,
                    'gamma': gamma,
                    'beta': beta,
                    'margin': margin_scale * y_std,
                    'noise_std': noise_std,
                    'loss_type': 'mse',
                    'seed' : seed,
                    'epochs': epochs,
                }

        vae_cost_model = VAECostPredictor(input_dim=input_dim, 
                                    latent_dim=latent_dim, 
                                    hidden_dim=hidden_dim, 
                                    predictor_layers=2,
                                    dropout=0.1, use_feature=False).to(device)
        vae_cost_model.load_pretrained_encoder(vae.state_dict())
        optimizer = torch.optim.AdamW([
            {'params': vae_cost_model.get_encoder_params(), 'lr': config['encoder_lr']},
            {'params': vae_cost_model.get_cost_predictor_params(), 'lr': config['cost_predictor_lr']}
        ], weight_decay=1e-5)
    return vae_cost_model, optimizer, config
def train_regression(vae_cost_model, optimizer, train_loader, val_loader, input_data_scaled, costs, config):

    print("Train size :", len(train_loader.dataset))

    # all_reg_results = []

    seed_everything(config['seed'])

    

    for epoch in range(1, config['epochs']+1):
        vae_cost_model.train()
        for x_batch, labels in train_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device).squeeze(-1)
            
        
            cost_pred, mean, logvar, z = vae_cost_model(x_batch, use_mean=True)

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

                    val_loss, val_components = compute_total_loss(vae_cost_model, cost_pred, mean, logvar, z, labels, None, config)
                val_reg_r2 = r2_score(cost_pred.detach().cpu().numpy(), labels.detach().cpu().numpy())
                
                print(f"Train loss epoch {epoch} : reg={train_components['reg_loss']: .4f} rank={train_components['pair_loss']: .4f} kl={train_components['kld_loss']: .4f}")
                print(f"Val loss epoch {epoch}: reg={val_components['reg_loss']: .4f} rank={val_components['pair_loss']: .4f} kl={val_components['kld_loss']: .4f}")
                
                print(f"Regression R2 : {val_reg_r2:.4f}, ")
        
        # rank r2 계산
        vae_cost_model.eval()
        with torch.no_grad():
            if epoch % config['epochs'] == 0:
                input_data_tensor = torch.from_numpy(input_data_scaled).float().to(device)
                all_preds = vae_cost_model(input_data_tensor, use_mean=True)[0].detach().cpu().numpy()
                val_rank_r2 = pair_accuracy(all_preds, costs)
                recall_top_k = recall_at_k(torch.tensor(all_preds), torch.from_numpy(costs), k=64)
                print(f"Rank R2 : {val_rank_r2:.4f}")
                print(f"Recall@64 : {recall_top_k}")
                if recall_top_k:
                    break_signal = True
                else:
                    break_signal = False

    print("=============================================")
    # all_reg_results.append({
    #     "lambda_reg": lambda_reg,
    #     "lambda_pair": lambda_pair,
    #     "margin_scale": margin_scale,
    #     "gamma": gamma,
    #     "beta": beta,
    #     "noise_std": noise_std,
    #     "encoder_lr": encoder_lr,
    #     "feature_predictor_lr": feature_predictor_lr,
    #     "cost_predictor_lr": cost_predictor_lr,
    #     "seed": seed,
    #     "reg_r2": val_reg_r2,
    #     "rank_r2": val_rank_r2,
    #     "recall@64": recall_top_k
    # })
    return vae_cost_model, break_signal, val_reg_r2, val_rank_r2
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
weight_list = generate_weight_grid(step=0.1)
costs = np.array(records["costs"], dtype=np.float32)


seed = 43

seed_everything(seed)


hyperparameter = {
    # 'alpha': [1e-4, 1e-3],
    
    
    'lambda_reg' : [0.01],
    'lambda_pair': [3.0],
    'margin_scale': [0.3],
    'gamma': [0.01],
    'beta': [0.01],
    'noise_std': [0.001],
    # 'alpha': [1e-5],

    'encoder_lr': [1e-4],
    'feature_predictor_lr': [0],
    'cost_predictor_lr': [1e-2],
    'seed': [seed],
    'epochs': [600],
    # 'seed': range(2023, 2028),
    
}

measure_size = 32
sampling_hyper = {
    "measure_size": [16,32,64],
    "T_mc": [10, 20, 30],
    "w" : weight_list,
    "seed" : [42, 43, 44],
}




# 데이터셋 길이만큼의 인덱스 numpy 배열 생성
all_indices = np.arange(len(input_data_scaled))

# used_indices : 이미 측정된 인덱스 집합. train_indices와 동일
# remaining_indices : 아직 측정되지 않은 인덱스 집합. val_indices와 동일
used_indices = set()
remaining_indices = set(all_indices)


print("=============== 측정 Phase 1 ================")
random_indices, remaining_indices = random_select_indices(remaining_indices, select_size=measure_size)
used_indices.update(random_indices)

all_results = []

import pandas as pd


for params in itertools.product(*sampling_hyper.values()):
    tic = time.time()
    measure_size, T_mc, w, sampling_seed = params
    print(f"measure_size: {measure_size}, T_mc: {T_mc}, w: {w}, sampling_seed: {sampling_seed}")
    seed_everything(sampling_seed)

    train_loader, val_loader, y_mean, y_std = make_vae_reg_dataloaders(input_data_scaled, costs, used_indices, remaining_indices)
    vae_cost_model, optimizer, config = make_vae_reg_model(vae, hyperparameter, input_dim, latent_dim, hidden_dim, y_std)
    vae_cost_model, break_signal, val_reg_r2, val_rank_r2 = train_regression(vae_cost_model, optimizer, train_loader, val_loader, input_data_scaled, costs, config)
    all_results.append({
            "phase": 1,
            "measure_size": measure_size,
            "T_mc": T_mc,
            "w_cost": 0,
            "w_unc": 0,
            "w_grad": 0,
            'break_signal': break_signal,
            "reg_r2": val_reg_r2,
            "rank_r2": val_rank_r2,
            "sampling_seed": sampling_seed,
        })

    for phase in range(len(input_data_scaled) // measure_size):
        if break_signal:
            print("VAE 최적화 종료 신호 감지")
            print(f"총 측정 시간: {time.time() - tic:.2f} 초")
            break

        print(f"=============== 측정 Phase {phase+2} ================")
        

        # 다음 측정할 샘플 선택
        selected_indices, remaining_indices = select_programs(
            model=vae_cost_model,
            input_data_scaled=input_data_scaled,
            remaining_indices=remaining_indices,
            num_select=measure_size,
            T_mc=20,
            w_cost=0.6,
            w_grad=0.2,
            w_unc=0.2,
            rand_num=0,
            device=device
        )
        used_indices.update(selected_indices)

        # DataLoader 갱신
        train_loader, val_loader, y_mean, y_std = make_vae_reg_dataloaders(input_data_scaled, costs, used_indices, remaining_indices)
        
        # 회귀 모델 재학습
        vae_cost_model, optimizer, config = make_vae_reg_model(vae, hyperparameter, input_dim, latent_dim, hidden_dim, y_std)
        vae_cost_model, break_signal, val_reg_r2, val_rank_r2 = train_regression(vae_cost_model, optimizer, train_loader, val_loader, input_data_scaled, costs, config)

        toc = time.time()

        all_results.append({
            
            "measure_size": measure_size,
            "T_mc": T_mc,
            "w_cost": w[0],
            "w_unc": w[1],
            "w_grad": w[2],
            "phase": phase + 2,
            'break_signal': break_signal,
            "reg_r2": val_reg_r2,
            "rank_r2": val_rank_r2,
            "time_elapsed": toc - tic,
            "sampling_seed": sampling_seed,
        })
        df_results = pd.DataFrame(all_results)
        df_results.to_csv("vae_active_learning_results.csv", index=False)