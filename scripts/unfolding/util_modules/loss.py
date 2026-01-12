import torch
import torch.nn.functional as F



def reconstruction_loss(x_recon, x):
    """
    VAE 재구성 손실 (MSE)
    """
    return F.mse_loss(x_recon, x, reduction="mean")


def kld_loss(mean, logvar):
    """
    KL Divergence: q(z|x) || N(0, I)
    """
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return kld




def reg_loss(cost_pred, cost_true, loss_type='mse'):
    """
    기본 회귀 손실 (MSE 또는 MAE)
    """
    if cost_pred.shape != cost_true.shape:
        if cost_true.ndim == 2 and cost_true.size(1) == 1:
            cost_true = cost_true.view(-1)
        elif cost_pred.ndim == 2 and cost_pred.size(1) == 1:
            cost_pred = cost_pred.view(-1)

    if loss_type == 'mse':
        return F.mse_loss(cost_pred, cost_true)
    else:  # mae
        return F.l1_loss(cost_pred, cost_true)


def pair_loss(cost_pred, cost_true, margin=0.1):
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





def smooth_loss(model, z, noise_std=0.1):
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




def infonce_loss(
    z: torch.Tensor,      # (B, D)
    c: torch.Tensor,      # (B,)
    tau: float = 0.1,     # similarity temperature (필수 하이퍼 1개)
    tau_c: float = None,  # cost temperature (None이면 배치에서 자동 추정)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cost-weighted InfoNCE (최소 버전)
    - 입력: 임베딩 z, 실제 cost c
    - 하이퍼: tau(유사도 temperature)만 사실상 튜닝하면 됨
    - tau_c는 None이면 배치 내 |c_i-c_j|의 median으로 자동 설정

    L_i = -log( sum_{j!=i} w_ij * exp(sim_ij/tau) / sum_{k!=i} exp(sim_ik/tau) )
    w_ij = exp(-|c_i-c_j|/tau_c)
    """
    # print(z.shape, c.shape)
    c = c.view(-1)
    assert z.dim() == 2 and c.dim() == 1
    B = z.size(0)
    if B < 2:
        return z.new_tensor(0.0)

    # cosine similarity
    z = F.normalize(z, p=2, dim=1)
    sim = z @ z.t()  # (B, B)

    # exclude diagonal
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    mask = ~eye

    # cost diff
    dc = (c.view(-1, 1) - c.view(1, -1)).abs()  # (B, B)

    # auto tau_c: median of valid pairwise diffs
    if tau_c is None:
        tau_c = dc[mask].median().clamp_min(eps).item()
    else:
        tau_c = float(tau_c)
        if tau_c <= 0:
            raise ValueError("tau_c must be > 0")

    # weights
    w = torch.exp(-dc / max(tau_c, eps)) * mask.float()  # (B, B)

    # logits
    logits = sim / tau

    # log denominator: log sum_{k!=i} exp(logits_ik)
    neg_inf = torch.finfo(logits.dtype).min
    logits_den = logits.masked_fill(~mask, neg_inf)
    den = torch.logsumexp(logits_den, dim=1)  # (B,)

    # log numerator: log sum_{j!=i} w_ij * exp(logits_ij)
    # = logsumexp(logits + log(w))
    logits_num = (logits + torch.log(w.clamp_min(eps))).masked_fill(~mask, neg_inf)
    num = torch.logsumexp(logits_num, dim=1)  # (B,)

    loss = -(num - den)  # (B,)

    # safety: in case of NaNs
    loss = loss[torch.isfinite(loss)]
    return loss.mean() if loss.numel() > 0 else z.new_tensor(0.0)



def feature_loss(use_feature, feature_pred, feature_true, coef=0.1):
    """
    Feature 예측 손실 (MSE)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_feature:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(feature_pred, feature_true) * coef