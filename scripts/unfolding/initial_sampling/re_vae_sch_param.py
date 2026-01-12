import numpy as np
import torch
import sys
import os


sys.path.append("/root/work/tenset/scripts")
from unfolding.extract_i_vectors import process_only_diff

json_diffs = process_only_diff("/root/work/tenset/dataset/measure_records_tenset/k80/([0bcc0b358b2b1d00bc591087e839592d,1,35,35,64,4,4,96,64,1,1,1,96,1,35,35,96],cuda).json", limit=None)
raw_input = json_diffs["diff_values"]
input_data = np.log2(raw_input+1e-8)
cost = -np.log(json_diffs["cost"])


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

input_data = np.log(json_diffs["diff_values"]+1e-8)

scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

X_train, X_val, y_train, y_val = train_test_split(
    input_data_scaled, cost, test_size=0.5, random_state=42
)

# input_data, mean, std = transform_schedule(json_diffs["diff_values"])


def config_filename(first, config, type="png"):
    os.makedirs(os.path.dirname(first), exist_ok=True)
    filename = f"{first}"
    for k, v in config.items():
        if v is not None or v > 0:
            key_name = k.replace("lambda_", "")
            if len(key_name) > 3:
                key_name = key_name[:3]
            filename += f"_{key_name}{v}"

    filename += f".{type}"
    return filename



from torch.utils.data import DataLoader
sys.path.append("/root/work/tenset/scripts")
from unfolding.torch.models.vae_cost import VAE_cost
import pandas as pd

from sklearn.metrics import r2_score
import torch
sys.path.append("/root/work/tenset/scripts")
from unfolding.torch.dataset import X_y_regression_Dataset
from unfolding.util_modules.loss import reconstruction_loss, kld_loss, infonce_loss, reg_loss, pair_loss, smooth_loss


import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = X_y_regression_Dataset(X_train, y_train)
val_dataset   = X_y_regression_Dataset(X_val,   y_val)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)



input_dim = X_train.shape[-1]
latent_dim = 64
hidden_dim = 256

hyperparams = {
    'epochs': [1000, 2000],
    'lr': [5e-5, 1e-4],
    # "lambda_recon": [1.0],
    "lambda_kld": [0.0, 0.01, 0.1],
    "lambda_reg": [0.01, 0.1, 0.5],
    "lambda_pair": [0.0, 0.1, 0.5, 1.0],
    "lambda_smooth": [0.0, 0.01, 0.1],
    "lambda_infonce": [0.0, 0.01, 0.1, 0.2],
}

results_interpolation_geo_gg = []
results_interpolation_geo_gb = []
results_interpolation_geo_bb = []
results_interpolation_search = []
results_train = []


for idx, config_tup in enumerate(itertools.product(*hyperparams.values())):
    
    config = dict(zip(hyperparams.keys(), config_tup))
    print("Training with config:", config)

    geometry_fig_name_umap = config_filename("results_plot/umap/geo", config, type="png")
    if os.path.exists(geometry_fig_name_umap):
        print(f"Skipping existing config: {geometry_fig_name_umap.replace(".png","")}")
        continue

    print(f"=== Hyperparameter set {idx+1} / {np.prod([len(v) for v in hyperparams.values()])} ===")

    # config 딕셔너리 생성

    vae = VAE_cost(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=config['lr'])


    for epoch in range(1, config['epochs'] + 1):
        # 여기서는 그냥 전체를 한 번에 돌린다고 가정 (실제로는 DataLoader로 배치 쪼개기)
        vae.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)  # (N, D)
            y_batch = y_batch.to(device)  # (N,)

            # x_cont, _, _ = transform_schedule(x_batch, mean=mean, std=std)  # (N, 2D)

            x_recon, mu, logvar, z, cost_pred = vae(x_batch)
            loss = 0

            recon_loss = reconstruction_loss(x_recon, x_batch)
            kl = config['lambda_kld'] * kld_loss(mu, logvar)
            cost_loss = config['lambda_reg'] * reg_loss(cost_pred, y_batch.view(-1))

            loss += recon_loss
            loss += kl
            loss += cost_loss

            loss += config['lambda_pair'] * pair_loss(cost_pred, y_batch.view(-1))
            loss += config['lambda_smooth'] * smooth_loss(vae, z, noise_std=0.01)
            loss += config['lambda_infonce'] * infonce_loss(z, y_batch.view(-1))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        vae.eval()

        cost_preds = []
        cost_trues = []
        x_batch_all = []
        x_recon_all = []

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_recon, mu, logvar, z, cost_pred = vae(x_batch)

            val_recon_loss = reconstruction_loss(x_recon, x_batch)
            val_kl = config['lambda_kld'] * kld_loss(mu, logvar)
            val_cost_loss = config['lambda_reg'] * reg_loss(cost_pred, y_batch.view(-1))
            val_total_loss = val_recon_loss + val_kl + 0.1* val_cost_loss


            cost_preds.append(cost_pred.detach().cpu())
            cost_trues.append(y_batch.view(-1).detach().cpu())
            x_batch_all.append(x_batch.detach().cpu())
            x_recon_all.append(x_recon.detach().cpu())

        if epoch % config['epochs'] == 0:
            print(f"epoch {epoch}: loss={loss.item():.4f}, recon={recon_loss.item():.4f}, kl={kl.item():.4f}, cost={cost_loss.item():.4f}")
            print(f"epoch {epoch}: val loss={val_total_loss.item():.4f}, val recon={val_recon_loss.item():.4f}, val kl={val_kl.item():.4f}, val cost={val_cost_loss.item():.4f}")
            
            cost_trues = torch.cat(cost_trues).numpy()
            cost_preds = torch.cat(cost_preds).numpy()
            val_recon_r2 = r2_score(torch.cat(x_batch_all).numpy(), torch.cat(x_recon_all).numpy())
            val_reg_r2 = r2_score(cost_trues, cost_preds)
            print(f"Validation Cost R2: {val_recon_r2:.4f}, Reconstruction R2: {val_reg_r2:.4f}")
    from unfolding.util_modules.plot.interpolation_geometry import plot_interpolation_geometry
    from unfolding.util_modules.plot.interpolation_search import plot_interpolation_search
    from unfolding.util_modules.plot.latent_cost_geometry import plot_latent_cost_geometry
    inter_geometry_summary = plot_interpolation_geometry(vae, X_train, y_train, X_val, y_val, device, type="latent", only_summary=True) # inter_geometry_summary : dict
    inter_search_summary = plot_interpolation_search(vae, X_train, y_train, X_val, y_val, device, type="latent", only_summary=True) # inter_search_summary : dict

    results_interpolation_geo_gg.append({
        "config": config,
        "summary": inter_geometry_summary['good_good'],
    })
    results_interpolation_geo_gb.append({
        "config": config,
        "summary": inter_geometry_summary['good_bad'],
    })
    results_interpolation_geo_bb.append({
        "config": config,
        "summary": inter_geometry_summary['bad_bad'],
    })
    results_interpolation_search.append({
        "config": config,
        "summary": inter_search_summary,
    })
    results_train.append({
        "config": config,
        "val_recon_r2": val_recon_r2,
        "val_reg_r2": val_reg_r2,
    })

    # pandas DataFrame으로 저장
    df_geo_gg = pd.DataFrame([{**r["config"], **r["summary"]} for r in results_interpolation_geo_gg])
    df_geo_gb = pd.DataFrame([{**r["config"], **r["summary"]} for r in results_interpolation_geo_gb])
    df_geo_bb = pd.DataFrame([{**r["config"], **r["summary"]} for r in results_interpolation_geo_bb])

    df_search = pd.DataFrame([{**r["config"], **r["summary"]} for r in results_interpolation_search])
    df_train = pd.DataFrame([{**r["config"], "val_recon_r2": r["val_recon_r2"], "val_reg_r2": r["val_reg_r2"]} for r in results_train])
    df_geo_gg.to_csv("results/interpolation_geometry_gg.csv", index=False)
    df_geo_gb.to_csv("results/interpolation_geometry_gb.csv", index=False)
    df_geo_bb.to_csv("results/interpolation_geometry_bb.csv", index=False)
    df_search.to_csv("results/interpolation_search.csv", index=False)
    df_train.to_csv("results/train.csv", index=False)
    

    with torch.no_grad():
        x_recon, mu, logvar, z, cost_pred = vae(torch.tensor(X_val).float().to(device))
    geometry_fig_name_pca = config_filename("results_plot/pca/geo", config, type="png")
    plot_latent_cost_geometry(Z=z.detach().cpu().numpy(), cost=y_val, method="pca", seed=0, save_path=geometry_fig_name_pca, show=False)
    geometry_fig_name_umap = config_filename("results_plot/umap/geo", config, type="png")
    plot_latent_cost_geometry(Z=z.detach().cpu().numpy(), cost=y_val, method="umap", seed=0, save_path=geometry_fig_name_umap, show=False)