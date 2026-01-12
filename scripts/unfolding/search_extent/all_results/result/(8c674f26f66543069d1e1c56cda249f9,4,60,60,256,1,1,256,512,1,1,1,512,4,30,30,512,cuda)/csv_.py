from glob import glob
import pandas as pd



csv_dir = glob(
    "./**/*.csv", recursive=True
)

csvs = []
for csv in csv_dir:
    if "avg" in csv or "total" in csv or "sampling" in csv or "prev" in csv:
       continue
    csvs.append(csv) 

dfs = []
for p in csvs:
    sub_df = pd.read_csv(p)

    sub_df = sub_df.loc[:, ~sub_df.columns.str.startswith("Unnamed")]

    if "rank_warmup_epochs" not in sub_df.columns:
        sub_df["rank_warmup_epochs"] = 0
    if "measure_size" not in sub_df.columns:
        sub_df["measure_size"] = 64
    if "scratch" not in sub_df.columns:
        sub_df["scratch"] = False
    if "encoder_freeze" not in sub_df.columns:
        sub_df["encoder_freeze"] = False
    if "lambda_pair" not in sub_df.columns:
        sub_df["lambda_pair"] = 2.0
    # if "T_mc" not in sub_df.columns:
    #     sub_df["T_mc"] = 20
    
    dfs.append(sub_df)

    
df_total = pd.concat(dfs, ignore_index=True)

# measure_size 컬럼을 맨 앞으로 이동
cols = df_total.columns.tolist()
df_total = df_total[["measure_size"] + [c for c in cols if c != "measure_size"]]

# T_mc drop
df_total = df_total.drop(columns=["T_mc"], errors='ignore')

df_total.to_csv("./vae_extent_total.csv", index=True)



agg_kwargs = {
    "phase": ("phase", "mean"),
    "train_size": ("train_size", "mean"),
    "used_time": ("used_time", "mean"),
    "val_reg_r2": ("val_reg_r2", "first"),
    "seed_n": ("sampling_seed", "nunique"),
    "sampling_seed": ("sampling_seed", list),
}

topk_col = f"top-1"
if topk_col in df_total.columns:
    agg_kwargs[topk_col] = (topk_col, "mean")

ignore_group_cols = {
    "phase",
    "train_size",
    "used_time",
    "top-1",
    "val_reg_r2",
    "val_rank_r2",
    "sampling_seed",  # seed는 집계 대상이지 그룹 기준 아님
}

group_cols = [
    c for c in df_total.columns
    if c not in ignore_group_cols
]


df_total_avg = (
    df_total
    .groupby(group_cols, as_index=False, dropna=False)
    .agg(**agg_kwargs)
)

tail_cols = ["used_time", "val_reg_r2", "seed_n", "sampling_seed"]

cols = list(df_total_avg.columns)
front_cols = [c for c in cols if c not in tail_cols]

df_total_avg = df_total_avg[front_cols + tail_cols]

# breakpoint()

df_total_avg.to_csv("./vae_extent_total_avg.csv", index=False)
