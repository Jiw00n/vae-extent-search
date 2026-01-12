import pandas as pd
import itertools
import os


class CSVLogger:
    def __init__(self, filename, top_k=1):
        self.filename = filename.replace(".json", "")
        self.top_k = top_k
        self.all_results = []


    def append_result(self, hyper_config, phase, used_time,
                    topk_recall_signal, reg_history, rank_history):

        # 1) hyper_config를 그대로 복사 (추가되는 하이퍼 자동 반영)
        record = dict(hyper_config)


        # 3) 타입 정리 (groupby/비교 안정성)
        if "weights" in record:
            record["weights"] = str(record["weights"])

        # 4) 결과/로그 컬럼 추가
        record.update({
            "phase": phase,
            "train_size": phase * hyper_config["measure_size"],
            "used_time": round(used_time, 2),
            f"top-{self.top_k}": topk_recall_signal,
            "val_reg_r2": reg_history,
            "val_rank_r2": rank_history,
        })

        self.all_results.append(record)

        # 5) DataFrame 갱신 (지금 구조 유지: 매번 재생성)
        self.df_results = pd.DataFrame(self.all_results)

        # 6) 컬럼 순서 정리: 지정한 것들만 맨 뒤로
        tail_cols = ["used_time", "val_reg_r2", "val_rank_r2", "sampling_seed"]
        tail_cols = [c for c in tail_cols if c in self.df_results.columns]
        front_cols = [c for c in self.df_results.columns if c not in tail_cols]
        self.df_results = self.df_results[front_cols + tail_cols]



    def save_result(self):            
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.df_results.to_csv(self.filename, index=True)



    def save_result_avg(self):
        ignore_cols = {
            "phase",
            "train_size",
            "used_time",
            f"top-{self.top_k}",
            "val_reg_r2",
            "val_rank_r2",
            "sampling_seed",  # seed는 그룹키가 아니라 집계 대상
        }

        group_cols = [c for c in self.df_results.columns if c not in ignore_cols]

        agg_dict = dict(
            phase=("phase", "mean"),
            train_size=("train_size", "mean"),
            used_time=("used_time", "mean"),
            val_reg_r2=("val_reg_r2", "first"),
            val_rank_r2=("val_rank_r2", "first"),
            seed_n=("sampling_seed", "nunique"),
            sampling_seed=("sampling_seed", list),
        )

        topk_col = f"top-{self.top_k}"
        if topk_col in self.df_results.columns:
            agg_dict[topk_col] = (topk_col, "mean")

        df_avg = (
            self.df_results
            .groupby(group_cols, as_index=False, dropna=False)
            .agg(**agg_dict)
        )

        # used_time, val_reg_r2, val_rank_r2, seed_n, sampling_seed를 맨 뒤로
        tail_cols = ["used_time", "val_reg_r2", "val_rank_r2", "seed_n", "sampling_seed"]
        tail_cols = [c for c in tail_cols if c in df_avg.columns]
        front_cols = [c for c in df_avg.columns if c not in tail_cols]
        df_avg = df_avg[front_cols + tail_cols]

        df_avg.to_csv(self.filename.replace(".csv", "_avg.csv"), index=False)




    def filter_already_measured(self, total_csv_path, sampling_hyper):
        ignore_cols = {
            "phase",
            "train_size",
            "used_time",
            f"top-{self.top_k}",
            "val_reg_r2",
            "val_rank_r2",
        }

        # 파일이 없을 때도 compare_cols는 정의돼야 함
        compare_cols = [c for c in sampling_hyper.keys() if c not in ignore_cols]

        measured_keys = set()

        if total_csv_path is not None and os.path.exists(total_csv_path):
            total_csv = pd.read_csv(total_csv_path)

            # CSV 기준으로 비교 컬럼 재정의 (더 정확)
            compare_cols = [c for c in total_csv.columns if c not in ignore_cols]

            measured_keys = {
                frozenset((col, str(row.get(col))) for col in compare_cols)
                for _, row in total_csv.iterrows()
            }

        to_measure_configs = []

        for params in itertools.product(*sampling_hyper.values()):
            hyper_config = dict(zip(sampling_hyper.keys(), params))

            key = frozenset(
                (col, str(hyper_config.get(col)))
                for col in compare_cols
            )

            if key in measured_keys:
                continue

            to_measure_configs.append(hyper_config)

        return to_measure_configs
