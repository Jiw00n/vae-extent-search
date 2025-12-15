import os
ppath = os.environ.get("PYTHONPATH")
buildpath = os.environ.get("TVM_LIBRARY_PATH")
gdb_mode = os.environ.get("TVM_GDB_MODE")
use_ncu = os.environ.get("USE_NCU")

# breakpoint()
import sys

from models.dataset import load_dataset
import util_manager
if gdb_mode == "1":
    gdb_manager = util_manager.GDBManager()
    gdb_manager.gdb_log_set()

print("="*80)
print("PYTHONPATH :", ppath)
print("TVM_LIBRARY_PATH :", buildpath)
print("USE_NCU :", use_ncu)

if buildpath.endswith("build"):
    print("DEBUG MODE")
elif buildpath.endswith("release"):
    print("RELEASE MODE")
else:
    AssertionError("Set Environment release/debug")

import numpy as np
import torch
from util_manager import PathManager, get_network, get_arg, seed_everything
from tuning import make_states, select_features
from trainer import VAE_Trainer, Regression_Trainer
from models.dataset import load_dataset
import tvm
from tvm import relay, auto_scheduler
# from tvm.contrib import graph_executor
import argparse
from tvm.auto_scheduler.measure import ProgramMeasurer
from tvm.auto_scheduler.search_policy import SketchPolicy
from tvm.auto_scheduler.feature import get_per_store_features_from_measure_pairs, get_per_store_features_from_states
sys.path.append("/root/work/tenset/scripts")

import random


TARGET = tvm.target.Target("cuda")

seed_everything(42)

def get_tasks(mod, params, path_manager, verbose=True, get_pkl=True):
    if get_pkl:
        tasks, task_weights = path_manager.tasks_pkl_use()
    
    if get_pkl is False or tasks is None:
        print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, TARGET)
        if path_manager.tasks_pkl_check() is False:
            path_manager.tasks_pkl_save(tasks, task_weights)

    if verbose:
        for idx, task in enumerate(tasks):
            print("========== Task %d : (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)
            # breakpoint()
    
    print(f"Total tasks length : {len(tasks)}")
    # breakpoint()
    return tasks, task_weights




def run_tuning(tasks, task_weights, paths):
    print("="*80)
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10000)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(paths["json"])],
    )
    measurer = ProgramMeasurer(
                tune_option.builder,
                tune_option.runner,
                tune_option.measure_callbacks,
                tune_option.verbose,
                )
    task = tasks[0]
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    state_size = 2000
    sample_size = 64
    
    measure_inputs, measure_results = make_states(paths["json"], task, size=state_size, max_retry_iter=10)
    features_tuple = get_per_store_features_from_measure_pairs(measure_inputs, measure_results)
    features = features_tuple[0]  # (N, max_n_buf, feature_len) 형태
    
    train_loader, val_loader, _ = load_dataset(features, measure_results, type='vae', test_size=0.1)

    #train_vae
    vae_trainer = VAE_Trainer(train_loader, val_loader)
    pretrained_vae = vae_trainer.train_vae(epochs=200, lr=2e-4, beta=1e-4, patience=30, verbose=False)

    # select states & train cost model
    reg_trainer = Regression_Trainer(pretrained_vae)

    used_indices = set()
    all_measured_results = []  # 전체 측정 결과를 누적 저장
    fea_norm_vec = None  # 정규화 벡터 (첫 phase에서 초기화)
    
    for phase in range(state_size//sample_size):
        print(f"\n{'='*80}")
        print(f"Phase {phase}: 측정 및 학습 시작")
        print(f"{'='*80}")
        
        if phase == 0:
            # Phase 0: 랜덤으로 인덱스 sample_size개 선택
            feature_batch_indices = random.sample(range(state_size), sample_size)
        else:
            # Phase 1~: select_features를 통해 선택
            feature_batch_indices, _ = select_features(reg_trainer.model, features, used_indices)

        inp_batch = [measure_inputs[i] for i in feature_batch_indices]

        # 측정
        res_batch = measurer.measure(task, empty_policy, inp_batch)
        print(f"Phase {phase}: {len(res_batch)}개 샘플 측정 완료")

        used_indices.update(feature_batch_indices)
        all_measured_results.extend(res_batch)  # 전체 결과에 누적
        
        # 학습: 전체 측정 데이터를 사용
        print(f"Phase {phase}: 전체 {len(all_measured_results)}개 샘플로 학습 시작")
        
        # 전체 측정된 인덱스에 해당하는 features 수집
        measured_features = [features[i] for i in sorted(used_indices)]
        measured_segment_sizes = np.array([f.shape[0] for f in measured_features], dtype=np.int32)
        measured_costs = np.array([np.mean([c.value for c in r.costs]) for r in all_measured_results], dtype=np.float32)
        
        # normalize vector 계산 (첫 phase에서만)
        if phase == 0:
            flatten_features = np.concatenate(measured_features, axis=0).astype(np.float32)
            fea_norm_vec = torch.ones((flatten_features.shape[1],))
            for i in range(flatten_features.shape[1]):
                max_val = flatten_features[:, i].max()
                if max_val > 0:
                    fea_norm_vec[i] = max_val
        
        # regression 학습 설정
        # Phase 1부터는 mini_epochs를 늘려서 새 데이터를 더 많이 학습
        config = {
            'encoder_lr': 1e-4,
            'predictor_lr': 1e-3,
            'lambda_pair': 0.1,
            'gamma': 0.01,
            'beta': 0.001,
            'mini_epochs': 20 if phase == 0 else 30,  # Phase 1부터 더 많이 학습
            'loss_type': 'mse',
            'margin': 0.1,
            'noise_std': 0.1
        }
        
        reg_trainer.train_regression(
            measured_features, measured_segment_sizes, measured_costs,
            fea_norm_vec, config, phases=1
        )
        




def main_(args):

    network = args.network
    batch_size = args.batch_size
    layout = args.layout
    dtype = "float32"

    # resnet_18, resnet_50
    network = "resnet_18"
    # batch_size = 1
    # layout = "NHWC"
    


    # 네트워크 불러오기
    mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)


    # 경로 설정
    path_manager = PathManager(network, input_shape, args, gdb_mode)
    # path_manager = PathManager(network, input_shape, args, gdb_mode, json="/root/work/tvm-ansor/gallery/logs_json/resnet_18/resnet_18-B1.json")


    # task 추출
    tasks, task_weights = get_tasks(mod, params, path_manager, verbose=True, get_pkl=True)

    
    # 튜닝
    run_tuning(tasks, task_weights, path_manager.paths)




if __name__ == "__main__":
    print("="*80)
    parser = argparse.ArgumentParser(description="Ansor CUDA")
    args = get_arg(parser)

    main_(args)
