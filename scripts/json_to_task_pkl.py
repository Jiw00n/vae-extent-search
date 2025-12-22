import pickle
from common import load_and_register_tasks
from tvm.auto_scheduler.feature import get_per_store_features_from_file

# network_info_tenset을 사용해야 해당 workload가 등록됨
tasks = load_and_register_tasks("/root/work/tenset/dataset/network_info_tenset")
# 실제 파일 경로: measure_records_tenset에 있음
local_json_file = "/root/work/tenset/dataset/measure_records_tenset/unknown/([0bcb8746286db050cd088f375c85372d,1,64,64,128,6,6,32,128,1,64,64,32],cuda).json"
raw_features, raw_normalized_throughputs, task_ids, min_latency = get_per_store_features_from_file(local_json_file, 10000)

print(f"raw_features shape: {raw_features.shape}")
print(f"raw_normalized_throughputs shape: {raw_normalized_throughputs.shape}")
print(f"task_ids shape: {task_ids.shape}")
print(f"min_latency shape: {min_latency.shape}")