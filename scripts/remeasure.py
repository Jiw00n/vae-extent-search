"""Remeasure all programs from a specific to_measure_programs file

This script remeasures all records in a specific JSON file:
/root/work/tenset/dataset/to_measure_programs/([0bcb8746286db050cd088f375c85372d,1,64,64,128,6,6,32,128,1,64,64,32],cuda).json

Usage:
python3 remeasure.py
python3 remeasure.py --batch-size 64
python3 remeasure.py --run-timeout 10 --repeat 3
"""

import argparse
import os
import time

import tvm
from tvm import auto_scheduler

from common import (
    load_and_register_tasks,
    get_measure_record_filename,
    MEASURE_RECORD_FOLDER,
    TO_MEASURE_PROGRAM_FOLDER,
    clean_name,
)


# Target file to remeasure
TARGET_WORKLOAD_KEY = '["0bcb8746286db050cd088f375c85372d", 1, 64, 64, 128, 6, 6, 32, 128, 1, 64, 64, 32]'
TARGET_KIND = "cuda"
TARGET_FILENAME = f"{TO_MEASURE_PROGRAM_FOLDER}/([0bcb8746286db050cd088f375c85372d,1,64,64,128,6,6,32,128,1,64,64,32],cuda).json"


def make_measurer(run_timeout, repeat, number, enable_cpu_cache_flush, verbose, log_filename):
    """Create a measurer for program measurement."""
    builder = auto_scheduler.measure.LocalBuilder()
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout,
        repeat=repeat,
        number=number,
        enable_cpu_cache_flush=enable_cpu_cache_flush,
    )
    measurer = auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        [auto_scheduler.RecordToFile(log_filename)],
        verbose=verbose,
    )
    return measurer


def remeasure_file(target, target_host, batch_size, measurer_kwargs, max_records=None):
    """Remeasure all records in the target file."""
    
    # Check if target file exists
    if not os.path.exists(TARGET_FILENAME):
        raise FileNotFoundError(f"Target file not found: {TARGET_FILENAME}")
    
    print(f"Reading records from: {TARGET_FILENAME}")
    
    # Read reference measurement inputs
    inputs, _ = auto_scheduler.RecordReader(TARGET_FILENAME).read_lines()
    print(f"Total records to remeasure: {len(inputs)}")
    
    if max_records is not None:
        inputs = inputs[:max_records]
        print(f"Limiting to first {max_records} records for testing")
    
    # Recover task from the first input
    recovered_input = auto_scheduler.measure.recover_measure_input(inputs[0])
    original_task = recovered_input.task
    
    # Create new task with specified target
    target_obj = tvm.target.Target(target)
    task = auto_scheduler.SearchTask(
        workload_key=original_task.workload_key,
        target=target_obj,
        target_host=target_host,
        hardware_params=original_task.hardware_params,
        layout_rewrite_option=original_task.layout_rewrite_option,
    )
    
    print(f"Task workload_key: {task.workload_key}")
    print(f"Task target: {task.target}")
    
    # Make output folder and log filename
    task_key = (task.workload_key, str(target_obj.kind))
    output_dir = f"{MEASURE_RECORD_FOLDER}/{target_obj.model}"
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"{output_dir}/{clean_name(task_key)}.json"
    
    print(f"Output log file: {log_filename}")
    
    # Create measurer
    measurer_kwargs['log_filename'] = log_filename
    measurer = make_measurer(**measurer_kwargs)
    
    # Create empty policy for measurement
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)
    
    # Do measurement in batches
    total_measured = 0
    total_timeout = 0
    total_error = 0
    
    for i in range(0, len(inputs), batch_size):
        batch_end = min(len(inputs), i + batch_size)
        print(f"\n===== Measuring programs: {i+1}-{batch_end}/{len(inputs)} =====")
        
        # Prepare batch inputs
        inp_batch = []
        for inp in inputs[i:batch_end]:
            inp_batch.append(auto_scheduler.MeasureInput(task, inp.state))
        
        # Measure batch
        start_time = time.time()
        res_batch = measurer.measure(task, empty_policy, inp_batch)
        elapsed = time.time() - start_time
        
        # Count results
        timeout_ct = 0
        error_ct = 0
        success_ct = 0
        
        for res in res_batch:
            if res.error_no == auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT:
                timeout_ct += 1
            elif res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                error_ct += 1
            else:
                success_ct += 1
        
        total_timeout += timeout_ct
        total_error += error_ct
        total_measured += len(res_batch)
        
        print(f"Batch results: success={success_ct}, timeout={timeout_ct}, error={error_ct}")
        print(f"Batch time: {elapsed:.2f}s")
    
    print(f"\n===== Measurement Complete =====")
    print(f"Total measured: {total_measured}")
    print(f"Total timeouts: {total_timeout}")
    print(f"Total errors: {total_error}")
    print(f"Results saved to: {log_filename}")
    
    return log_filename


def main():
    parser = argparse.ArgumentParser(description="Remeasure programs from target file")
    parser.add_argument("--target", type=str, default="cuda", 
                        help="Target for measurement (default: cuda)")
    parser.add_argument("--target-host", type=str, default=None,
                        help="Target host")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for measurement (default: 64)")
    parser.add_argument("--run-timeout", type=int, default=10,
                        help="Run timeout in seconds (default: 10)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of repeat measurements (default: 3)")
    parser.add_argument("--number", type=int, default=1,
                        help="Number of runs per repeat (default: 1)")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum number of records to measure (for testing)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (default: 1)")
    args = parser.parse_args()
    
    # Load and register tasks first
    print("Loading and registering tasks...")
    tasks = load_and_register_tasks()
    print(f"Loaded {len(tasks)} tasks")
    
    # Set measurement arguments
    measurer_kwargs = {
        "run_timeout": args.run_timeout,
        "repeat": args.repeat,
        "number": args.number,
        "enable_cpu_cache_flush": False,  # For CUDA, this is typically False
        "verbose": args.verbose,
    }
    
    print(f"\nMeasurement settings:")
    print(f"  Target: {args.target}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Run timeout: {args.run_timeout}s")
    print(f"  Repeat: {args.repeat}")
    print(f"  Number: {args.number}")
    
    # Run remeasurement
    log_filename = remeasure_file(
        target=args.target,
        target_host=args.target_host,
        batch_size=args.batch_size,
        measurer_kwargs=measurer_kwargs,
        max_records=args.max_records,
    )
    
    # Verify results
    print(f"\n===== Verification =====")
    if os.path.exists(log_filename):
        with open(log_filename, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"Output file exists with {line_count} records")
    else:
        print("WARNING: Output file was not created!")


if __name__ == "__main__":
    main()
