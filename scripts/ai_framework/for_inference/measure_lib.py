from tvm import auto_scheduler


def make_measurer():
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10000)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile("tmp.json")],
    )
    measurer = auto_scheduler.measure.ProgramMeasurer(
                tune_option.builder,
                tune_option.runner,
                tune_option.measure_callbacks,
                tune_option.verbose,
                )
    return measurer