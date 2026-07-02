from laboneq.simple import from_json


def submit_and_wait_for_compiled_experiment(
    handler,
    task_manager,
    profile_name="main",
    do_emulation=False,
):
    compiled_experiment = handler.get_compiled_experiment()

    task_id = task_manager.submit_compiled_experiment(
        experiment_name=handler.experiment_name,
        profile_name=profile_name,
        qubit_names=handler.qubit_names,
        compiled_experiment=compiled_experiment,
        do_emulation=do_emulation,
    )

    task_result = task_manager.wait_for_result(task_id)
    handler.experiment_result = from_json(task_result.raw_data)
    return task_result
