from inspect_ai import Task, task


@task
def taubench():
    return Task(
        name="taubench",
        # dataset=get_taubench_dataset(),
        # scorer=get_taubench_scorer(),
    )
