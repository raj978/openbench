from inspect_ai import Task, task


@task
def bbh() -> Task:
    """ """

    return Task(
        # dataset=dataset,
        # solver=custom_solver(),
        # scorer=custom_scorer(),
        # config=GenerateConfig(temperature=0.7),
    )
