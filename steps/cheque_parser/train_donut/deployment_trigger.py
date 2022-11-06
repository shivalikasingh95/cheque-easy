from zenml.steps import step, BaseParameters

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    #deploy: bool = True
    min_accuracy: float = 0.8


@step(enable_cache=False)
def deployment_trigger(
    mean_accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return mean_accuracy > config.min_accuracy #config.deploy