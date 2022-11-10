from zenml.steps import step, BaseParameters

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger deployment of trained model"""

    min_accuracy: float = 0.8


@step(enable_cache=False)
def deployment_trigger(
    mean_accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """ 
    simple model deployment trigger that compares mean accuracy obtained on test set
    after model evaluation with the set min threshold value for model accuracy
    """

    return mean_accuracy > config.min_accuracy 