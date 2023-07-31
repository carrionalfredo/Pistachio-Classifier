from prefect.deployments import Deployment
from training_aws import run

deployment = Deployment.build_from_flow(
    flow=run,
    name="Pistachio Classifier deployment",
)