import ray
import ray.serve
import vertex_ray
from ray.job_submission import JobSubmissionClient
from google.cloud import aiplatform

REGION = 'asia-northeast3'
PROJECT_NUMBER = '----'
CLUSTER_RESOURCE_NAME=f"vertex_ray://projects/{PROJECT_NUMBER}/locations/{REGION}/persistentResources/test"
aiplatform.init(location=REGION)
client = JobSubmissionClient(CLUSTER_RESOURCE_NAME)

job_id = client.submit_job(
  # Entrypoint shell command to execute
  #entrypoint="serve build object_detection:entrypoint -o serve_config.yaml",
  #entrypoint="python object_detection.py",
  entrypoint="python tune.py",
  #entrypoint="serve run object_detect.yaml",
  # Path to the local directory that contains the my_script.py file.
  runtime_env={
    "working_dir": "./xgboost",
    #"env_vars": {"RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1"},
    "pip": [
        "immutabledict",
        "etils",
        "google-cloud-aiplatform[tensorboard]",
        "xgboost",
        ]
  }
)

#client.stop_job()
# Ensure that the Ray job has been created.
print(job_id)

#client.delete_job()
#import ray.serve
#from google.cloud import aiplatform
#aiplatform.init(location='asia-northeast1')
#ray.init(CLUSTER_RESOURCE_NAME)
#ray.serve.delete("default")