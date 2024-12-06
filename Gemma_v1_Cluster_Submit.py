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
    "working_dir": "./gemma_v1",
    "config": {"setup_timeout_seconds": 3600},
    #"env_vars": {"RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1"},
    "pip": [
        "etils==1.7.0",
        "google-cloud-aiplatform[tensorboard]",
        "torch==2.2.1",
        "transformers==4.38.1",
        "datasets==2.17.0",
        "peft==0.8.2",
        "trl==0.7.10",
        "evaluate==0.4.1",
        "bitsandbytes==0.42.0",
        "rouge-score==0.1.2",
        "nltk==3.8.1",
        "accelerate==0.27.1",
        "importlib-resources==6.1.2"
        #"fastapi",
        #"ray==2.9.3", # pin the Ray version to prevent it from being overwritten
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