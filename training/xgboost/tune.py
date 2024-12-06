import sklearn.datasets
import sklearn.metrics
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ray import tune, train
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

from google.cloud import aiplatform

EXPERIMENT_NAME = "breast-cancer"
LOG_DIR = "----/experiments"
HF_TOKEN = "----"

def train_breast_cancer(config: dict):
    # This is a simple training function to be passed into Tune
    # Load dataset
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier, using the Tune callback
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        verbose_eval=False,
        # `TuneReportCheckpointCallback` defines the checkpointing frequency and format.
        callbacks=[TuneReportCheckpointCallback(frequency=1)],
    )

def get_best_model_checkpoint(results):
    best_result = results.get_best_result()

    # `TuneReportCheckpointCallback` provides a helper method to retrieve the
    # model from a checkpoint.
    best_bst = TuneReportCheckpointCallback.get_model(best_result.checkpoint)

    accuracy = 1.0 - best_result.metrics["eval-error"]
    print(f"Best model parameters: {best_result.config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst

from google.cloud.aiplatform.tensorboard import uploader_utils
def folder_callback(trial):
    #Folder name should name + {}
    return f"{EXPERIMENT_NAME}-{uploader_utils.reformat_run_name(str(trial))}".lower()

def tune_xgboost(num_sample = 10):
    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "device": "cuda",
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2  # 10 training iterations
    )

    tuner = tune.Tuner(
        tune.with_resources(train_breast_cancer, resources={"cpu": 0.3, "gpu": 0.1}),
        tune_config=tune.TuneConfig(
            metric="eval-logloss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_sample,
            trial_dirname_creator=folder_callback
        ),
        param_space=search_space,
        run_config=train.RunConfig(
            name=EXPERIMENT_NAME,
            storage_path=f"/gcs/{LOG_DIR}"
        )
    )
    results = tuner.fit()
    return results

def upload_model(model_name: str, artifact_uri: str):
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
        sync=True,
    )
    model.wait()

#import ray.util.placement_group as placement_group
#import ray
#pg = placement_group([{"CPU": 2, "GPU": 1}])
#ray.get(pg.ready(), timeout=120)

#refer following link to use BQ (read, transform, write)
#https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/bigquery-integration

results = tune_xgboost(num_sample=10)

# Load the best model checkpoint.
best_bst = get_best_model_checkpoint(results)

#use filename to store model
best_bst.save_model(f"/gcs/{LOG_DIR}/final/model.bst")

# You could now do further predictions with
# best_bst.predict(...)

#register model to registry, set folder
upload_model(EXPERIMENT_NAME, f"gs://{LOG_DIR}/final")

tensorboard = aiplatform.Tensorboard.create(
    display_name=EXPERIMENT_NAME
)

aiplatform.init(experiment_tensorboard=tensorboard)
aiplatform.upload_tb_log(
    tensorboard_id=tensorboard.name,
    tensorboard_experiment_name=EXPERIMENT_NAME,
    logdir=f"gs://{LOG_DIR}/{EXPERIMENT_NAME}",
    verbosity = 1
)
