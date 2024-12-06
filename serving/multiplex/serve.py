from ray import serve
import starlette
import ray

#https://docs.ray.io/en/releases-2.33.0/serve/model-multiplexing.html

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.3, "num_gpus": 0})
class ModelInferencer:
    def __init__(self):
        pass

    #Load max 3 models in a replica and evict least recently used model
    @serve.multiplexed(max_num_models_per_replica=3)
    async def get_model(self, model_id: str):
        return f"Model {model_id}"

    async def __call__(self, request: starlette.requests.Request):
        model_id = serve.get_multiplexed_model_id()
        model = await self.get_model(model_id)
        return f"Hello from {model}"

entry = ModelInferencer.bind()
serve.run(entry, blocking=True)
#if no serve_ header, routed to random replica
#import requests
#resp = requests.get("http://localhost:8000", headers={"serve_multiplexed_model_id": str("1")})
