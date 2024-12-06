import ray
import raydp
import pyspark

@ray.remote
class SparkExecutor:
  spark: pyspark.sql.SparkSession = None

  def __init__(self):
    self.spark = raydp.init_spark(
      app_name="RAYDP ACTOR EXAMPLE",
      num_executors=1,
      executor_cores=1,
      executor_memory="500M",
    )

  def get_data(self):
    df = self.spark.createDataFrame(
        [
            ("sue", 32),
            ("li", 3),
            ("bob", 75),
            ("heo", 13),
        ],
        ["first_name", "age"],
    )
    return df.toJSON().collect()

  def stop_spark(self):
    import raydp
    raydp.stop_spark()

s = SparkExecutor.remote()
data = ray.get(s.get_data.remote())
print(data)
ray.get(s.stop_spark.remote())

