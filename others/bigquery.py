import ray
import google.cloud.aiplatform

@ray.remote
def read_bigquery():
    import vertex_ray
    dataset = DATASET
    parallelism = PARALLELISM
    query = QUERY

    ds = vertex_ray.data.read_bigquery(
        dataset=dataset,
        parallelism=parallelism,
        query=query
    )
    ds.materialize()

@ray.remote
def transform_data():
    # BigQuery Read first
    import pandas as pd
    import pyarrow as pa

    def filter_batch(table: pa.Table) -> pa.Table:
        df = table.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
        # PANDAS_TRANSFORMATIONS_HERE
        return pa.Table.from_pandas(df)

    ds = ds.map_batches(filter_batch, batch_format="pyarrow").random_shuffle()
    ds.materialize()

    # You can repartition before writing to determine the number of write blocks
    ds = ds.repartition(4)
    ds.materialize()

@ray.remote
def write_bigquery():
    # BigQuery Read and optional data transformation first
    import vertex_ray
    dataset=DATASET
    vertex_ray.data.write_bigquery(
        ds,
        dataset=dataset
    )
