from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def predict(
    model: PipelineModel,
    data: DataFrame,
    threshold: float,
) -> DataFrame:
    predictions = model.transform(
        data
    )

    predictions = predictions.withColumn(
        "probability_positive",
        vector_to_array(
            "probability"
        )[1],
    )

    predictions = predictions.withColumn(
        "prediction",
        F.when(
            F.col(
                "probability_positive"
            )
            >= float(threshold),
            1,
        ).otherwise(0),
    )

    return predictions