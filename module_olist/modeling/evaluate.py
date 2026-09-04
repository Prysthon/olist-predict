from loguru import logger

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from pyspark.ml.functions import vector_to_array


def evaluate_model(
    model,
    model_name: str,
    test_df: DataFrame,
    threshold: float,
) -> dict:
    """
    Avalia o modelo final
    no conjunto de teste.
    """

    roc_auc_evaluator = (
        BinaryClassificationEvaluator(
            labelCol="is_late",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
    )

    precision_evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="is_late",
            predictionCol="prediction_threshold",
            metricName="precisionByLabel",
            metricLabel=1.0,
        )
    )

    recall_evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="is_late",
            predictionCol="prediction_threshold",
            metricName="recallByLabel",
            metricLabel=1.0,
        )
    )

    f1_evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="is_late",
            predictionCol="prediction_threshold",
            metricName="fMeasureByLabel",
            metricLabel=1.0,
            beta=1.0,
        )
    )

    logger.info(
        f"Avaliando modelo: "
        f"{model_name}"
    )

    predictions = model.transform(
        test_df
    )

    predictions = predictions.withColumn(
        "probability_positive",
        vector_to_array(
            "probability"
        )[1],
    )

    predictions = predictions.withColumn(
        "prediction_threshold",
        F.when(
            F.col(
                "probability_positive"
            )
            >= float(threshold),
            1.0,
        ).otherwise(0.0),
    )

    roc_auc = (
        roc_auc_evaluator.evaluate(
            predictions
        )
    )

    precision = (
        precision_evaluator.evaluate(
            predictions
        )
    )

    recall = (
        recall_evaluator.evaluate(
            predictions
        )
    )

    f1 = (
        f1_evaluator.evaluate(
            predictions
        )
    )

    logger.info(
        f"Modelo: "
        f"{model_name}"
    )

    logger.info(
        f"Threshold: "
        f"{threshold:.2f}"
    )

    logger.info(
        f"Precision: "
        f"{precision:.3f}"
    )

    logger.info(
        f"Recall: "
        f"{recall:.3f}"
    )

    logger.info(
        f"F1: "
        f"{f1:.3f}"
    )

    logger.info(
        f"ROC-AUC: "
        f"{roc_auc:.3f}"
    )

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }