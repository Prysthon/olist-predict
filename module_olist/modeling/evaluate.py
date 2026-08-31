import numpy as np

from loguru import logger

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


def evaluate_models(
    models: dict,
    test_df: DataFrame,
):
    evaluator = BinaryClassificationEvaluator(
        labelCol="is_late",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    results = {}

    for name, model in models.items():

        logger.info(f"Avaliando modelo: {name}")

        predictions = model.transform(test_df)

        predictions = predictions.withColumn(
            "probability_positive",
            vector_to_array("probability")[1],
        )

        roc_auc = evaluator.evaluate(predictions)

        # Coleta os dados uma única vez para evitar múltiplas passadas
        pred_data = predictions.select("probability_positive", "is_late").toPandas()

        best_threshold = None
        best_f1 = -1
        best_precision = None
        best_recall = None

        for threshold in np.arange(0.05, 0.51, 0.01):

            preds = (pred_data["probability_positive"] >= threshold).astype(int)
            labels = pred_data["is_late"]

            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = threshold

        logger.info(f"Modelo: {name}")
        logger.info(f"Melhor threshold: {best_threshold:.2f}")
        logger.info(f"Melhor precision: {best_precision:.3f}")
        logger.info(f"Melhor recall: {best_recall:.3f}")
        logger.info(f"Melhor F1: {best_f1:.3f}")
        logger.info(f"ROC-AUC: {roc_auc:.3f}")

        results[name] = {
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "roc_auc": roc_auc,
        }

    return results