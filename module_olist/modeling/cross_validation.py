import gc

import numpy as np

from loguru import logger

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
)

from pyspark.ml.functions import vector_to_array

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
)


NUM_FOLDS = 3
SEED = 42


def find_best_threshold(
    predictions: DataFrame,
) -> tuple[float, float]:
    """
    Encontra o threshold que maximiza
    o F1 da classe positiva.
    """

    evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="is_late",
            predictionCol="prediction_threshold",
            metricName="fMeasureByLabel",
            metricLabel=1.0,
            beta=1.0,
        )
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(
        0.05,
        0.51,
        0.01,
    ):
        threshold_predictions = (
            predictions.withColumn(
                "prediction_threshold",
                F.when(
                    F.col(
                        "probability_positive"
                    )
                    >= float(threshold),
                    1.0,
                ).otherwise(0.0),
            )
        )

        f1 = evaluator.evaluate(
            threshold_predictions
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1,
    )


def cross_validate_models(
    train_df: DataFrame,
) -> tuple[str, float]:
    """
    Executa Cross Validation dos modelos.

    Para cada modelo:

        1. Divide o train_df em folds.

        2. Treina em K-1 folds.

        3. Prediz no fold restante.

        4. Junta as probabilidades
           Out-of-Fold.

        5. Testa diferentes thresholds.

        6. Calcula o melhor F1.

    Retorna:

        - nome do melhor modelo;
        - melhor threshold.
    """

    pipelines = {
        "Gradient Boosting": (
            create_gradient_boosting_pipeline()
        ),
        "XGBoost": (
            create_xgboost_pipeline()
        ),
    }

    cv_results = {}

    # Cria uma coluna indicando
    # a qual fold cada registro pertence.
    train_with_folds = (
        train_df.withColumn(
            "fold",
            (
                F.rand(SEED)
                * NUM_FOLDS
            ).cast("int"),
        )
    )

    train_with_folds = (
        train_with_folds.cache()
    )

    for name, pipeline in pipelines.items():
        logger.info(
            f"Iniciando Cross Validation: "
            f"{name}"
        )

        oof_predictions = None

        for fold in range(NUM_FOLDS):
            logger.info(
                f"{name} | "
                f"Fold {fold + 1}/"
                f"{NUM_FOLDS}"
            )

            fold_train_df = (
                train_with_folds
                .filter(
                    F.col("fold") != fold
                )
                .drop("fold")
            )

            fold_validation_df = (
                train_with_folds
                .filter(
                    F.col("fold") == fold
                )
                .drop("fold")
            )

            model = pipeline.fit(
                fold_train_df
            )

            predictions = (
                model.transform(
                    fold_validation_df
                )
            )

            predictions = (
                predictions.withColumn(
                    "probability_positive",
                    vector_to_array(
                        "probability"
                    )[1],
                )
            )

            fold_predictions = (
                predictions.select(
                    "is_late",
                    "probability_positive",
                )
            )

            if oof_predictions is None:
                oof_predictions = (
                    fold_predictions
                )

            else:
                oof_predictions = (
                    oof_predictions.unionByName(
                        fold_predictions
                    )
                )

            del model

            gc.collect()

        oof_predictions = (
            oof_predictions.cache()
        )

        logger.info(
            f"Selecionando threshold: "
            f"{name}"
        )

        (
            best_threshold,
            best_f1,
        ) = find_best_threshold(
            predictions=oof_predictions,
        )

        cv_results[name] = {
            "threshold": (
                best_threshold
            ),
            "f1": best_f1,
        }

        logger.info(
            f"{name} | "
            f"Threshold="
            f"{best_threshold:.2f} | "
            f"F1 OOF="
            f"{best_f1:.3f}"
        )

        oof_predictions.unpersist()

    best_model_name = max(
        cv_results,
        key=lambda name: (
            cv_results[name]["f1"]
        ),
    )

    best_threshold = (
        cv_results[
            best_model_name
        ]["threshold"]
    )

    best_f1 = (
        cv_results[
            best_model_name
        ]["f1"]
    )

    logger.success(
        "MODELO SELECIONADO"
    )

    logger.info(
        f"Modelo: "
        f"{best_model_name}"
    )

    logger.info(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    logger.info(
        f"F1 OOF: "
        f"{best_f1:.3f}"
    )

    train_with_folds.unpersist()

    return (
        best_model_name,
        best_threshold,
    )