from pathlib import Path

from loguru import logger

from pyspark.sql import DataFrame

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
)


def train_model(
    model_name: str,
    train_df: DataFrame,
):
    """
    Treina o modelo selecionado
    usando todo o conjunto de treino.

    Salva somente o modelo vencedor.
    """

    pipelines = {
        "Gradient Boosting": (
            create_gradient_boosting_pipeline()
        ),
        "XGBoost": (
            create_xgboost_pipeline()
        ),
    }

    if model_name not in pipelines:
        raise ValueError(
            f"Modelo não encontrado: "
            f"{model_name}"
        )

    logger.info(
        f"Treinando modelo final: "
        f"{model_name}"
    )

    pipeline = pipelines[
        model_name
    ]

    model = pipeline.fit(
        train_df
    )

    models_dir = Path(
        "models"
    )

    model_path = (
        models_dir
        / model_name.lower().replace(
            " ",
            "_",
        )
    )

    model.write().overwrite().save(
        str(model_path)
    )

    logger.info(
        f"Modelo salvo em: "
        f"{model_path}"
    )

    logger.success(
        f"Modelo {model_name} "
        f"treinado com sucesso."
    )

    return model