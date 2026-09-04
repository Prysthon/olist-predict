import json

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
    threshold: float,
):
    """
    Treina o modelo selecionado usando todo o conjunto de treino.

    Salva:
        - modelo vencedor;
        - nome do modelo;
        - threshold selecionado.
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

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / "best_model"
    )

    metadata_path = (
        models_dir
        / "metadata.json"
    )

    model.write().overwrite().save(
        str(model_path)
    )

    metadata = {
        "model_name": model_name,
        "threshold": threshold,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    logger.info(
        f"Modelo salvo em: "
        f"{model_path}"
    )

    logger.info(
        f"Metadata salva em: "
        f"{metadata_path}"
    )

    logger.info(
        f"Threshold salvo: "
        f"{threshold:.2f}"
    )

    logger.success(
        f"Modelo {model_name} "
        f"treinado com sucesso."
    )

    return model