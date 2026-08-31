import gc
from pathlib import Path

from loguru import logger
from pyspark.sql import DataFrame

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)


def train_models(train_df: DataFrame):

    pipelines = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        # LightGBM: SynapseML 1.1.3 é incompatível com PySpark 4.0.0
        # Requer downgrade para PySpark 3.5.x ou aguardar SynapseML atualizado
        # "LightGBM": create_lightgbm_pipeline(),
    }

    trained_models = {}
    models_dir = Path("models")

    # Treina um modelo por vez para economizar memória
    for name, pipeline in pipelines.items():
        logger.info(f"Treinando modelo: {name}")

        fitted_model = pipeline.fit(train_df)
        trained_models[name] = fitted_model

        # Salva o modelo em disco
        model_path = models_dir / name.lower().replace(" ", "_")
        fitted_model.write().overwrite().save(str(model_path))
        logger.info(f"Modelo salvo em: {model_path}")

        # Força garbage collection entre modelos
        gc.collect()

        logger.success(f"Modelo {name} treinado com sucesso.")

    return trained_models