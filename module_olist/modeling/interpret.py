import numpy as np
import pandas as pd
import shap

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame


def _get_feature_names(transformed_df: DataFrame) -> list[str]:
    """
    Recupera os nomes das features armazenados nos metadados
    da coluna 'features' criada pelo VectorAssembler.
    """
    metadata = transformed_df.schema["features"].metadata

    attrs = metadata.get("ml_attr", {}).get("attrs", {})

    feature_names = []

    for attr_type in ["numeric", "binary", "nominal"]:
        for attr in attrs.get(attr_type, []):
            feature_names.append(
                (attr["idx"], attr["name"])
            )

    feature_names.sort(key=lambda x: x[0])

    return [name for _, name in feature_names]


def prepare_data_for_shap(
    pipeline: PipelineModel,
    X: DataFrame,
    sample_size: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Aplica apenas as etapas de pré-processamento do PipelineModel
    e converte uma amostra das features para Pandas.

    O último estágio do pipeline é assumido como sendo o modelo.
    """

    preprocessing_stages = pipeline.stages[:-1]

    transformed_df = X

    for stage in preprocessing_stages:
        transformed_df = stage.transform(transformed_df)

    feature_names = _get_feature_names(transformed_df)

    total_rows = transformed_df.count()

    if total_rows > sample_size:
        fraction = min(
            1.0,
            sample_size / total_rows,
        )

        transformed_df = (
            transformed_df
            .sample(
                withReplacement=False,
                fraction=fraction,
                seed=seed,
            )
            .limit(sample_size)
        )

    rows = (
        transformed_df
        .select("features")
        .collect()
    )

    feature_matrix = np.array(
        [
            row["features"].toArray()
            for row in rows
        ]
    )

    X_transformed = pd.DataFrame(
        feature_matrix,
        columns=feature_names,
    )

    return X_transformed


def create_explainer(
    pipeline: PipelineModel,
):
    """
    Cria o SHAP TreeExplainer a partir do modelo XGBoost
    treinado presente no último estágio do PipelineModel.
    """

    model = pipeline.stages[-1]

    if not hasattr(model, "get_booster"):
        raise TypeError(
            "O modelo final do pipeline não possui get_booster(). "
            "Este explainer foi implementado para SparkXGBClassifierModel."
        )

    booster = model.get_booster()

    explainer = shap.TreeExplainer(booster)

    return explainer


def calculate_shap_values(
    explainer,
    X_transformed: pd.DataFrame,
):
    """
    Calcula os valores SHAP para os dados transformados.
    """

    shap_values = explainer(X_transformed)

    return shap_values