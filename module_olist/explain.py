from pathlib import Path

import matplotlib.pyplot as plt
import shap

from loguru import logger
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from module_olist.config import (
    FIGURES_DIR,
    SILVER_DATA_DIR,
    MODELS_DIR,
)

from module_olist.modeling.interpret import (
    prepare_data_for_shap,
    create_explainer,
    calculate_shap_values,
)


SAMPLE_SIZE = 2000


def main():
    logger.info("Iniciando análise SHAP...")

    spark = (
        SparkSession.builder
        .appName("Olist SHAP Analysis")
        .getOrCreate()
    )

    model_path = Path(MODELS_DIR) / "best_model"

    data_path = Path(SILVER_DATA_DIR) / "olist_features"

    logger.info(
        "Carregando modelo de {}",
        model_path,
    )

    model = PipelineModel.load(
        str(model_path)
    )

    logger.info(
        "Carregando dados de {}",
        data_path,
    )

    test_df = spark.read.parquet(
        str(data_path)
    )

    logger.info(
        "Preparando amostra de dados para SHAP..."
    )

    X_shap = prepare_data_for_shap(
        pipeline=model,
        X=test_df,
        sample_size=SAMPLE_SIZE,
    )

    logger.info(
        "Criando SHAP explainer..."
    )

    explainer = create_explainer(
        pipeline=model
    )

    logger.info(
        "Calculando valores SHAP..."
    )

    shap_values = calculate_shap_values(
        explainer=explainer,
        X_transformed=X_shap,
    )

    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Gerando gráfico SHAP de importância global..."
    )

    shap.plots.bar(
        shap_values,
        show=False,
    )

    plt.tight_layout()

    bar_path = (
        figures_dir
        / "shap_feature_importance.png"
    )

    plt.savefig(
        bar_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(
        "Gráfico salvo em {}",
        bar_path,
    )

    logger.info(
        "Gerando gráfico SHAP beeswarm..."
    )

    shap.plots.beeswarm(
        shap_values,
        show=False,
    )

    plt.tight_layout()

    beeswarm_path = (
        figures_dir
        / "shap_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(
        "Gráfico salvo em {}",
        beeswarm_path,
    )

    logger.info(
        "Análise SHAP finalizada com sucesso."
    )

    spark.stop()


if __name__ == "__main__":
    main()