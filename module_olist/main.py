from loguru import logger

from pyspark.sql import SparkSession

from module_olist.config import (
    BRONZE_DATA_DIR,
    SILVER_DATA_DIR,
)

from module_olist.dataset import (
    create_dataset,
    load_data,
    save_dataset,
)

from module_olist.features import create_features

from module_olist.modeling.split import split_data
from module_olist.modeling.train import train_models
from module_olist.modeling.evaluate import evaluate_models


def main() -> None:
    """
    Executa o pipeline de preparação e modelagem dos dados.

    O pipeline realiza:
        1. Carregamento dos dados da camada bronze.
        2. Criação do dataset consolidado.
        3. Criação das features.
        4. Salvamento do dataset na camada silver.
        5. Divisão dos dados em treino e teste.
        6. Treinamento dos modelos.
        7. Avaliação dos modelos.
    """

    logger.info("Iniciando pipeline de dados...")

    spark = (
        SparkSession.builder
        .appName("olist-pipeline")
        .master("local[*]")  # Usa todos os cores disponíveis
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .getOrCreate()
    )

    try:
        orders, items, customers = load_data(
            orders_path=BRONZE_DATA_DIR / "olist_orders_dataset.csv",
            items_path=BRONZE_DATA_DIR / "olist_order_items_dataset.csv",
            customers_path=BRONZE_DATA_DIR / "olist_customers_dataset.csv",
            spark=spark,
        )

        dataset = create_dataset(
            orders=orders,
            items=items,
            customers=customers,
        )

        dataset = create_features(dataset)

        save_dataset(
            dataset=dataset,
            output_path=SILVER_DATA_DIR / "olist_features",
        )

        logger.info("Dividindo dados em treino e teste...")

        train_df, test_df = split_data(
            data=dataset,
            train_size=0.8,
            seed=42,
        )

        # Cache para evitar recomputação durante treinamento
        train_df = train_df.cache()
        test_df = test_df.cache()

        # Força avaliação para popular o cache
        logger.info(f"Dados de treino: {train_df.count():,} registros")
        logger.info(f"Dados de teste: {test_df.count():,} registros")

        logger.info("Treinando modelos...")

        models = train_models(
            train_df=train_df,
        )

        logger.info("Avaliando modelos...")

        results = evaluate_models(
            models=models,
            test_df=test_df,
        )

        for name, metrics in results.items():
            logger.info(
                f"{name} | "
                f"F1={metrics['f1']:.3f} | "
                f"Precision={metrics['precision']:.3f} | "
                f"Recall={metrics['recall']:.3f} | "
                f"ROC-AUC={metrics['roc_auc']:.3f} | "
                f"Threshold={metrics['threshold']:.2f}"
            )

        # Libera cache dos DataFrames
        train_df.unpersist()
        test_df.unpersist()

        logger.success("Pipeline executado com sucesso.")

    finally:
        try:
            spark.stop()

        except Exception:
            logger.warning("Spark já estava encerrado.")


if __name__ == "__main__":
    main()