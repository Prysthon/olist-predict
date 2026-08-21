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


def main() -> None:
    """
    Executa o pipeline de preparação dos dados.

    O pipeline realiza:
        1. Carregamento dos dados da camada bronze.
        2. Criação do dataset consolidado.
        3. Criação das features.
        4. Salvamento do dataset na camada silver.
    """

    logger.info("Iniciando pipeline de dados...")

    spark = (
        SparkSession.builder
        .appName("olist-feature-engineering")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
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

        logger.success("Pipeline executado com sucesso.")

    finally:
        try:
            spark.stop()
        except Exception:
            logger.warning("Spark já estava encerrado.")


if __name__ == "__main__":
    main()