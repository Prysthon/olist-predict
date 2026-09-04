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

from module_olist.features import (
    create_features,
)

from module_olist.modeling.split import (
    split_data,
)

from module_olist.modeling.cross_validation import (
    cross_validate_models,
)

from module_olist.modeling.train import (
    train_model,
)

from module_olist.modeling.evaluate import (
    evaluate_model,
)


def main() -> None:
    logger.info(
        "Iniciando pipeline de dados..."
    )

    spark = (
        SparkSession.builder
        .appName("olist-pipeline")
        .master("local[*]")
        .config(
            "spark.ui.enabled",
            "false",
        )
        .config(
            "spark.sql.shuffle.partitions",
            "4",
        )
        .config(
            "spark.driver.memory",
            "2g",
        )
        .config(
            "spark.executor.memory",
            "2g",
        )
        .config(
            "spark.driver.maxResultSize",
            "1g",
        )
        .config(
            (
                "spark.sql.execution."
                "arrow.pyspark.enabled"
            ),
            "true",
        )
        .config(
            (
                "spark.sql."
                "autoBroadcastJoinThreshold"
            ),
            "10485760",
        )
        .config(
            "spark.sql.adaptive.enabled",
            "true",
        )
        .config(
            (
                "spark.sql.adaptive."
                "coalescePartitions.enabled"
            ),
            "true",
        )
        .config(
            "spark.memory.fraction",
            "0.8",
        )
        .config(
            (
                "spark.memory."
                "storageFraction"
            ),
            "0.3",
        )
        .getOrCreate()
    )

    try:
        (
            orders,
            items,
            customers,
        ) = load_data(
            orders_path=(
                BRONZE_DATA_DIR
                / "olist_orders_dataset.csv"
            ),
            items_path=(
                BRONZE_DATA_DIR
                / "olist_order_items_dataset.csv"
            ),
            customers_path=(
                BRONZE_DATA_DIR
                / "olist_customers_dataset.csv"
            ),
            spark=spark,
        )

        dataset = create_dataset(
            orders=orders,
            items=items,
            customers=customers,
        )

        dataset = create_features(
            dataset
        )

        save_dataset(
            dataset=dataset,
            output_path=(
                SILVER_DATA_DIR
                / "olist_features"
            ),
        )

        logger.info(
            "Dividindo dados em "
            "treino e teste..."
        )

        (
            train_df,
            test_df,
        ) = split_data(
            data=dataset,
            train_size=0.8,
            seed=42,
        )

        train_df = train_df.cache()
        test_df = test_df.cache()

        logger.info(
            f"Dados de treino: "
            f"{train_df.count():,} registros"
        )

        logger.info(
            f"Dados de teste: "
            f"{test_df.count():,} registros"
        )

        logger.info(
            "Iniciando "
            "Cross Validation..."
        )

        (
            best_model_name,
            best_threshold,
        ) = cross_validate_models(
            train_df=train_df,
        )

        logger.info(
            "Treinando modelo "
            "selecionado..."
        )

        best_model = train_model(
            model_name=best_model_name,
            train_df=train_df,
            threshold=best_threshold,
        )

        logger.info(
            "Avaliando modelo final "
            "no conjunto de teste..."
        )

        results = evaluate_model(
            model=best_model,
            model_name=best_model_name,
            test_df=test_df,
            threshold=best_threshold,
        )

        logger.success(
            "RESULTADOS FINAIS"
        )

        logger.info(
            f"{results['model']} | "
            f"F1="
            f"{results['f1']:.3f} | "
            f"Precision="
            f"{results['precision']:.3f} | "
            f"Recall="
            f"{results['recall']:.3f} | "
            f"ROC-AUC="
            f"{results['roc_auc']:.3f} | "
            f"Threshold="
            f"{results['threshold']:.2f}"
        )

        train_df.unpersist()
        test_df.unpersist()

        logger.success(
            "Pipeline executado "
            "com sucesso."
        )

    finally:
        try:
            spark.stop()

        except Exception:
            logger.warning(
                "Spark já estava encerrado."
            )


if __name__ == "__main__":
    main()