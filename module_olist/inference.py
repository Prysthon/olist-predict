import json

from pathlib import Path

from loguru import logger

from pyspark.ml import PipelineModel

from pyspark.sql import SparkSession

from module_olist.modeling.predict import (
    predict,
)


MODEL_PATH = Path(
    "models/best_model"
)

METADATA_PATH = Path(
    "models/metadata.json"
)

INPUT_PATH = Path(
    "data/silver/olist_features"
)

OUTPUT_PATH = Path(
    "data/gold/predictions"
)


def main() -> None:
    logger.info(
        "Iniciando pipeline "
        "de inferência..."
    )

    spark = (
        SparkSession.builder
        .appName("olist-inference")
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
        .getOrCreate()
    )

    try:
        logger.info(
            f"Carregando metadata: "
            f"{METADATA_PATH}"
        )

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

        model_name = metadata[
            "model_name"
        ]

        threshold = metadata[
            "threshold"
        ]

        logger.info(
            f"Modelo selecionado: "
            f"{model_name}"
        )

        logger.info(
            f"Threshold: "
            f"{threshold:.2f}"
        )

        logger.info(
            f"Carregando modelo: "
            f"{MODEL_PATH}"
        )

        model = PipelineModel.load(
            str(MODEL_PATH)
        )

        logger.info(
            f"Carregando dados: "
            f"{INPUT_PATH}"
        )

        data = spark.read.parquet(
            str(INPUT_PATH)
        )

        logger.info(
            f"Dados carregados: "
            f"{data.count():,} registros"
        )

        logger.info(
            "Executando previsões..."
        )

        predictions = predict(
            model=model,
            data=data,
            threshold=threshold,
        )

        logger.info(
            "Salvando previsões..."
        )

        predictions.write \
            .mode("overwrite") \
            .parquet(
                str(OUTPUT_PATH)
            )

        logger.info(
            f"Previsões salvas em: "
            f"{OUTPUT_PATH}"
        )

        predictions.select(
            "probability_positive",
            "prediction",
        ).show(
            20,
            truncate=False,
        )

        logger.success(
            "Inferência executada "
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