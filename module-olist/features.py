from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def create_features(data: DataFrame) -> DataFrame:
    """
    Cria as features mínimas definidas no EDA.

    As variáveis utilizam apenas informações conhecidas até a aprovação
    do pagamento.

    Args:
        orders: DataFrame contendo os pedidos.

    Returns:
        DataFrame com as features de prazo e momento da compra.
    """
    orders = (
        data
        .withColumn(
            "promised_days",
            (
                F.col("order_estimated_delivery_date").cast("double")
                - F.col("order_approved_at").cast("double")
            ) / 86_400,
        )
        .withColumn(
            "purchase_month",
            F.month("order_purchase_timestamp"),
        )
        .withColumn(
            "purchase_weekday",
            F.pmod(
                F.dayofweek("order_purchase_timestamp") + F.lit(5),
                F.lit(7),
            ),
        )
        .withColumn(
            "purchase_hour",
            F.hour("order_purchase_timestamp"),
        )
    )

    logger.info(
        "Features criadas: promised_days, purchase_month, "
        "purchase_weekday e purchase_hour."
    )

    return orders
