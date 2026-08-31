from pathlib import Path

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def load_data(
    orders_path: Path,
    items_path: Path,
    customers_path: Path,
    spark: SparkSession,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Carrega os dados de pedidos, itens e clientes a partir de arquivos CSV.

    Args:
        orders_path: caminho para o arquivo CSV contendo os pedidos.
        items_path: caminho para o arquivo CSV contendo os itens dos pedidos.
        customers_path: caminho para o arquivo CSV contendo os clientes.
        spark: sessão Spark utilizada para leitura dos dados.

    Returns:
        orders: DataFrame contendo os pedidos.
        items: DataFrame contendo os itens dos pedidos.
        customers: DataFrame contendo os clientes.
    """

    logger.info("Carregando os dados...")

    orders = spark.read.csv(
        str(orders_path),
        header=True,
        inferSchema=True,
    )

    items = spark.read.csv(
        str(items_path),
        header=True,
        inferSchema=True,
    )

    customers = spark.read.csv(
        str(customers_path),
        header=True,
        inferSchema=True,
    )

    logger.success("Dados carregados com sucesso.")

    return orders, items, customers


def save_dataset(
    dataset: DataFrame,
    output_path: Path,
) -> None:
    """
    Salva um DataFrame Spark em formato Parquet.

    Args:
        dataset: DataFrame a ser salvo.
        output_path: caminho onde o dataset será salvo.
    """

    dataset.write.mode("overwrite").parquet(
        str(output_path),
    )

    logger.success(
        f"Dataset salvo com sucesso, na pasta: {output_path}."
    )


def create_target(
    orders: DataFrame,
) -> DataFrame:
    """
    Filtra pedidos válidos para análise e cria a variável alvo is_late.

    Args:
        orders: DataFrame contendo os pedidos.

    Returns:
        DataFrame com pedidos entregues e a variável alvo is_late.
    """

    orders = (
        orders
        .filter(
            (F.col("order_status") == "delivered")
            & F.col("order_delivered_customer_date").isNotNull()
            & F.col("order_estimated_delivery_date").isNotNull()
            & F.col("order_approved_at").isNotNull()
        )
        .withColumn(
            "is_late",
            F.when(
                F.col("order_delivered_customer_date")
                > F.col("order_estimated_delivery_date"),
                1,
            ).otherwise(0),
        )
    )

    logger.info("Target 'is_late' criado com filtros aplicados.")

    return orders


def aggregate_items(
    items: DataFrame,
) -> DataFrame:
    """
    Agrega os itens no nível do pedido.

    Args:
        items: DataFrame contendo os itens dos pedidos.

    Returns:
        DataFrame com uma linha por pedido e métricas agregadas dos itens.
    """

    items_agg = (
        items
        .groupBy("order_id")
        .agg(
            F.count("order_item_id").alias("item_count"),
            F.countDistinct("seller_id").alias("seller_count"),
            F.sum("price").alias("total_price"),
            F.sum("freight_value").alias("total_freight"),
        )
    )

    logger.info("Itens agregados por pedido.")

    return items_agg


def create_dataset(
    orders: DataFrame,
    items: DataFrame,
    customers: DataFrame,
) -> DataFrame:
    """
    Cria o dataset consolidado no nível do pedido.

    Args:
        orders: DataFrame contendo os pedidos.
        items: DataFrame contendo os itens dos pedidos.
        customers: DataFrame contendo os clientes.

    Returns:
        DataFrame consolidado no nível do pedido.
    """

    orders = create_target(orders)

    items_agg = aggregate_items(items)

    data = orders.join(
        items_agg,
        on="order_id",
        how="left",
    )

    data = data.join(
        customers.select(
            "customer_id",
            "customer_city",
            "customer_state",
        ),
        on="customer_id",
        how="left",
    )

    return data