from pyspark.sql import DataFrame


FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",
]

TARGET = "is_late"


def split_data(
    data: DataFrame,
    train_size: float = 0.8,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame]:
    """
    Divide o dataset em conjuntos de treino e teste de forma estratificada.

    Mantém aproximadamente a mesma proporção da variável target
    nos conjuntos de treino e teste.

    Args:
        data: DataFrame Spark contendo as features e o target.
        train_size: Proporção dos dados destinada ao treino.
        seed: Seed para garantir reprodutibilidade.

    Returns:
        Tupla contendo:
            - DataFrame de treino
            - DataFrame de teste
    """

    dataset = data.select(*FEATURES, TARGET)

    classes = (
        dataset
        .select(TARGET)
        .distinct()
        .rdd
        .flatMap(lambda row: row)
        .collect()
    )

    fractions = {
        target_class: train_size
        for target_class in classes
    }

    train = dataset.sampleBy(
        col=TARGET,
        fractions=fractions,
        seed=seed,
    )

    test = dataset.subtract(train)

    return train, test