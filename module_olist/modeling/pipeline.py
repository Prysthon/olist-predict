from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)

from pyspark.ml.classification import GBTClassifier

from xgboost.spark import SparkXGBClassifier
from synapse.ml.lightgbm import LightGBMClassifier


NUMERIC_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
]


CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state",
]


def create_preprocessing_stages():
    """
    Cria as etapas de pré-processamento.
    """

    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_index",
            handleInvalid="keep",
        )
        for col in CATEGORICAL_FEATURES
    ]

    indexed_cols = [
        f"{col}_index"
        for col in CATEGORICAL_FEATURES
    ]

    encoded_cols = [
        f"{col}_ohe"
        for col in CATEGORICAL_FEATURES
    ]

    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols,
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURES + encoded_cols,
        outputCol="features",
        handleInvalid="keep",
    )

    return indexers + [encoder, assembler]


def create_gradient_boosting_pipeline():
    preprocessing = create_preprocessing_stages()

    model = GBTClassifier(
        labelCol="is_late",
        featuresCol="features",
        maxIter=100,
        stepSize=0.1,
        maxDepth=3,
        seed=42,
    )

    return Pipeline(
        stages=preprocessing + [model]
    )


def create_xgboost_pipeline():
    preprocessing = create_preprocessing_stages()

    model = SparkXGBClassifier(
        features_col="features",
        label_col="is_late",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    return Pipeline(
        stages=preprocessing + [model]
    )


def create_lightgbm_pipeline():
    preprocessing = create_preprocessing_stages()

    model = LightGBMClassifier(
        labelCol="is_late",
        featuresCol="features",
        objective="binary",
        numIterations=200,
        learningRate=0.05,
        maxDepth=5,
        numLeaves=31,
        featureFraction=0.8,
        baggingFraction=0.8,
        baggingFreq=1,
        seed=42,
    )

    return Pipeline(
        stages=preprocessing + [model]
    )