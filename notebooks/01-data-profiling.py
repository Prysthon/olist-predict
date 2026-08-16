# %%
import pandas as pd

# %%
df = pd.read_csv("../data/bronze/olist_orders_dataset.csv")
df.head()

# %%
from ydata_profiling import ProfileReport

profile = ProfileReport(
    df,
    title="Olist Data Profiling",
    explorative=True,
)
profile.to_file("../reports/olist_data_profiling")

# %%
import pandera.pandas as pa
from pandera import Check

VALID_ORDER_STATUS = [
    "delivered",
    "shipped",
    "canceled",
    "unavailable",
    "invoiced",
    "processing",
    "created",
    "approved",
]

orders_schema = pa.DataFrameSchema(
    columns={
        "order_id": pa.Column(
            str,
            nullable=False,
            unique=True,
            checks=Check.str_length(32, 32),
        ),
        "customer_id": pa.Column(
            str,
            nullable=False,
            unique=True,
            checks=Check.str_length(32, 32),
        ),
        "order_status": pa.Column(
            str,
            nullable=False,
            checks=Check.isin(VALID_ORDER_STATUS),
        ),
        "order_purchase_timestamp": pa.Column(
            pa.DateTime,
            nullable=False,
        ),
        "order_approved_at": pa.Column(
            pa.DateTime,
            nullable=True,
        ),
        "order_delivered_carrier_date": pa.Column(
            pa.DateTime,
            nullable=True,
        ),
        "order_delivered_customer_date": pa.Column(
            pa.DateTime,
            nullable=True,
        ),
        "order_estimated_delivery_date": pa.Column(
            pa.DateTime,
            nullable=False,
        ),
    },
    checks=[
        Check(
            lambda df: (
                df["order_approved_at"].isna()
                | (df["order_approved_at"] >= df["order_purchase_timestamp"])
            ),
            error="Approval date before purchase date",
        )
    ],
    strict=True,
    coerce=True,
)
try:
    validated_df = orders_schema.validate(
        df,
        lazy=True,
    )
    print("Dataset successfully validated!")

except pa.errors.SchemaErrors as err:
    print("Dataset validation failed.")
    failure_cases = err.failure_cases
    display(failure_cases)  # noqa: F821

validated_df.head()

# %%
