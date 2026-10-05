"""Load the Olist Brazilian E-commerce CSVs (Kaggle: olistbr/brazilian-ecommerce) into DuckDB.

    python scripts/load_olist.py --csv-dir data/olist_csv --out data/olist.duckdb

Tables get short names and column comments; those comments are what schema retrieval and the
SQL prompt see, so they matter for accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

TIMESTAMPS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

RENAMES = {"products": {"product_name_lenght": "product_name_length",
                        "product_description_lenght": "product_description_length"}}

TABLE_DOCS = {
    "customers": "One row per customer_id (an order-scoped id). customer_unique_id identifies the real person.",
    "orders": "One row per order, with status and lifecycle timestamps.",
    "order_items": "One row per item in an order. Order revenue = SUM(price); shipping = SUM(freight_value).",
    "payments": "One row per payment method used on an order (orders can have several).",
    "reviews": "Customer review per order, score 1-5.",
    "products": "Product catalogue; category names are in Portuguese.",
    "sellers": "Marketplace sellers.",
    "geolocation": "Zip-code prefix to lat/lng; many rows per prefix.",
    "category_translation": "Portuguese to English product category names.",
}

COLUMN_DOCS = {
    "customers": {"customer_id": "key to orders.customer_id; unique per order",
                  "customer_unique_id": "the actual customer; use for repeat-purchase analysis",
                  "customer_state": "two-letter Brazilian state code, e.g. 'SP'"},
    "orders": {"order_status": "delivered, shipped, canceled, unavailable, invoiced, processing, created, approved",
               "order_purchase_timestamp": "when the order was placed",
               "order_delivered_customer_date": "actual delivery to customer; NULL if not delivered",
               "order_estimated_delivery_date": "promised delivery date; late = delivered after this"},
    "order_items": {"order_item_id": "sequence number of the item within the order (1, 2, ...)",
                    "price": "item price in BRL", "freight_value": "shipping cost for the item in BRL"},
    "payments": {"payment_type": "credit_card, boleto, voucher, debit_card, not_defined",
                 "payment_installments": "number of installments", "payment_value": "amount paid in BRL"},
    "reviews": {"review_score": "1 (worst) to 5 (best)"},
    "products": {"product_category_name": "Portuguese category; join category_translation for English"},
    "sellers": {"seller_state": "two-letter Brazilian state code"},
    "category_translation": {"product_category_name_english": "English category name"},
}

EXTRA_JOINS = [  # joins that don't follow the *_id naming convention
    {"table": "products", "column": "product_category_name",
     "ref_table": "category_translation", "ref_column": "product_category_name"},
    {"table": "customers", "column": "customer_zip_code_prefix",
     "ref_table": "geolocation", "ref_column": "geolocation_zip_code_prefix"},
    {"table": "sellers", "column": "seller_zip_code_prefix",
     "ref_table": "geolocation", "ref_column": "geolocation_zip_code_prefix"},
]


def load(csv_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    for table, fname in FILES.items():
        path = csv_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - download the Kaggle dataset into {csv_dir}")
        ts = TIMESTAMPS.get(table, [])
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto('{path.as_posix()}', header=true)")
        for old, new in RENAMES.get(table, {}).items():
            con.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
        for col in ts:
            con.execute(f"ALTER TABLE {table} ALTER COLUMN {col} TYPE TIMESTAMP USING TRY_CAST({col} AS TIMESTAMP)")
        con.execute(f"COMMENT ON TABLE {table} IS '{TABLE_DOCS[table]}'")
        for col, doc in COLUMN_DOCS.get(table, {}).items():
            con.execute(f"COMMENT ON COLUMN {table}.{col} IS '{doc.replace(chr(39), chr(39) * 2)}'")
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:22s} {n:>9,} rows")
    con.close()
    Path(str(out) + ".joins.json").write_text(json.dumps(EXTRA_JOINS, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="data/olist_csv")
    ap.add_argument("--out", default="data/olist.duckdb")
    a = ap.parse_args()
    load(Path(a.csv_dir), Path(a.out))
