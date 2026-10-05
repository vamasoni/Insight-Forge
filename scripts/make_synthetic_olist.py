"""Generate Olist-shaped CSVs (same file names and columns as the Kaggle release) for tests and CI.

The numbers are synthetic but have real structure (state effects, late deliveries lowering review
scores, installment behaviour) so the analysis tool and the critic have something to find.

    python scripts/make_synthetic_olist.py --out data/olist_csv_synth --orders 8000
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "PE", "CE", "AM"]
STATE_P = np.array([42, 13, 12, 6, 5, 4, 4, 2, 2, 2, 2, 1], float)
CITIES = {"SP": "sao paulo", "RJ": "rio de janeiro", "MG": "belo horizonte", "RS": "porto alegre",
          "PR": "curitiba", "SC": "florianopolis", "BA": "salvador", "DF": "brasilia", "GO": "goiania",
          "PE": "recife", "CE": "fortaleza", "AM": "manaus"}
CATS = {"cama_mesa_banho": "bed_bath_table", "beleza_saude": "health_beauty", "esporte_lazer": "sports_leisure",
        "informatica_acessorios": "computers_accessories", "moveis_decoracao": "furniture_decor",
        "utilidades_domesticas": "housewares", "relogios_presentes": "watches_gifts", "telefonia": "telephony",
        "brinquedos": "toys", "automotivo": "auto", "perfumaria": "perfumery", "eletronicos": "electronics"}
CAT_PRICE = dict(zip(CATS, [90, 130, 115, 120, 85, 90, 200, 70, 115, 140, 125, 95]))


def uid(rng) -> str:
    return uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))).hex


def main(out: Path, n_orders: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    zips = {s: rng.integers(1000, 99999, 8) for s in STATES}

    # sellers
    n_sellers = max(50, n_orders // 30)
    s_state = rng.choice(STATES, n_sellers, p=np.r_[60, 8, 12, 4, 6, 5, 1, 1, 1, 1, 0.5, 0.5] / 100)
    sellers = pd.DataFrame({"seller_id": [uid(rng) for _ in range(n_sellers)],
                            "seller_zip_code_prefix": [rng.choice(zips[s]) for s in s_state],
                            "seller_city": [CITIES[s] for s in s_state], "seller_state": s_state})

    # products
    n_products = max(200, n_orders // 4)
    cats = rng.choice(list(CATS), n_products)
    products = pd.DataFrame({
        "product_id": [uid(rng) for _ in range(n_products)], "product_category_name": cats,
        "product_name_lenght": rng.integers(10, 70, n_products),
        "product_description_lenght": rng.integers(50, 3000, n_products),
        "product_photos_qty": rng.integers(1, 8, n_products),
        "product_weight_g": rng.lognormal(6.5, 1.0, n_products).round(),
        "product_length_cm": rng.integers(10, 80, n_products), "product_height_cm": rng.integers(2, 60, n_products),
        "product_width_cm": rng.integers(8, 60, n_products)})
    base_price = products.product_category_name.map(CAT_PRICE).to_numpy() * rng.lognormal(0, 0.5, n_products)

    # customers: ~3% repeat buyers (same unique id, new customer_id per order, as in real Olist)
    n_people = int(n_orders * 0.97)
    people = [uid(rng) for _ in range(n_people)]
    person_state = rng.choice(STATES, n_people, p=STATE_P / STATE_P.sum())
    who = np.r_[np.arange(n_people), rng.integers(0, n_people, n_orders - n_people)]
    customers = pd.DataFrame({"customer_id": [uid(rng) for _ in range(n_orders)],
                              "customer_unique_id": [people[i] for i in who],
                              "customer_zip_code_prefix": [rng.choice(zips[person_state[i]]) for i in who],
                              "customer_city": [CITIES[person_state[i]] for i in who],
                              "customer_state": person_state[who]})

    # orders
    start = np.datetime64("2017-01-01")
    days = (rng.beta(2.2, 1.3, n_orders) * 600).astype(int)  # growth over time
    purchase = start + days.astype("timedelta64[D]") + rng.integers(0, 86400, n_orders).astype("timedelta64[s]")
    status = rng.choice(["delivered", "shipped", "canceled", "unavailable", "invoiced", "processing"], n_orders,
                        p=[0.965, 0.012, 0.007, 0.006, 0.005, 0.005])
    far = np.isin(customers.customer_state.to_numpy(), ["AM", "CE", "PE", "BA"])
    est = rng.integers(15, 30, n_orders) + far * 8
    actual = np.maximum(2, rng.gamma(4, 3, n_orders) + far * 9).astype(int)
    delivered = status == "delivered"
    orders = pd.DataFrame({
        "order_id": [uid(rng) for _ in range(n_orders)], "customer_id": customers.customer_id,
        "order_status": status, "order_purchase_timestamp": purchase,
        "order_approved_at": purchase + np.timedelta64(3, "h"),
        "order_delivered_carrier_date": np.where(delivered | (status == "shipped"),
                                                 purchase + np.timedelta64(2, "D"), np.datetime64("NaT")),
        "order_delivered_customer_date": np.where(delivered, purchase + actual.astype("timedelta64[D]"),
                                                  np.datetime64("NaT")),
        "order_estimated_delivery_date": (purchase + est.astype("timedelta64[D]")).astype("datetime64[D]")})
    late = delivered & (actual > est)

    # items
    n_items = rng.choice([1, 1, 1, 1, 1, 1, 1, 2, 2, 3], n_orders)
    rows = []
    for oid, k, ts in zip(orders.order_id, n_items, purchase):
        pidx = rng.integers(0, n_products, k)
        sid = sellers.seller_id.iloc[rng.integers(0, n_sellers)]
        for j, p in enumerate(pidx, 1):
            rows.append((oid, j, products.product_id.iloc[p], sid, ts + np.timedelta64(6, "D"),
                         round(float(base_price[p]), 2), round(float(8 + rng.gamma(2, 6)), 2)))
    items = pd.DataFrame(rows, columns=["order_id", "order_item_id", "product_id", "seller_id",
                                        "shipping_limit_date", "price", "freight_value"])

    # payments
    totals = items.groupby("order_id")[["price", "freight_value"]].sum().sum(axis=1)
    ptype = rng.choice(["credit_card", "boleto", "voucher", "debit_card"], n_orders, p=[0.74, 0.19, 0.055, 0.015])
    tot = totals.reindex(orders.order_id).to_numpy()
    inst = np.where(ptype == "credit_card", np.clip(np.round(tot / 60 + rng.normal(0, 1.5, n_orders)), 1, 10), 1)
    payments = pd.DataFrame({"order_id": orders.order_id, "payment_sequential": 1, "payment_type": ptype,
                             "payment_installments": inst.astype(int), "payment_value": tot.round(2)})

    # reviews: late delivery drags scores down
    p_good = np.where(late, 0.35, 0.82)
    score = np.where(rng.random(n_orders) < p_good, rng.choice([4, 5], n_orders, p=[0.3, 0.7]),
                     rng.choice([1, 2, 3], n_orders, p=[0.55, 0.15, 0.30]))
    reviews = pd.DataFrame({"review_id": [uid(rng) for _ in range(n_orders)], "order_id": orders.order_id,
                            "review_score": score, "review_comment_title": None, "review_comment_message": None,
                            "review_creation_date": (purchase + (actual + 1).astype("timedelta64[D]")).astype(
                                "datetime64[D]"),
                            "review_answer_timestamp": purchase + (actual + 3).astype("timedelta64[D]")})

    geo = pd.DataFrame([(z, -23.5 + rng.normal(0, 3), -46.6 + rng.normal(0, 3), CITIES[s], s)
                        for s in STATES for z in zips[s]],
                       columns=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
                                "geolocation_city", "geolocation_state"])
    trans = pd.DataFrame(list(CATS.items()), columns=["product_category_name", "product_category_name_english"])

    for name, df in [("olist_customers_dataset", customers), ("olist_orders_dataset", orders),
                     ("olist_order_items_dataset", items), ("olist_order_payments_dataset", payments),
                     ("olist_order_reviews_dataset", reviews), ("olist_products_dataset", products),
                     ("olist_sellers_dataset", sellers), ("olist_geolocation_dataset", geo),
                     ("product_category_name_translation", trans)]:
        df.to_csv(out / f"{name}.csv", index=False)
    print(f"wrote synthetic Olist CSVs to {out} ({n_orders} orders, {len(items)} items)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/olist_csv_synth")
    ap.add_argument("--orders", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(Path(a.out), a.orders, a.seed)
