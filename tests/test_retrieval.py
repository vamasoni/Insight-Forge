from insightforge.retrieval import SchemaRetriever, tokenize


def test_tokenize():
    assert tokenize("orderItems freight_values") == ["order", "item", "freight", "value"]


def test_retrieval_and_values(db):
    r = SchemaRetriever(db).retrieve("What share of orders paid with boleto were delivered late?", top_k=3)
    assert {"orders", "payments"} <= set(r.tables)
    assert ("payments", "payment_type", "boleto") in r.matched_values


def test_join_path_completion(db):
    # category_translation reaches sellers only via products -> order_items; both must be added
    out = SchemaRetriever(db)._complete_join_paths(["category_translation", "sellers"])
    assert {"products", "order_items"} <= set(out)


def test_revenue_question_with_enough_tables(db):
    r = SchemaRetriever(db).retrieve("revenue by product category english name", top_k=4)
    assert {"category_translation", "products", "order_items"} <= set(r.tables)
