from tempotrack_v10.cov_category_ontology import build_category_mapping


def test_cov_labels_bind_by_official_name_and_keep_unknowns_negative(tmp_path):
    source = tmp_path / "cov"
    class_file = source / "data/lvis/annotations/lvis_classes_v1.txt"
    class_file.parent.mkdir(parents=True)
    class_file.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    annotation = {
        "categories": [
            {"id": 41, "name": "alpha"},
            {"id": 99, "name": "gamma"},
        ]
    }

    category_ids, metadata = build_category_mapping(
        annotation=annotation,
        cov_source=source,
    )

    assert category_ids == [41, -2, 99]
    assert metadata["mapped_category_count"] == 2
    assert metadata["unknown_category_count"] == 1
    assert metadata["unknown_categories"] == [
        {"label": 1, "name": "beta", "sentinel_category_id": -2}
    ]
