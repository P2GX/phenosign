from types import SimpleNamespace

import pytest

from ppkt2synergy.io.phenopacket_loader import EnrichedPhenopacket
from ppkt2synergy.preprocessing.adapters import (
    phenopacket_to_record,
    phenopackets_to_records,
    enriched_phenopackets_to_records,
)


def make_feature(term_id=None, excluded=False):
    feature_type = None if term_id is None else SimpleNamespace(id=term_id)
    return SimpleNamespace(type=feature_type, excluded=excluded)


def make_disease(label=None, excluded=False):
    term = None if label is None else SimpleNamespace(label=label)
    return SimpleNamespace(term=term, excluded=excluded)


def make_external_reference(ref_id):
    return SimpleNamespace(id=ref_id)


def make_subject(sex=None, age=None):
    if age is None:
        encounter = None
    else:
        encounter = SimpleNamespace(
            age=SimpleNamespace(iso8601duration=age)
        )
    return SimpleNamespace(
        sex=sex,
        time_at_last_encounter=encounter,
    )


def make_phenopacket(
    ppkt_id="P1",
    phenotypic_features=None,
    diseases=None,
    subject=None,
    external_references=None,
):
    if phenotypic_features is None:
        phenotypic_features = []
    if diseases is None:
        diseases = []
    if external_references is None:
        external_references = []

    meta_data = SimpleNamespace(external_references=external_references)

    return SimpleNamespace(
        id=ppkt_id,
        phenotypic_features=phenotypic_features,
        diseases=diseases,
        subject=subject,
        meta_data=meta_data,
    )


def test_phenopacket_to_record_basic():
    phenopacket = make_phenopacket(
        ppkt_id="P1",
        phenotypic_features=[
            make_feature("HP:0001250", excluded=False),
            make_feature("HP:0004322", excluded=True),
        ],
        diseases=[
            make_disease("Marfan syndrome", excluded=False),
            make_disease("Loeys-Dietz syndrome", excluded=True),
        ],
        subject=make_subject(sex=2, age="P10Y"),
        external_references=[
            make_external_reference("PMID:12345"),
            make_external_reference("PMID:67890"),
            make_external_reference("PMID:12345"),
            make_external_reference("NOTPMID:1"),
        ],
    )

    record = phenopacket_to_record(phenopacket, cohort="FBN1")

    assert record.individual_id == "P1"
    assert record.cohort == "FBN1"

    assert record.observed_hpo_terms == {"HP:0001250"}
    assert record.excluded_hpo_terms == {"HP:0004322"}

    assert record.observed_diseases == {"Marfan syndrome"}
    assert record.excluded_diseases == {"Loeys-Dietz syndrome"}

    assert record.sex == "male"
    assert record.age == "P10Y"

    assert record.metadata == {"pmids": ["12345", "67890"]}


def test_phenopacket_to_record_skips_invalid_features_and_diseases():
    phenopacket = make_phenopacket(
        ppkt_id="P2",
        phenotypic_features=[
            make_feature(None, excluded=False),
            SimpleNamespace(type=None, excluded=False),
            make_feature("HP:0001250", excluded=False),
        ],
        diseases=[
            make_disease(None, excluded=False),
            SimpleNamespace(term=None, excluded=False),
            make_disease("Marfan syndrome", excluded=False),
        ],
    )

    record = phenopacket_to_record(phenopacket)

    assert record.observed_hpo_terms == {"HP:0001250"}
    assert record.excluded_hpo_terms == set()
    assert record.observed_diseases == {"Marfan syndrome"}
    assert record.excluded_diseases == set()


@pytest.mark.parametrize(
    ("sex_value", "expected"),
    [
        (1, "female"),
        (2, "male"),
        (0, None),
        (99, None),
        (None, None),
    ],
)
def test_phenopacket_to_record_sex_mapping(sex_value, expected):
    phenopacket = make_phenopacket(
        ppkt_id="P3",
        subject=make_subject(sex=sex_value),
    )

    record = phenopacket_to_record(phenopacket)

    assert record.sex == expected


def test_phenopacket_to_record_missing_subject_age_and_metadata():
    phenopacket = make_phenopacket(
        ppkt_id="P4",
        subject=None,
        external_references=[],
    )

    record = phenopacket_to_record(phenopacket)

    assert record.sex is None
    assert record.age is None
    assert record.metadata == {"pmids": []}


def test_phenopacket_to_record_missing_metadata_object():
    phenopacket = SimpleNamespace(
        id="P5",
        phenotypic_features=[],
        diseases=[],
        subject=None,
        meta_data=None,
    )

    record = phenopacket_to_record(phenopacket)

    assert record.individual_id == "P5"
    assert record.metadata == {"pmids": []}


def test_phenopackets_to_records_basic():
    phenopackets = [
        make_phenopacket(ppkt_id="P1"),
        make_phenopacket(ppkt_id="P2"),
    ]

    records = phenopackets_to_records(phenopackets, cohort="COHORT_A")

    assert len(records) == 2
    assert [r.individual_id for r in records] == ["P1", "P2"]
    assert all(r.cohort == "COHORT_A" for r in records)


def test_phenopackets_to_records_skips_duplicate_ids(caplog):
    phenopackets = [
        make_phenopacket(ppkt_id="P1"),
        make_phenopacket(ppkt_id="P1"),
        make_phenopacket(ppkt_id="P2"),
    ]

    with caplog.at_level("WARNING"):
        records = phenopackets_to_records(phenopackets)

    assert len(records) == 2
    assert [r.individual_id for r in records] == ["P1", "P2"]
    assert "Skipping duplicate phenopacket P1" in caplog.text


def test_phenopackets_to_records_skips_invalid_phenopacket_and_continues(caplog):
    valid = make_phenopacket(ppkt_id="P1")
    invalid = SimpleNamespace(
        id="BAD",
        phenotypic_features=None,  # this will break iteration in phenopacket_to_record
        diseases=[],
        subject=None,
        meta_data=None,
    )
    valid2 = make_phenopacket(ppkt_id="P2")

    with caplog.at_level("WARNING"):
        records = phenopackets_to_records([valid, invalid, valid2])

    assert len(records) == 2
    assert [r.individual_id for r in records] == ["P1", "P2"]
    assert "Skipping phenopacket BAD" in caplog.text


def test_enriched_phenopackets_to_records_basic():
    enriched = [
        EnrichedPhenopacket(
            phenopacket=make_phenopacket(ppkt_id="P1"),
            cohort="C1",
        ),
        EnrichedPhenopacket(
            phenopacket=make_phenopacket(ppkt_id="P2"),
            cohort="C2",
        ),
    ]

    records = enriched_phenopackets_to_records(enriched)

    assert len(records) == 2
    assert [r.individual_id for r in records] == ["P1", "P2"]
    assert [r.cohort for r in records] == ["C1", "C2"]


def test_enriched_phenopackets_to_records_skips_duplicates(caplog):
    shared = make_phenopacket(ppkt_id="P1")
    enriched = [
        EnrichedPhenopacket(phenopacket=shared, cohort="C1"),
        EnrichedPhenopacket(phenopacket=shared, cohort="C2"),
    ]

    with caplog.at_level("WARNING"):
        records = enriched_phenopackets_to_records(enriched)

    assert len(records) == 1
    assert records[0].individual_id == "P1"
    assert records[0].cohort == "C1"
    assert "Skipping duplicate enriched phenopacket P1" in caplog.text


def test_enriched_phenopackets_to_records_skips_invalid_and_continues(caplog):
    valid = EnrichedPhenopacket(
        phenopacket=make_phenopacket(ppkt_id="P1"),
        cohort="C1",
    )
    invalid = EnrichedPhenopacket(
        phenopacket=SimpleNamespace(
            id="BAD",
            phenotypic_features=None,
            diseases=[],
            subject=None,
            meta_data=None,
        ),
        cohort="C2",
    )
    valid2 = EnrichedPhenopacket(
        phenopacket=make_phenopacket(ppkt_id="P2"),
        cohort="C3",
    )

    with caplog.at_level("WARNING"):
        records = enriched_phenopackets_to_records([valid, invalid, valid2])

    assert len(records) == 2
    assert [r.individual_id for r in records] == ["P1", "P2"]
    assert [r.cohort for r in records] == ["C1", "C3"]
    assert "Skipping enriched phenopacket BAD" in caplog.text