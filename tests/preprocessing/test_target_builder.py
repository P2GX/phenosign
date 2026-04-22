import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from ppkt2synergy.core import PhenopacketRecord
from ppkt2synergy.preprocessing.target_builder import TargetDataBuilder


class DummyHierarchyEngine:
    def __init__(self, hpo="dummy_hpo"):
        self.term_manager = SimpleNamespace(hpo=hpo)


@pytest.fixture
def record1():
    return PhenopacketRecord(
        individual_id="P1",
        cohort="C1",
        observed_hpo_terms=set(),
        excluded_hpo_terms=set(),
        observed_diseases={"DiseaseA"},
        excluded_diseases={"DiseaseB"},
        sex="male",
        age=None,
        metadata={},
    )


@pytest.fixture
def record2():
    return PhenopacketRecord(
        individual_id="P2",
        cohort="C1",
        observed_hpo_terms=set(),
        excluded_hpo_terms=set(),
        observed_diseases=set(),
        excluded_diseases={"DiseaseA"},
        sex="female",
        age=None,
        metadata={},
    )


def test_init_empty_records_raises():
    with pytest.raises(ValueError) as excinfo:
        TargetDataBuilder(records=[])

    assert "records cannot be empty" in str(excinfo.value)


def test_init_sets_individual_index(record1, record2):
    builder = TargetDataBuilder(records=[record2, record1])

    assert list(builder.individual_index) == ["P2", "P1"]
    assert builder.individual_index.name == "individual_id"


def test_build_disease_matrix_fill_unknown_with_zero(record1, record2):
    builder = TargetDataBuilder(records=[record1, record2])

    matrix = builder.build_disease_matrix(fill_unknown_with_zero=True)

    assert isinstance(matrix, pd.DataFrame)
    assert list(matrix.index) == ["P1", "P2"]
    assert set(matrix.columns) == {"DiseaseA", "DiseaseB"}

    assert matrix.loc["P1", "DiseaseA"] == 1.0
    assert matrix.loc["P1", "DiseaseB"] == 0.0

    # P2 does not explicitly observe DiseaseB, unknown is encoded as 0
    assert matrix.loc["P2", "DiseaseA"] == 0.0
    assert matrix.loc["P2", "DiseaseB"] == 0.0


def test_build_disease_matrix_keep_unknown_as_nan(record1, record2):
    builder = TargetDataBuilder(records=[record1, record2])

    matrix = builder.build_disease_matrix(fill_unknown_with_zero=False)

    assert matrix.loc["P1", "DiseaseA"] == 1.0
    assert matrix.loc["P1", "DiseaseB"] == 0.0

    assert matrix.loc["P2", "DiseaseA"] == 0.0
    assert pd.isna(matrix.loc["P2", "DiseaseB"])


def test_build_disease_matrix_preserves_record_order(record1, record2):
    builder = TargetDataBuilder(records=[record2, record1])

    matrix = builder.build_disease_matrix()

    assert list(matrix.index) == ["P2", "P1"]


def test_build_disease_matrix_observed_overrides_excluded():
    record = PhenopacketRecord(
        individual_id="P1",
        cohort=None,
        observed_hpo_terms=set(),
        excluded_hpo_terms=set(),
        observed_diseases={"DiseaseA"},
        excluded_diseases={"DiseaseA"},
        sex=None,
        age=None,
        metadata={},
    )
    builder = TargetDataBuilder(records=[record])

    matrix = builder.build_disease_matrix(fill_unknown_with_zero=False)

    assert matrix.loc["P1", "DiseaseA"] == 1.0


def test_build_returns_target_data_without_variant_matrix(record1, record2):
    builder = TargetDataBuilder(records=[record1, record2])

    target_data = builder.build()

    assert target_data.disease_matrix is not None
    assert target_data.variant_condition_matrix is None
    assert isinstance(target_data.disease_matrix, pd.DataFrame)


def test_build_warns_when_variant_args_incomplete(record1, caplog):
    builder = TargetDataBuilder(records=[record1])

    with caplog.at_level("WARNING"):
        target_data = builder.build(variant_effect_type="not_used")

    assert target_data.variant_condition_matrix is None
    assert "Variant-condition matrix was not built" in caplog.text


def test_build_variant_condition_matrix_requires_raw_phenopackets(record1):
    builder = TargetDataBuilder(
        records=[record1],
        raw_phenopackets=None,
        hpo_hierarchy=DummyHierarchyEngine(),
    )

    with pytest.raises(ValueError) as excinfo:
        builder.build_variant_condition_matrix(
            variant_effect_type=object(),
            mane_tx_id="NM_000000.1",
        )

    assert "Raw phenopackets are required" in str(excinfo.value)


def test_build_variant_condition_matrix_requires_hpo(record1):
    builder = TargetDataBuilder(
        records=[record1],
        raw_phenopackets=[SimpleNamespace(id="P1")],
        hpo_hierarchy=None,
    )

    with pytest.raises(ValueError) as excinfo:
        builder.build_variant_condition_matrix(
            variant_effect_type=object(),
            mane_tx_id="NM_000000.1",
        )

    assert "HPO ontology object is required" in str(excinfo.value)