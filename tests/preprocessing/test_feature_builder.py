import numpy as np
import pandas as pd
import pytest

from ppkt2synergy.core import PhenopacketRecord
from ppkt2synergy.preprocessing.feature_builder import HpoFeatureBuilder


class DummyHPOHierarchyEngine:
    def __init__(self, propagated_matrix=None, label_mapping=None, relationship_mask=None):
        self._propagated_matrix = propagated_matrix
        self._label_mapping = label_mapping or {}
        self._relationship_mask = relationship_mask
        self.propagate_called_with = None
        self.relationship_mask_called_with = None

    def propagate(self, matrix: pd.DataFrame) -> pd.DataFrame:
        self.propagate_called_with = matrix.copy()
        if self._propagated_matrix is not None:
            return self._propagated_matrix.copy()
        return matrix.copy()

    def get_labels(self) -> dict[str, str]:
        return dict(self._label_mapping)

    def build_relationship_mask(self, terms) -> pd.DataFrame:
        self.relationship_mask_called_with = list(terms)
        if self._relationship_mask is not None:
            return self._relationship_mask.copy()

        terms = list(terms)
        mask = pd.DataFrame(0.0, index=terms, columns=terms)
        for term in terms:
            mask.loc[term, term] = np.nan
        return mask


@pytest.fixture
def record1():
    return PhenopacketRecord(
        individual_id="P1",
        cohort="C1",
        observed_hpo_terms={"HP:1", "HP:2"},
        excluded_hpo_terms={"HP:3"},
        observed_diseases=set(),
        excluded_diseases=set(),
        sex="male",
        age=None,
        metadata={},
    )


@pytest.fixture
def record2():
    return PhenopacketRecord(
        individual_id="P2",
        cohort="C1",
        observed_hpo_terms={"HP:2"},
        excluded_hpo_terms=set(),
        observed_diseases=set(),
        excluded_diseases=set(),
        sex="female",
        age=None,
        metadata={},
    )


def test_init_empty_records_raises():
    engine = DummyHPOHierarchyEngine()

    with pytest.raises(ValueError) as excinfo:
        HpoFeatureBuilder(records=[], hpo_hierarchy=engine)

    assert "records cannot be empty" in str(excinfo.value)


def test_build_raw_matrix_basic(record1, record2):
    engine = DummyHPOHierarchyEngine()
    builder = HpoFeatureBuilder(records=[record1, record2], hpo_hierarchy=engine)

    matrix = builder.build_raw_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert list(matrix.index) == ["P1", "P2"]
    assert set(matrix.columns) == {"HP:1", "HP:2", "HP:3"}

    assert matrix.loc["P1", "HP:1"] == 1.0
    assert matrix.loc["P1", "HP:2"] == 1.0
    assert matrix.loc["P1", "HP:3"] == 0.0

    assert pd.isna(matrix.loc["P2", "HP:1"])
    assert matrix.loc["P2", "HP:2"] == 1.0
    assert pd.isna(matrix.loc["P2", "HP:3"])


def test_build_raw_matrix_preserves_record_order(record2, record1):
    engine = DummyHPOHierarchyEngine()
    builder = HpoFeatureBuilder(records=[record2, record1], hpo_hierarchy=engine)

    matrix = builder.build_raw_matrix()

    assert list(matrix.index) == ["P2", "P1"]


def test_build_raw_matrix_logs_conflicts_and_excluded_wins(caplog):
    engine = DummyHPOHierarchyEngine()
    record = PhenopacketRecord(
        individual_id="P1",
        cohort=None,
        observed_hpo_terms={"HP:1"},
        excluded_hpo_terms={"HP:1"},
        observed_diseases=set(),
        excluded_diseases=set(),
        sex=None,
        age=None,
        metadata={},
    )
    builder = HpoFeatureBuilder(records=[record], hpo_hierarchy=engine)

    with caplog.at_level("WARNING"):
        matrix = builder.build_raw_matrix()

    assert "conflicting HPO annotations" in caplog.text
    assert matrix.loc["P1", "HP:1"] == 0.0


def test_filter_by_missingness_invalid_threshold_low():
    matrix = pd.DataFrame({"HP:1": [1.0, np.nan]})

    with pytest.raises(ValueError) as excinfo:
        HpoFeatureBuilder.filter_by_missingness(matrix, missing_threshold=-0.1)

    assert "missing_threshold must be between 0 and 1" in str(excinfo.value)


def test_filter_by_missingness_invalid_threshold_high():
    matrix = pd.DataFrame({"HP:1": [1.0, np.nan]})

    with pytest.raises(ValueError) as excinfo:
        HpoFeatureBuilder.filter_by_missingness(matrix, missing_threshold=1.1)

    assert "missing_threshold must be between 0 and 1" in str(excinfo.value)


def test_filter_by_missingness_threshold_one_keeps_all_columns():
    matrix = pd.DataFrame(
        {
            "HP:1": [1.0, np.nan],
            "HP:2": [np.nan, np.nan],
        }
    )

    filtered = HpoFeatureBuilder.filter_by_missingness(matrix, missing_threshold=1.0)

    assert list(filtered.columns) == ["HP:1", "HP:2"]


def test_filter_by_missingness_threshold_zero_keeps_only_complete_columns():
    matrix = pd.DataFrame(
        {
            "HP:1": [1.0, 0.0],
            "HP:2": [1.0, np.nan],
            "HP:3": [np.nan, np.nan],
        }
    )

    filtered = HpoFeatureBuilder.filter_by_missingness(matrix, missing_threshold=0.0)

    assert list(filtered.columns) == ["HP:1"]


def test_filter_by_missingness_intermediate_threshold():
    matrix = pd.DataFrame(
        {
            "HP:1": [1.0, np.nan, np.nan],   # 1 non-missing
            "HP:2": [1.0, 0.0, np.nan],      # 2 non-missing
            "HP:3": [1.0, 0.0, 1.0],         # 3 non-missing
        }
    )

    filtered = HpoFeatureBuilder.filter_by_missingness(matrix, missing_threshold=0.5)

    # len(matrix)=3, min_non_missing=int((1-0.5)*3)=1
    assert list(filtered.columns) == ["HP:1", "HP:2", "HP:3"]


def test_build_raises_when_no_hpo_terms_found():
    engine = DummyHPOHierarchyEngine()
    records = [
        PhenopacketRecord(
            individual_id="P1",
            cohort=None,
            observed_hpo_terms=set(),
            excluded_hpo_terms=set(),
            observed_diseases=set(),
            excluded_diseases=set(),
            sex=None,
            age=None,
            metadata={},
        )
    ]
    builder = HpoFeatureBuilder(records=records, hpo_hierarchy=engine)

    with pytest.raises(ValueError) as excinfo:
        builder.build()

    assert "No HPO terms found in input records" in str(excinfo.value)


def test_build_raises_when_no_terms_remain_after_filtering(record1):
    propagated = pd.DataFrame(
        {"HP:1": [np.nan]},
        index=pd.Index(["P1"], name="individual_id"),
    )
    engine = DummyHPOHierarchyEngine(propagated_matrix=propagated)
    builder = HpoFeatureBuilder(records=[record1], hpo_hierarchy=engine)

    with pytest.raises(ValueError) as excinfo:
        builder.build(missing_threshold=0.0)

    assert "No HPO terms remain after propagation and missingness filtering" in str(excinfo.value)


def test_build_returns_hpo_feature_data(record1, record2):
    propagated = pd.DataFrame(
        {
            "HP:1": [1.0, np.nan],
            "HP:2": [1.0, 1.0],
        },
        index=pd.Index(["P1", "P2"], name="individual_id"),
    )
    relationship_mask = pd.DataFrame(
        [[np.nan, 0.0], [0.0, np.nan]],
        index=["HP:1", "HP:2"],
        columns=["HP:1", "HP:2"],
    )
    engine = DummyHPOHierarchyEngine(
        propagated_matrix=propagated,
        label_mapping={"HP:1": "Label 1", "HP:2": "Label 2"},
        relationship_mask=relationship_mask,
    )
    builder = HpoFeatureBuilder(records=[record1, record2], hpo_hierarchy=engine)

    feature_data = builder.build(missing_threshold=1.0)

    assert feature_data.matrix.equals(propagated)
    assert feature_data.label_mapping == {"HP:1": "Label 1", "HP:2": "Label 2"}
    assert feature_data.relationship_mask.equals(relationship_mask)


def test_build_calls_engine_with_expected_inputs(record1, record2):
    engine = DummyHPOHierarchyEngine(
        label_mapping={"HP:1": "Label 1", "HP:2": "Label 2", "HP:3": "Label 3"}
    )
    builder = HpoFeatureBuilder(records=[record1, record2], hpo_hierarchy=engine)

    feature_data = builder.build(missing_threshold=1.0)

    assert isinstance(engine.propagate_called_with, pd.DataFrame)
    assert list(engine.propagate_called_with.index) == ["P1", "P2"]

    assert engine.relationship_mask_called_with == list(feature_data.matrix.columns)