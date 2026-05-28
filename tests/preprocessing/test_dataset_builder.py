import numpy as np
import pandas as pd
import pytest

from phenopackets import (
    Phenopacket,
    PhenotypicFeature,
    OntologyClass,
)

from ppkt2synergy.preprocessing.dataset_builder import (
    PhenotypeDatasetBuilder,
    _HpoObservation,
)


# =========================
# helpers
# =========================

def make_feature(term_id: str, excluded: bool = False):
    return PhenotypicFeature(
        type=OntologyClass(id=term_id, label=term_id),
        excluded=excluded,
    )


def make_phenopacket(
    pid: str,
    features: list[PhenotypicFeature],
):
    return Phenopacket(
        id=pid,
        phenotypic_features=features,
    )


# =========================
# validation
# =========================

def test_validate_empty_phenopackets():
    with pytest.raises(ValueError, match="cannot be empty"):
        PhenotypeDatasetBuilder([])


def test_validate_invalid_type():
    with pytest.raises(TypeError, match="Phenopacket"):
        PhenotypeDatasetBuilder([123])


def test_validate_duplicate_ids():
    pp1 = make_phenopacket("P1", [])
    pp2 = make_phenopacket("P1", [])

    with pytest.raises(ValueError, match="Duplicate phenopacket id"):
        PhenotypeDatasetBuilder([pp1, pp2])


# =========================
# parsing
# =========================

def test_parse_hpo_observations():
    pp = make_phenopacket(
        "P1",
        [
            make_feature("HP:1"),
            make_feature("HP:2", excluded=True),
        ],
    )

    observations = (
        PhenotypeDatasetBuilder
        ._parse_hpo_observations([pp])
    )

    obs = observations[0]

    assert isinstance(obs, _HpoObservation)
    assert obs.individual_id == "P1"
    assert obs.observed_terms == frozenset({"HP:1"})
    assert obs.excluded_terms == frozenset({"HP:2"})


def test_parse_conflicting_terms():
    pp = make_phenopacket(
        "P1",
        [
            make_feature("HP:1"),
            make_feature("HP:1", excluded=True),
        ],
    )

    observations = (
        PhenotypeDatasetBuilder
        ._parse_hpo_observations([pp])
    )

    obs = observations[0]

    assert obs.observed_terms == frozenset({"HP:1"})
    assert obs.excluded_terms == frozenset()


# =========================
# matrix building
# =========================

def test_build_raw_hpo_matrix():
    observations = [
        _HpoObservation(
            individual_id="P1",
            observed_terms=frozenset({"HP:1"}),
            excluded_terms=frozenset({"HP:2"}),
        ),
        _HpoObservation(
            individual_id="P2",
            observed_terms=frozenset(),
            excluded_terms=frozenset({"HP:1"}),
        ),
    ]

    matrix = (
        PhenotypeDatasetBuilder
        ._build_raw_hpo_matrix(observations)
    )

    assert isinstance(matrix, pd.DataFrame)

    assert matrix.loc["P1", "HP:1"] == 1
    assert matrix.loc["P1", "HP:2"] == 0
    assert matrix.loc["P2", "HP:1"] == 0

    assert np.isnan(matrix.loc["P2", "HP:2"])


# =========================
# missingness filter
# =========================

def test_filter_by_missingness():
    matrix = pd.DataFrame(
        {
            "HP:1": [1, np.nan],
            "HP:2": [1, 0],
        }
    )

    filtered = (
        PhenotypeDatasetBuilder
        ._filter_by_missingness(
            matrix,
            missing_threshold=0.4,
        )
    )

    assert "HP:1" not in filtered.columns
    assert "HP:2" in filtered.columns


@pytest.mark.parametrize(
    "threshold",
    [-1, 2],
)
def test_filter_by_missingness_invalid_range(threshold):
    matrix = pd.DataFrame({"HP:1": [1]})

    with pytest.raises(ValueError):
        PhenotypeDatasetBuilder._filter_by_missingness(
            matrix,
            missing_threshold=threshold,
        )


# =========================
# build
# =========================

def test_build_returns_dataset(monkeypatch):
    pp = make_phenopacket(
        "P1",
        [make_feature("HP:0000118")],
    )

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, matrix):
            return matrix

        def get_labels(self):
            return {"HP:0000118": "Phenotypic abnormality"}

        def build_relationship_mask(self, columns):
            return pd.DataFrame(
                np.nan,
                index=columns,
                columns=columns,
            )

    from ppkt2synergy.preprocessing import dataset_builder as module

    monkeypatch.setattr(
        module,
        "HPOHierarchyEngine",
        DummyEngine,
    )

    builder = PhenotypeDatasetBuilder([pp])

    dataset = builder.build(
        build_gpsea_cohort=False,
    )

    assert hasattr(dataset, "hpo_data")
    assert hasattr(dataset, "phenopackets")

    assert dataset.phenopackets == [pp]
    assert "HP:0000118" in dataset.hpo_data.matrix.columns