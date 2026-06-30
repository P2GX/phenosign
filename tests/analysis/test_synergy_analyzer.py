import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from phenosign import SynergyAnalyzer
from phenosign.core import PhenotypeDataset

from phenosign.core.features_data import HpoFeatureData


def _make_phenopackets(ids: list[str]) -> list:
    """Create minimal mock Phenopacket objects with matching IDs."""
    phenopackets = []
    for pid in ids:
        ppkt = MagicMock()
        ppkt.id = pid
        ppkt.diseases = []
        ppkt.interpretations = []
        phenopackets.append(ppkt)
    return phenopackets


@pytest.fixture
def dataset():
    np.random.seed(40)
    n = 40
    idx = [f"P{i}" for i in range(n)]

    hpo = pd.DataFrame(
        {
            "HPO1": np.random.randint(0, 2, n),
            "HPO2": np.random.randint(0, 2, n),
            "HPO3": np.random.randint(0, 2, n),
            "HPO4": np.random.randint(0, 2, n),
        },
        index=idx,
    )

    return PhenotypeDataset(
        hpo_data=HpoFeatureData(
            matrix=hpo,
            label_mapping={c: c for c in hpo.columns},
            relationship_mask=None,
        ),
        phenopackets=_make_phenopackets(idx),
    )


@pytest.fixture
def condition(dataset):
    def alternating(phenopacket) -> bool | None:
        index = int(phenopacket.id.lstrip("P"))
        return index % 2 == 0

    return dataset.get_condition(alternating, name="test_condition")

def test_init(dataset):
    analyzer = SynergyAnalyzer(dataset)

    assert analyzer.n_features == 4
    assert analyzer.X.shape[0] == 40


def test_compute_synergy(dataset, condition):
    analyzer = SynergyAnalyzer(dataset)

    res = analyzer.compute_synergy_matrix(
        condition=condition,
        n_jobs=1,
    )

    assert isinstance(res.results_table, pd.DataFrame)


def test_filter_synergy(dataset, condition):
    analyzer = SynergyAnalyzer(dataset)

    results = analyzer.compute_synergy_matrix(condition=condition, n_jobs=1)

    out = results.filter_weak_synergy(
        synergy_threshold=0.0,
        adj_pval_threshold=1.0,
    )

    if out is not None:
        s, p = out
        assert isinstance(s, pd.DataFrame)
        assert isinstance(p, pd.DataFrame)
