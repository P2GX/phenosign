import pandas as pd
import pytest

from phenosign.analysis.hpo_correlation_analyzer import (
    HPOCorrelationAnalyzer,
)
from phenosign.core import (
    PhenotypeDataset,
)

from phenosign.core.features_data import HpoFeatureData


@pytest.fixture
def large_dataset():
    n_samples = 60

    matrix = pd.DataFrame(
        {
            "HP:0001": [1] * 30 + [0] * 30,
            "HP:0002": [0] * 10 + [1] * 30 + [0] * 20,
            "HP:0003": [0] * 20 + [1] * 30 + [0] * 10,
            "HP:0004": [0] * 15 + [1] * 30 + [0] * 15,
        },
        index=[f"Patient_{i}" for i in range(n_samples)],
    )

    return PhenotypeDataset(
        hpo_data=HpoFeatureData(
            matrix=matrix,
            label_mapping={c: c for c in matrix.columns},
            relationship_mask=None,
        ),
        phenopackets=[],
    )


@pytest.fixture
def strongly_correlated_dataset():
    n_samples = 80

    hp1 = [1] * 40 + [0] * 40
    hp2 = [1] * 40 + [0] * 40
    hp3 = [0] * 40 + [1] * 40
    hp4 = [1, 0] * 40

    matrix = pd.DataFrame(
        {
            "HP:0001": hp1,
            "HP:0002": hp2,
            "HP:0003": hp3,
            "HP:0004": hp4,
        },
        index=[f"Patient_{i}" for i in range(n_samples)],
    )

    return PhenotypeDataset(
        hpo_data=HpoFeatureData(
            matrix=matrix,
            label_mapping={c: c for c in matrix.columns},
            relationship_mask=None,
        ),
        phenopackets=[],
    )


def test_initialization(large_dataset):
    analyzer = HPOCorrelationAnalyzer(
        large_dataset,
        min_individuals_for_correlation_test=10,
    )

    assert analyzer.hpo_matrix.shape == (60, 4)
    assert analyzer.n_features == 4


def test_compute_correlation_matrix(large_dataset):
    analyzer = HPOCorrelationAnalyzer(
        large_dataset,
        min_individuals_for_correlation_test=10,
    )

    results = analyzer.compute_correlation_matrix(
        n_jobs=1,
        include_pmids=False,
    )

    assert isinstance(results.results_table, pd.DataFrame)

    if not results.results_table.empty:
        expected = {
            "HPO_A",
            "HPO_B",
            "correlation",
            "p_value",
            "adj_p_value",
        }

        assert expected.issubset(results.results_table.columns)


def test_compute_correlation_matrix_happy_path(
    strongly_correlated_dataset,
):
    analyzer = HPOCorrelationAnalyzer(
        strongly_correlated_dataset,
        min_individuals_for_correlation_test=10,
    )

    results = analyzer.compute_correlation_matrix(
        n_jobs=1,
        include_pmids=False,
    )

    assert not results.results_table.empty


def test_filter_weak_correlations(
    strongly_correlated_dataset,
):
    analyzer = HPOCorrelationAnalyzer(
        strongly_correlated_dataset,
        min_individuals_for_correlation_test=10,
    )

    results = analyzer.compute_correlation_matrix(
        n_jobs=1,
        include_pmids=False,
    )

    result = results.filter_weak_correlations(
        corr_threshold=0.0,
        adj_pval_threshold=1.0,
    )

    assert result is not None

    coef, pval = result

    assert isinstance(coef, pd.DataFrame)
    assert isinstance(pval, pd.DataFrame)
