import pytest
import pandas as pd
import plotly.graph_objects as go

from ppkt2synergy.analysis.hpo_correlation_analyzer import HPOCorrelationAnalyzer
from ppkt2synergy.analysis.correlation_type import CorrelationType
from ppkt2synergy.core import HpoFeatureData, TargetData, PhenotypeDataset


@pytest.fixture
def large_dataset():
    n_samples = 60
    index = [f"Patient_{i}" for i in range(n_samples)]

    hpo_matrix = pd.DataFrame(
        {
            "HP:0001": [1] * 30 + [0] * 30,
            "HP:0002": [0] * 10 + [1] * 30 + [0] * 20,
            "HP:0003": [0] * 20 + [1] * 30 + [0] * 10,
            "HP:0004": [0] * 15 + [1] * 30 + [0] * 15,
        },
        index=index,
    )

    hpo_data = HpoFeatureData(
        matrix=hpo_matrix,
        label_mapping={col: col for col in hpo_matrix.columns},
        relationship_mask=None,
    )

    targets = TargetData(
        disease_matrix=pd.DataFrame(
            {"dummy_target": [0.0] * n_samples},
            index=index,
        ),
        variant_condition_matrix=None,
    )

    individual_metadata = pd.DataFrame(
        {"cohort": ["C1"] * n_samples},
        index=index,
    )

    return PhenotypeDataset(
        hpo_data=hpo_data,
        targets=targets,
        individual_metadata=individual_metadata,
    )


@pytest.fixture
def strongly_correlated_dataset():
    """
    Construct a dataset with at least one very obvious valid pair.
    """
    n_samples = 80
    index = [f"Patient_{i}" for i in range(n_samples)]

    hp1 = [1] * 40 + [0] * 40
    hp2 = [1] * 40 + [0] * 40   # identical to hp1 -> very strong positive correlation
    hp3 = [0] * 40 + [1] * 40   # inverse pattern
    hp4 = [1, 0] * 40           # alternating

    hpo_matrix = pd.DataFrame(
        {
            "HP:0001": hp1,
            "HP:0002": hp2,
            "HP:0003": hp3,
            "HP:0004": hp4,
        },
        index=index,
    )

    hpo_data = HpoFeatureData(
        matrix=hpo_matrix,
        label_mapping={col: col for col in hpo_matrix.columns},
        relationship_mask=None,
    )

    targets = TargetData(
        disease_matrix=pd.DataFrame(
            {"dummy_target": [0.0] * n_samples},
            index=index,
        ),
        variant_condition_matrix=None,
    )

    individual_metadata = pd.DataFrame(
        {"cohort": ["C1"] * n_samples},
        index=index,
    )

    return PhenotypeDataset(
        hpo_data=hpo_data,
        targets=targets,
        individual_metadata=individual_metadata,
    )


def test_initialization(large_dataset):
    analyzer = HPOCorrelationAnalyzer(
        large_dataset,
        min_individuals_for_correlation_test=10,
    )

    assert analyzer.hpo_matrix.shape[0] == 60
    assert analyzer.n_features == 4
    assert list(analyzer.hpo_terms) == ["HP:0001", "HP:0002", "HP:0003", "HP:0004"]


def test_compute_correlation_matrix_returns_dataframe(large_dataset):
    analyzer = HPOCorrelationAnalyzer(
        large_dataset,
        min_individuals_for_correlation_test=10,
    )

    results = analyzer.compute_correlation_matrix(
        correlation_type=CorrelationType.SPEARMAN,
        n_jobs=1,
        include_pmids=False
    )

    assert isinstance(results, pd.DataFrame)

    # allow empty result for this fixture, but if non-empty it must have expected columns
    if not results.empty:
        assert "HPO_A" in results.columns
        assert "HPO_B" in results.columns
        assert "correlation" in results.columns
        assert "p_value" in results.columns
        assert "p_value_corrected" in results.columns


def test_compute_correlation_matrix_happy_path(strongly_correlated_dataset):
    analyzer = HPOCorrelationAnalyzer(
        strongly_correlated_dataset,
        min_individuals_for_correlation_test=10,
    )

    results = analyzer.compute_correlation_matrix(
        correlation_type=CorrelationType.SPEARMAN,
        n_jobs=1,
        include_pmids=False
    )

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "HPO_A" in results.columns
    assert "HPO_B" in results.columns
    assert "correlation" in results.columns
    assert "p_value" in results.columns
    assert "p_value_corrected" in results.columns


def test_filter_weak_correlations_returns_none_or_pair(large_dataset):
    analyzer = HPOCorrelationAnalyzer(
        large_dataset,
        min_individuals_for_correlation_test=10,
    )

    analyzer.compute_correlation_matrix(
        correlation_type=CorrelationType.SPEARMAN,
        n_jobs=1,
        include_pmids=False
    )

    result = analyzer.filter_weak_correlations(
        abs_threshold=0.0,
        adj_pval_threshold=1.0,
    )

    if result is None:
        assert result is None
    else:
        coef_cleaned, pval_cleaned = result
        assert isinstance(coef_cleaned, pd.DataFrame)
        assert isinstance(pval_cleaned, pd.DataFrame)


def test_filter_weak_correlations_happy_path(strongly_correlated_dataset):
    analyzer = HPOCorrelationAnalyzer(
        strongly_correlated_dataset,
        min_individuals_for_correlation_test=10,
    )

    analyzer.compute_correlation_matrix(
        correlation_type=CorrelationType.SPEARMAN,
        n_jobs=1,
        include_pmids=False
    )

    result = analyzer.filter_weak_correlations(
        abs_threshold=0.0,
        adj_pval_threshold=1.0,
    )

    assert result is not None
    coef_cleaned, pval_cleaned = result

    assert isinstance(coef_cleaned, pd.DataFrame)
    assert isinstance(pval_cleaned, pd.DataFrame)


def test_plot_heatmap_happy_path(strongly_correlated_dataset):
    analyzer = HPOCorrelationAnalyzer(
        strongly_correlated_dataset,
        min_individuals_for_correlation_test=10,
    )

    analyzer.compute_correlation_matrix(
        correlation_type=CorrelationType.SPEARMAN,
        n_jobs=1,
        include_pmids=False
    )

    fig = analyzer.plot_correlation_heatmap_with_significance(
        stats_name="spearman",
        abs_threshold=0.0,
        adj_pval_threshold=1.0,
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)