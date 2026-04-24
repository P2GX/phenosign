import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ppkt2synergy import SynergyAnalyzer
from ppkt2synergy.core import HpoFeatureData, TargetData, PhenotypeDataset


@pytest.fixture
def large_dataset():
    n_samples = 40
    index = [f"P{i+1}" for i in range(n_samples)]

    rng = np.random.default_rng(42)

    hpo_matrix = pd.DataFrame(
        {
            "HPO1": rng.integers(0, 2, size=n_samples),
            "HPO2": rng.integers(0, 2, size=n_samples),
            "HPO3": rng.integers(0, 2, size=n_samples),
            "HPO4": rng.integers(0, 2, size=n_samples),
        },
        index=index,
    )

    relationship_mask = pd.DataFrame(
        np.zeros((4, 4), dtype=float),
        index=hpo_matrix.columns,
        columns=hpo_matrix.columns,
    )
    np.fill_diagonal(relationship_mask.values, np.nan)

    hpo_data = HpoFeatureData(
        matrix=hpo_matrix,
        label_mapping={col: col for col in hpo_matrix.columns},
        relationship_mask=relationship_mask,
    )

    disease_matrix = pd.DataFrame(
        {
            "dummy_target": rng.integers(0, 2, size=n_samples),
        },
        index=index,
    )

    targets = TargetData(
        disease_matrix=disease_matrix,
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


def test_initialize(large_dataset):
    analyzer = SynergyAnalyzer(large_dataset, n_permutations=10)

    assert analyzer.X.shape[0] == 40
    assert analyzer.n_features == 4
    assert list(analyzer.hpo_terms) == ["HPO1", "HPO2", "HPO3", "HPO4"]
    assert isinstance(analyzer.relationship_mask, np.ndarray)
    assert analyzer.n_permutations == 10
    


def test_compute_synergy_matrix(large_dataset):
    analyzer = SynergyAnalyzer(large_dataset, n_permutations=10)

    results = analyzer.compute_synergy_matrix(
        target_type="disease",
        target_name="dummy_target",
        n_jobs=1,
        include_pmids=False,
    )

    assert isinstance(results, pd.DataFrame)
    if not results.empty:
        assert "synergy" in results.columns
        assert "p_value" in results.columns
        assert "p_value_corrected" in results.columns


def test_filter_weak_synergy(large_dataset):
    analyzer = SynergyAnalyzer(large_dataset, n_permutations=10)

    analyzer.compute_synergy_matrix(
        target_type="disease",
        target_name="dummy_target",
        n_jobs=1,
        include_pmids=False,
    )

    result = analyzer.filter_weak_synergy(
        synergy_threshold=0.0,
        adj_pval_threshold=1.0,
    )

    if result is None:
        assert result is None
    else:
        filtered_synergy, filtered_p = result
        assert isinstance(filtered_synergy, pd.DataFrame)
        assert isinstance(filtered_p, pd.DataFrame)
        assert filtered_p.shape == filtered_synergy.shape


def test_plot_synergy_heatmap(large_dataset):
    analyzer = SynergyAnalyzer(large_dataset, n_permutations=10)

    analyzer.compute_synergy_matrix(
        target_type="disease",
        target_name="dummy_target",
        n_jobs=1,
        include_pmids=False,
    )

    fig = analyzer.plot_synergy_heatmap(
        synergy_threshold=0.0,
        adj_pval_threshold=1.0,
        target_name="dummy_target",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_save_synergy_heatmap(tmp_path, large_dataset):
    analyzer = SynergyAnalyzer(large_dataset, n_permutations=10)

    analyzer.compute_synergy_matrix(
        target_type="disease",
        target_name="dummy_target",
        n_jobs=1,
        include_pmids=False,
    )

    fig = analyzer.plot_synergy_heatmap(
        synergy_threshold=0.0,
        adj_pval_threshold=1.0,
        target_name="dummy_target",
    )

    output_file = tmp_path / "heatmap.html"
    analyzer.save_synergy_heatmap(fig, str(output_file))

    assert output_file.exists()