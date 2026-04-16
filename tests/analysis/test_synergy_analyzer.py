import pytest
import numpy as np
import pandas as pd
from ppkt2synergy import SynergyAnalyzer
from ppkt2synergy.preprocessing.matrices import HpoFeatureData


# ---- Mock class ----
class MockHpoFeatureMatrix(HpoFeatureData):
    def __init__(self, hpo_matrix, relationship_mask, pmids_matrix):
        self.hpo_matrix = hpo_matrix
        self.hpo_relationship_mask = relationship_mask
        self.patient_info_df = pmids_matrix
        self.label_mapping = {}


@pytest.fixture
def large_hpo_data():
    n_samples = 40

    hpo_matrix = pd.DataFrame({
        "HPO1": np.random.randint(0, 2, size=n_samples),
        "HPO2": np.random.randint(0, 2, size=n_samples),
        "HPO3": np.random.randint(0, 2, size=n_samples),
        "HPO4": np.random.randint(0, 2, size=n_samples),
    }, index=[f"P{i+1}" for i in range(n_samples)])

    pmids_matrix = pd.DataFrame({
        "pmids": [
            list(np.random.choice(["PM1", "PM2", "PM3"],
                                  size=np.random.randint(1, 3),
                                  replace=False))
            for _ in range(n_samples)
        ]
    }, index=hpo_matrix.index)

    relationship_mask = pd.DataFrame(
        np.ones((4, 4)),  # 推荐用 ones
        index=hpo_matrix.columns,
        columns=hpo_matrix.columns
    )

    return MockHpoFeatureMatrix(hpo_matrix, relationship_mask, pmids_matrix)


@pytest.fixture
def large_target():
    n_samples = 40
    return pd.Series(
        np.random.randint(0, 2, size=n_samples),
        index=[f"P{i+1}" for i in range(n_samples)]
    )


def test_initialize(large_hpo_data, large_target):
    analyzer = SynergyAnalyzer(large_hpo_data, large_target, n_permutations=10)
    assert analyzer.X.shape[0] == 40
    assert analyzer.n_features == 4
    assert not np.isnan(analyzer.synergy_matrix).all()


def test_compute_synergy_matrix(large_hpo_data, large_target):
    analyzer = SynergyAnalyzer(large_hpo_data, large_target, n_permutations=10)
    results = analyzer.compute_synergy_matrix(n_jobs=1)

    assert isinstance(results, pd.DataFrame)
    if not results.empty:
        assert "synergy" in results.columns
        assert "p_value" in results.columns


def test_filter_weak_synergy(large_hpo_data, large_target):
    analyzer = SynergyAnalyzer(large_hpo_data, large_target, n_permutations=10)
    analyzer.compute_synergy_matrix(n_jobs=1)

    filtered_synergy, filtered_p = analyzer.filter_weak_synergy(
        synergy_threshold=0.01,
        alpha=1.0,
        corrected_alpha=1.0
    )

    assert filtered_synergy.shape[0] <= analyzer.synergy_matrix.shape[0]
    assert filtered_p.shape == filtered_synergy.shape


def test_plot_synergy_heatmap(large_hpo_data, large_target):
    analyzer = SynergyAnalyzer(large_hpo_data, large_target, n_permutations=10)
    analyzer.compute_synergy_matrix(n_jobs=1)

    fig = analyzer.plot_synergy_heatmap(
        synergy_threshold=0.01,
        alpha=1.0,
        corrected_alpha=1.0,
        target_name="Test Target"
    )

    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_save_synergy_heatmap(tmp_path, large_hpo_data, large_target):
    analyzer = SynergyAnalyzer(large_hpo_data, large_target, n_permutations=10)
    analyzer.compute_synergy_matrix(n_jobs=1)

    fig = analyzer.plot_synergy_heatmap(
        synergy_threshold=0.01,
        alpha=1.0,
        corrected_alpha=1.0
    )

    output_file = tmp_path / "heatmap.html"
    analyzer.save_synergy_heatmap(fig, str(output_file))

    assert output_file.exists()