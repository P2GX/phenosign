import pytest
import pandas as pd
import numpy as np
from ppkt2synergy.analysis.hpo_correlation_analyzer import HPOStatisticsAnalyzer
from ppkt2synergy.analysis.correlation_type import CorrelationType

import pytest
import pandas as pd
import numpy as np
from ppkt2synergy.analysis.hpo_correlation_analyzer import HPOStatisticsAnalyzer
from ppkt2synergy.analysis.correlation_type import CorrelationType

@pytest.fixture
def large_hpo_data():
    n_samples = 60
    hpo_matrix = pd.DataFrame({
    "HP:0001": [1]*30 + [0]*30,                
    "HP:0002": [0]*10 + [1]*30 + [0]*20,     
    "HP:0003": [0]*20 + [1]*30 + [0]*10,       
    "HP:0004": [0]*15 + [1]*30 + [0]*15,  
    }, index=[f"Patient_{i}" for i in range(n_samples)])

    pmids_matrix = pd.DataFrame({"pmids":[["PMID:1"]] * n_samples}, index=hpo_matrix.index)
    #relationship_mask = pd.DataFrame(np.nan, index=hpo_matrix.columns, columns=hpo_matrix.columns)
    return (hpo_matrix, None, pmids_matrix)




def test_initialization(large_hpo_data):
    analyzer = HPOStatisticsAnalyzer(large_hpo_data, min_individuals_for_correlation_test=10)
    assert analyzer.hpo_matrix.shape[0] >= 10
    assert analyzer.n_features == 4
    assert list(analyzer.hpo_terms) == ["HP:0001", "HP:0002", "HP:0003", "HP:0004"]

def test_compute_correlation_matrix(large_hpo_data):
    analyzer = HPOStatisticsAnalyzer(large_hpo_data, min_individuals_for_correlation_test=10)
    results = analyzer.compute_correlation_matrix(correlation_type=CorrelationType.SPEARMAN, n_jobs=1)
    print(results)
    assert isinstance(results, pd.DataFrame)
    assert "HPO_A" in results.columns
    assert "HPO_B" in results.columns
    assert "correlation" in results.columns
    assert "p_value" in results.columns
    assert "p_value_corrected" in results.columns

def test_filter_weak_correlations(large_hpo_data):
    analyzer = HPOStatisticsAnalyzer(large_hpo_data, min_individuals_for_correlation_test=10)
    analyzer.compute_correlation_matrix(correlation_type=CorrelationType.SPEARMAN, n_jobs=1)
    
    coef_cleaned, pval_cleaned = analyzer.filter_weak_correlations(lower_bound=-1, upper_bound=1, alpha=1.0)
    
    assert isinstance(coef_cleaned, pd.DataFrame)
    assert isinstance(pval_cleaned, pd.DataFrame)

def test_plot_heatmap(large_hpo_data):
    import plotly.graph_objects as go
    
    analyzer = HPOStatisticsAnalyzer(large_hpo_data, min_individuals_for_correlation_test=30)
    analyzer.compute_correlation_matrix(correlation_type=CorrelationType.SPEARMAN, n_jobs=1)
    
    fig = analyzer.plot_correlation_heatmap_with_significance(lower_bound=-0.1, upper_bound=0.1, alpha=0.05)
    
    assert fig is not None
    assert isinstance(fig, go.Figure)
