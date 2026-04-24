# Correlation analysis

Once a dataset has been constructed, pairwise associations between HPO terms can be quantified using the `HPOCorrelationAnalyzer`.`

## What is correlation?

Correlation quantifies the **degree of association** between two variables.  
In the context of HPO features:

* **Positive correlation** → two phenotypes tend to appear together in the same individuals  
* **Negative correlation** → two phenotypes tend to occur in mutually exclusive individuals  
* **Values near zero** → little or no association  

The analyzer supports three correlation measures:

* `CorrelationType.SPEARMAN`  
* `CorrelationType.KENDALL`  
* `CorrelationType.PHI`

* **Spearman / Kendall** → measure how consistently the presence/absence of one feature ranks relative to another across individuals. Useful when you want a general sense of monotonic association, even with sparse binary data.  
* **Phi coefficient** → specifically designed for binary data, directly measuring co-occurrence (both 1s) and mutual exclusivity (1 vs 0) between two features.

---

## Basic usage

```python
from ppkt2synergy import HPOCorrelationAnalyzer, CorrelationType

correlation_type = CorrelationType.SPEARMAN

analyzer = HPOCorrelationAnalyzer(
    dataset=dataset_multi,
    min_individuals_for_correlation_test=30,
    min_cooccurrence_count=1,
)

results = analyzer.compute_correlation_matrix(
    correlation_type=correlation_type,
    n_jobs=32,
    include_pmids=False,
)
```

This computes pairwise correlations between HPO terms and returns a results table.

---

### Input data

Correlation analysis is performed on the HPO feature matrix:

```python
dataset.hpo_data.matrix
```

The matrix is expected to use:

* **1** → observed
* **0** → explicitly excluded
* **NaN** → unknown

Only HPO term pairs with enough valid individuals are evaluated.

---

### Key parameters

#### `min_individuals_for_correlation_test`

Minimum number of individuals with non-missing values required for a pairwise test:

```python
min_individuals_for_correlation_test=30
```

Higher values make the analysis more conservative by requiring more support before testing a pairwise association. Lower values allow more HPO term pairs to be tested, but may increase instability in small samples.

In practice, this parameter should be chosen with respect to cohort size and the expected sparsity of phenotypic annotations.

---

#### `min_cooccurrence_count`

This parameter controls the minimum number of shared `0/0` observations required before a feature pair is considered for correlation testing.

```python
min_cooccurrence_count=1
```

This helps exclude pairs with insufficient joint support.

---

#### `include_pmids`

If enabled, PMIDs associated with contributing individuals are aggregated and included in the result table.

```python
include_pmids=True
```

---

## Save results

You can save the correlation results to a CSV or Excel file.

```python
analyzer.save_correlation_results(
    abs_threshold=0.55,
    adj_pval_threshold=1.0,
    output_file="correlation_results_Loeys-Dietz syndrome.csv",
)
```

---

## Visualize correlation structure

A filtered correlation heatmap can be generated with:

```python
fig = analyzer.plot_correlation_heatmap_with_significance(
    stats_name=correlation_type.value,
    abs_threshold=0.55,
    adj_pval_threshold=0.1,
    title_name="Loeys-Dietz syndrome",
)

fig.show()
```

To save the heatmap:

```python
analyzer.save_correlation_heatmap(
    fig,
    output_file="correlation_heatmap_Loeys-Dietz syndrome.html",
)
```

---

### Thresholds for result filtering

#### `abs_threshold`

Minimum absolute correlation coefficient to retain.

* higher values keep only stronger associations
* lower values retain more pairs

#### `adj_pval_threshold`

Maximum corrected p-value to retain.

The analyzer applies multiple-testing correction using the Benjamini-Hochberg procedure and stores the corrected values in `p_value_corrected`.

A stricter threshold retains only more statistically supported pairs, while a more relaxed threshold can be useful during exploratory analysis.

---

## Notes

* Correlation analysis requires variation in the HPO feature matrix; both `1` and `0` values must be present
* Ontologically related HPO term pairs may be masked and excluded from testing
* Multiple-testing correction is applied automatically to the result table
* Threshold choices depend on the size and sparsity of the dataset and should be interpreted in context

---

## Next steps

After identifying pairwise associations, explore higher-order interactions with **Synergy analysis**.
