# Tutorial: Correlation and Synergy Analysis

In this tutorial, we analyze a cohort of phenopackets corresponding to the gene **FBN1**. The data is retrieved from the [phenopacket store](https://github.com/monarch-initiative/phenopacket-store).

Using this dataset, we demonstrate a complete workflow with **ppkt2synergy**, including dataset construction, correlation analysis, and synergy analysis of phenotypic features.

We recommend running this tutorial in a **Jupyter notebook** for interactive analysis and visualization, although it can also be executed as a standard Python script.

---

## 1. Load phenopackets

We start by loading phenopackets for the **FBN1** cohort from the [phenopacket store](https://github.com/monarch-initiative/phenopacket-store).

```python
from ppkt2synergy import load_phenopackets_by_cohort

phenopackets = load_phenopackets_by_cohort("FBN1")

print(f"Loaded {len(phenopackets)} phenopackets")
```

You can specify any cohort or cohorts of interest available in the phenopacket store (e.g., "FBN1" in this example).

---

## 2. Build the dataset

Next, we convert phenopackets into a structured dataset suitable for statistical analysis.

```python
from ppkt2synergy import PhenotypeDatasetBuilder
from gpsea.model import VariantEffect

dataset = PhenotypeDatasetBuilder(phenopackets).build(
    mane_tx_id="NM_000138.5",
    variant_effect_type=VariantEffect.MISSENSE_VARIANT,
    missing_threshold=0.9
)
```

This step transforms raw phenotypic annotations into a structured dataset for downstream analysis.The dataset consists of:

* **HPO feature matrix**
  A binary matrix where rows correspond to individuals and columns correspond to HPO terms.
  Values indicate whether a phenotype is observed, excluded, or unknown.

* **Target variables**
  Additional matrices, such as disease labels or variant conditions, used for synergy analysis.

* **Individual metadata**
  Optional information about individual (e.g., cohort, sex, or publication identifiers).

**Key Parameters:**

* `mane_tx_id`: Specifies the transcript of interest. Here, we use the MANE Select transcript **NM_000138.5** for the *FBN1* gene.
* `variant_effect_type`: Defines the variant class to be considered. In this case, we're focusing on individuals with **missense_variant** mutation.
* `missing_threshold` controls how much missing data is allowed for each HPO feature. For example, a threshold of **0.6** means that features with up to 60% missing values to be retained.

After this step, the `dataset` object is ready for correlation and synergy analysis.

---

## 3. Correlation analysis

Next, we compute pairwise correlations between the HPO features.

```python
from ppkt2synergy import HPOCorrelationAnalyzer, CorrelationType

analyzer = HPOCorrelationAnalyzer(
    dataset=dataset,
    min_individuals_for_correlation_test=30,
    min_cooccurrence_count=1
)

analyzer.compute_correlation_matrix(
    correlation_type=CorrelationType.SPEARMAN,
    n_jobs=32,
)
```

This step calculates the pairwise correlations between all HPO terms across individuals in the cohort. The result is a matrix of correlation values, along with statistical information like p-values and adjusted p-values.

* **positive correlation** → the two features tend to appear together
* **negative correlation** → the features tend to occur in mutually exclusive individuals
* **values near zero** → little or no association

**Key Parameters:**
* `min_individuals_for_correlation_test`: Ensures that correlations are only computed when there are enough individuals.
* `min_cooccurrence_count`: Filters out feature pairs that rarely co-occur.
* `n_jobs`: Controls parallelization (higher values are better for multi-core systems).

The correlation results will help identify phenotype–phenotype relationships, such as co-occurring phenotypes or mutually exclusive features.

---

## 4. Visualize correlation results

We can visualize the correlation results as a heatmap.

```python
fig = analyzer.plot_correlation_heatmap_with_significance(
    stats_name=correlation_type.value,
    abs_threshold=0.6,
    adj_pval_threshold=0.1,
    title_name="Cohort FBN1",
)
fig.show()
```

This visualization highlights the strongest and most statistically significant associations between phenotypic features.

**Key Parameters:**

* `abs_threshold`: Filters out weak correlations (only keeping stronger ones).
* `adj_pval_threshold`: Filters results based on statistical significance after multiple testing correction.

By adjusting these thresholds, you can control the sparsity of the heatmap and focus on the most relevant features.

---

## 5. Inspect available targets

Before running synergy analysis, it's helpful to inspect the available target variables in the dataset.

```python
dataset.describe_available_targets()
```

This summary shows which targets (such as disease labels or variant conditions) can be used in subsequent analysis.

---

## 6. Synergy analysis

While correlation identifies pairwise relationships, synergy analysis looks for higher-order interactions between features.

We begin by initializing the synergy analyzer:

```python
from ppkt2synergy import SynergyAnalyzer

synergy_analyzer = SynergyAnalyzer(
    dataset=dataset,
    n_permutations=1000,
    min_individuals_for_synergy_calculation=40,
    random_state=40
)
```

**Key Parameters:**

* `n_permutations`: Controls the number of permutations used for statistical significance.
* `min_individuals_for_synergy_calculation`: Ensures enough individuals are available for meaningful synergy analysis.
* `random_state`: Ensures reproducibility.


Next, we compute the synergy matrix for the selected target:

```python
synergy_analyzer.compute_synergy_matrix(
    target_type="variant_condition",
    target_name="missense_variant",
    n_jobs=32,
    include_pmids=False
)
```

Synergy values can be interpreted as follows:

* **positive synergy** → the two features jointly provide additional information about the target
* **near zero** → the features contribute largely independently
* **negative synergy** → the features are redundant with respect to the target

---

## 7. Visualize synergy results

We can visualize synergy results as a heatmap.

```python
fig = synergy_analyzer.plot_synergy_heatmap(
    synergy_threshold=0.08,
    adj_pval_threshold=0.1,
    target_name="missense_variant",
)
fig.show()
```

This visualization highlights feature pairs that show positive and statistically significant synergy with respect to the selected target.

**Key Parameters:**

* `synergy_threshold` filters out weak interactions and keeps only stronger synergistic effects
* `adj_pval_threshold` ensures that only statistically significant interactions are shown

---

## Summary

In this tutorial, we:

- Loaded phenopacket data  
- Constructed a structured dataset  
- Performed correlation analysis  
- Identified higher-order feature interactions using synergy  

For additional usage patterns and parameter options, see the **Usage** section.
