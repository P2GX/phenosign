# Tutorial: Correlation and Synergy Analysis

In this tutorial, we analyze a cohort of phenopackets corresponding to the gene **FBN1**. The data is retrieved from the phenopacket store (see [here](https://github.com/monarch-initiative/phenopacket-store)).

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

The cohort name (`"FBN1"`) should match an available dataset in the phenopacket store.

The function returns a list of `EnrichedPhenopacket` objects, which wrap the original GA4GH `Phenopacket` together with cohort information.

At this stage, `phenopackets` contains individual-level records with phenotypic annotations, and serves as the input for dataset construction in the next step.

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

This step transforms raw phenotypic annotations into a structured dataset that can be used for downstream analysis.

Conceptually, the dataset contains three main components:

* **HPO feature matrix**
  A binary matrix where rows correspond to individuals and columns correspond to HPO terms.
  Values indicate whether a phenotype is observed, excluded, or unknown.

* **target variables**
  Additional matrices describing outcomes of interest, such as disease labels or variant conditions.
  These are used in synergy analysis.

* **individual metadata**
  Optional metadata for each individual (e.g., cohort, sex, or publication identifiers).

Together, these components provide a unified representation of phenotypic data for statistical analysis.

In this example:

* `mane_tx_id` specifies the transcript of interest. In this example, we use the MANE Select transcript **NM_000138.5** for the *FBN1* gene.
* `variant_effect_type` specifies the variant class used to define the target variable. Here, we focus on individuals carrying **missense_variant** changes.
* `missing_threshold` controls how much missing data is allowed for each HPO feature. For example, a threshold of **0.9** allows features with up to 90% missing values to be retained.

After this step, the `dataset` object is ready for correlation and synergy analysis.


---

## 3. Correlation analysis

We now compute pairwise correlations between HPO features.

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

This step computes pairwise associations between all HPO terms across individuals in the cohort.

The parameters control basic filtering and computation settings:

* `min_individuals_for_correlation_test` ensures that correlations are only computed when a sufficient number of individuals are available
* `min_cooccurrence_count` filters out feature pairs that rarely co-occur
* `n_jobs` controls parallel execution (use higher values for faster computation on multi-core systems)

The result is a table where each row corresponds to a pair of HPO terms, along with their correlation statistics and supporting information.

A typical result includes:

* the two HPO terms being compared
* the correlation value
* statistical significance (p-value and adjusted p-value)
* counts of co-occurrence across individuals

In general:

* **positive correlation** → the two features tend to appear together
* **negative correlation** → the features tend to occur in mutually exclusive individuals
* **values near zero** → little or no association

For example, strong positive correlations may indicate phenotypic patterns that frequently co-occur, while strong negative correlations may suggest mutually exclusive features.

These results can be further filtered or visualized in the next step.

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

The thresholds control which feature pairs are displayed:

* `abs_threshold` filters out weak correlations and keeps only stronger associations
* `adj_pval_threshold` filters results based on statistical significance after multiple testing correction

Together, these parameters help focus the visualization on the most relevant phenotype–phenotype relationships.

In practice, increasing the thresholds results in a sparser heatmap with only the strongest signals, while lower thresholds reveal more comprehensive but potentially noisier patterns.

---

## 5. Inspect available targets

Before running synergy analysis, it is useful to inspect which target variables are available in the dataset.

```python
dataset.describe_available_targets()
```

This summary shows which targets can be used in downstream analysis.

In this example, the dataset provides:

* built targets such as **disease** and **variant_condition**
* metadata-derived targets such as **sex**

For the following synergy analysis, we use the built target `"variant_condition"` with target name `"missense_variant"`.

---

## 6. Synergy analysis

Correlation captures pairwise relationships, but it does not account for interactions involving multiple features.

Synergy analysis addresses this by asking:

> Do two features together provide more information about a target than each feature alone?

We first initialize the analyzer:

```python
from ppkt2synergy import SynergyAnalyzer

synergy_analyzer = SynergyAnalyzer(
    dataset=dataset,
    n_permutations=1000,
    min_individuals_for_synergy_calculation=40,
    random_state=40
)
```

In this example:

* `n_permutations` controls the number of permutation tests used to assess significance
* `min_individuals_for_synergy_calculation` ensures that synergy is only evaluated when enough individuals are available
* `random_state` makes the analysis reproducible

We then compute the synergy matrix with respect to the selected target:

```python
synergy_analyzer.compute_synergy_matrix(
    target_type="variant_condition",
    target_name="missense_variant",
    n_jobs=32,
    include_pmids=False
)
```

Here, `target_type="variant_condition"` specifies the type of target, and `target_name="missense_variant"` selects the specific target within that category.

Synergy values can be interpreted as follows:

* **positive synergy** → the two features jointly provide additional information about the target
* **near zero** → the features contribute largely independently
* **negative synergy** → the features are redundant with respect to the target

This makes synergy analysis useful for identifying phenotype combinations that may be biologically meaningful or diagnostically informative.

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

This visualization highlights feature pairs that show strong and statistically significant synergy with respect to the selected target.

The thresholds control which interactions are displayed:

* `synergy_threshold` filters out weak interactions and keeps only stronger synergistic effects
* `adj_pval_threshold` ensures that only statistically significant interactions are shown

In practice:

* higher thresholds result in a sparser heatmap focused on the strongest interactions
* lower thresholds reveal more interactions but may include weaker or noisier signals

This plot helps identify combinations of phenotypic features that jointly provide meaningful information about the target variable.


## Summary

In this tutorial, we:

- loaded phenopacket data  
- constructed a structured dataset  
- performed correlation analysis  
- identified higher-order feature interactions using synergy  

For additional usage patterns and parameter options, see the **Usage** section.

