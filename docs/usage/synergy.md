# Synergy analysis

After constructing a dataset, the `SynergyAnalyzer` identifies pairs of HPO features that jointly provide more information about a target variable than individual features alone.

## What is synergy?

Synergy quantifies whether a **pair of HPO features provides more information about a target** than expected from their individual contributions alone.

* **Positive synergy** → the combination of two features is more informative than each one separately  
* **Zero synergy** → the features provide independent information  
* **Negative synergy** → the features are redundant with respect to the target  

In ppkt2synergy, synergy is computed using **mutual information** to capture dependencies between features and the target, and **permutation testing** to assess statistical significance.  

This allows identification of **feature interactions** that are specifically relevant to a disease, variant, or cohort, beyond simple pairwise co-occurrence.

---

## Check available targets

Before running synergy analysis, inspect available targets:

```python
dataset_multi.describe_available_targets()
```

This returns **built targets** (e.g., `disease`, `variant_condition`) and **metadata-derived targets** (e.g., `cohort`, `sex`), along with their available classes.

This summary helps determine which targets can be used in synergy analysis.

---

## Initialize the analyzer

```python
from ppkt2synergy import SynergyAnalyzer

synergy_analyzer = SynergyAnalyzer(
    dataset=dataset_multi,
    n_permutations=1000,
    min_individuals_for_synergy_calculation=40,
    random_state=40,
)
```

### Key parameters

#### `n_permutations`

Controls the number of permutations used for p-value estimation.

```python
n_permutations=1000
```

Higher values give more stable p-value estimates, but increase runtime.


#### `min_individuals_for_synergy_calculation`

Minimum number of valid individuals required for evaluating a pair of HPO terms.

```python
min_individuals_for_synergy_calculation=40
```

Higher values make the analysis more conservative by requiring more support for each pair. Lower values allow more HPO term pairs to be tested, but may lead to less stable estimates in sparse datasets.

This parameter should be chosen in relation to cohort size and the amount of missing data.

---

## Target selection

Synergy analysis requires a **binary target**. There are two ways to specify the target:


### 1. Built targets

Built targets are stored directly in the dataset.

These include:

* `disease`
* `variant_condition`

#### Example: disease target

```python
synergy_results = synergy_analyzer.compute_synergy_matrix(
    target_type="disease",
    target_name="Loeys-Dietz syndrome 2",
    n_jobs=32,
)
```

#### Example: variant-condition target

```python
synergy_results = synergy_analyzer.compute_synergy_matrix(
    target_type="variant_condition",
    target_name="missense_variant",
    n_jobs=32,
)
```

Use this mode when the target already exists as a pre-built column in the dataset.


### 2. Metadata-derived targets

Metadata targets are derived from sample-level annotations.

These include, for example:

* `cohort`
* `sex`

In this mode, a target is defined by choosing one class as the positive class.

#### Example: cohort target

```python
synergy_results = synergy_analyzer.compute_synergy_matrix(
    target_type="cohort",
    positive_class="SMAD2",
    n_jobs=32,
)
```

This creates a binary target where:

* **1** → sample belongs to the selected cohort
* **0** → sample belongs to another cohort

---

### Important requirement: target variation

Synergy analysis requires both positive and negative target values.

In other words, the selected target must contain both:

* **1**
* **0**

If all samples belong to the same target class, synergy analysis cannot be performed.

Examples:

* If all phenopackets come from a single cohort, using `target_type="cohort"` is not informative
* If all loaded samples correspond to a single disease, using that disease as a target is not possible
* If a variant-condition target contains only positives or only negatives, it cannot be used for synergy analysis

---

## Save results

You can save the synergy results to a CSV or Excel file.

```python
synergy_analyzer.save_synergy_results(
    synergy_threshold=0.08,
    adj_pval_threshold=0.2,
    output_file="synergy_results_Loeys-Dietz syndrome 2.csv",
)
```

---

## Visualize synergy structure

A filtered heatmap can be generated with:

```python
fig = synergy_analyzer.plot_synergy_heatmap(
    synergy_threshold=0.08,
    adj_pval_threshold=0.2,
    target_name="Loeys-Dietz syndrome 2",
)

fig.show()
```

To save the heatmap:

```python
synergy_analyzer.save_synergy_heatmap(
    fig,
    output_file="synergy_heatmap_Loeys-Dietz syndrome 2.html",
)
```

---

### Thresholds for filtering

#### `synergy_threshold`

Minimum synergy value to retain.

* higher values keep only stronger interactions
* lower values retain more HPO term pairs

#### `adj_pval_threshold`

Maximum corrected p-value to retain.

Corrected p-values are computed automatically using the Benjamini-Hochberg procedure.

A stricter threshold retains only more strongly supported interactions, while a more relaxed threshold may be useful for exploratory analysis.

---

## Notes

* Synergy analysis requires a binary target with both positive and negative samples
* Built targets are generally the most direct option when available
* Metadata targets are useful for comparisons such as one cohort vs all others
* Ontologically related HPO term pairs may be masked and excluded from testing
* Multiple-testing correction is applied automatically

---

## Next steps

After identifying informative HPO feature pairs, you can compare synergy patterns across different targets or datasets.
You may also combine synergy analysis with correlation analysis to distinguish pairwise association from target-dependent interaction.
