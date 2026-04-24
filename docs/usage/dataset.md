# Build dataset

After loading phenopackets, the next step is to construct an analysis-ready dataset.

The `PhenotypeDatasetBuilder` transforms raw phenopackets into a structured dataset that can be used for correlation and synergy analysis.

---

## Basic usage

```python
from ppkt2synergy import PhenotypeDatasetBuilder
from gpsea.model import VariantEffect

builder = PhenotypeDatasetBuilder(phenopackets_multi)

dataset = builder.build(
    mane_tx_id=['NM_004612.4','NM_003242.6','NM_005902.4'],
    variant_effect_type=VariantEffect.MISSENSE_VARIANT,
    missing_threshold=0.9
)
```

---

## What happens during dataset construction

The resulting dataset consists of three main components:

---

### HPO feature data

A binary feature matrix is always constructed, representing the presence or absence of HPO terms across individuals.

* **1** → HPO term observed
* **0** → explicitly excluded
* **NaN** → unknown

The feature matrix is stored in a structured object:

```python
dataset.hpo_data.matrix
```

Rows correspond to individuals, and columns correspond to HPO terms.

Additional information may include:

* mapping from HPO IDs to labels
* relationships between HPO terms (e.g., hierarchical structure)

---

### Target variables

Target variables define the outcome used in downstream analysis.

#### Disease target (always available)

A disease-based target is always constructed from the input phenopackets.

This can be used for association or stratification analyses.

---

#### Variant condition target (optional)

If both `mane_tx_id` and `variant_effect_type` are provided, the builder constructs a **binary variant-condition target**.

* **1** → individual matches the specified variant condition
* **0** → individual does not match

This target is typically used for **synergy analysis**.

Multiple transcript IDs can be provided and are combined into a single condition.

---

### Individual metadata

Metadata are constructed for each individual, including:

* cohort
* sex
* age
* associated publications (PMIDs)

These metadata can be used for filtering, grouping, or downstream analysis.

```python
dataset.individual_metadata
```

---

## Parameters

### `mane_tx_id`

Defines the MANE transcript(s) used for constructing the variant-condition target.

* single transcript:

```python
mane_tx_id = "NM_004612.4"
```

* multiple transcripts:

```python
mane_tx_id = ["NM_004612.4", "NM_003242.6"]
```

---

### `variant_effect_type`

Defines the variant effect used for classification, for example:

```python
VariantEffect.MISSENSE_VARIANT
```

---

### `missing_threshold`

Controls filtering of HPO terms based on missingness:

* values close to **1.0** → keep most features
* lower values → remove features with high missing rates

---

## Advanced: HPO configuration

By default, the HPO ontology is loaded automatically.

You can optionally specify a custom HPO file or release:

```python
builder = PhenotypeDatasetBuilder(
    phenopackets,
    hpo_file="path/to/hp.json",
    hpo_release="2023-10-09",
)
```

This is useful for reproducibility or when working with a local ontology file.

---

## Advanced: restricting the feature space to a reference cohort

In some analyses, it can be useful to evaluate a target across the full dataset while restricting the HPO feature space to terms observed in a specific reference cohort.

For example, the full dataset can be aligned to the HPO terms observed in the `FBN1` cohort:

```python
phenopackets_all = load_phenopackets_by_cohort()
dataset_all = PhenotypeDatasetBuilder(phenopackets_all).build(missing_threshold=1.0)

phenopackets_fbn1 = load_phenopackets_by_cohort(cohorts="FBN1")
dataset_fbn1 = PhenotypeDatasetBuilder(phenopackets_fbn1).build(missing_threshold=1.0)

dataset_all.hpo_data.matrix = dataset_all.hpo_data.matrix[
    dataset_fbn1.hpo_data.matrix.columns
]

dataset_all.hpo_data.relationship_mask = dataset_all.hpo_data.relationship_mask.loc[
    dataset_fbn1.hpo_data.matrix.columns,
    dataset_fbn1.hpo_data.matrix.columns
]
```

This setup defines a condition-specific feature space:

* the **population** is the full dataset
* the **target** can represent membership in a specific cohort (e.g. `FBN1` vs others)
* the **features** are restricted to HPO terms observed in the reference cohort

This can be useful when the goal is to study interactions among phenotype features that are specifically relevant to one condition, while still evaluating them against a broader background population.

Using the corresponding relationship mask ensures that the ontology-aware filtering remains consistent after restricting the feature space.

---

## Notes

* HPO feature data and disease targets are always constructed
* Variant-condition targets require both `mane_tx_id` and `variant_effect_type`
* If no individuals match the variant condition, the target may contain only zeros
* The resulting dataset can be used directly for correlation and synergy analysis

---

## Next steps

Once the dataset has been constructed, you can perform downstream analysis.

* See **Correlation analysis** to explore pairwise relationships between HPO terms
* See **Synergy analysis** to identify higher-order interactions
