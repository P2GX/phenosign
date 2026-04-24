# Build dataset

After loading phenopackets, the next step is to construct an **analysis-ready dataset**.  

The `PhenotypeDatasetBuilder` transforms raw phenopackets into a structured dataset suitable for **correlation** and **synergy** analysis.

---

## Basic usage

```python
from ppkt2synergy import PhenotypeDatasetBuilder
from gpsea.model import VariantEffect

builder = PhenotypeDatasetBuilder(phenopackets_multi)

dataset = builder.build(
    mane_tx_id=['NM_004612.4', 'NM_003242.6', 'NM_005902.4'],
    variant_effect_type=VariantEffect.MISSENSE_VARIANT,
    missing_threshold=0.9
)
```

This constructs a dataset containing HPO feature matrices, target variables, and individual metadata.

---

### Dataset components

The resulting dataset has three main components:

#### HPO feature data

A binary feature matrix is always constructed, representing the presence or absence of HPO terms across individuals.

* **1** → HPO term observed
* **0** → explicitly excluded
* **NaN** → unknown

Stored as:

```python
dataset.hpo_data.matrix
```

Rows correspond to individuals, and columns correspond to HPO terms. Additional information may include:

* mapping from HPO IDs to labels
* relationships between HPO terms (e.g., hierarchical structure)

---

#### Target variables

Target variables define the outcomes for downstream analysis.

##### Disease target (always available)

A disease-based target is automatically constructed from input phenopackets and can be used for association or stratification analysis.

Accessible via:
```python
dataset.target_data.disease
```

##### Variant condition target (optional)

If both `mane_tx_id` and `variant_effect_type` are provided, the builder constructs a **binary variant-condition target**.

* **1** → individual matches the specified variant condition
* **0** → individual does not match

Multiple transcript IDs can be combined into a single target. This target is typically used in **synergy analysis**.

---

#### Individual metadata

Metadata are constructed for each individual, including:

* cohort
* sex
* age
* associated publications (PMIDs)

Accessible via:

```python
dataset.individual_metadata
```

---

### Key Parameters

#### `mane_tx_id`

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

#### `variant_effect_type`

Defines the variant effect type to consider:

```python
VariantEffect.MISSENSE_VARIANT
```

---

#### `missing_threshold`

Controls filtering of HPO terms based on missingness:

* values close to **1.0** → keep all features
* lower values → remove features with high missing rates

---

## Advanced: Custom HPO configuration

By default, the HPO ontology is loaded automatically.

You can specify a custom HPO ontology file or release for reproducibility:

```python
builder = PhenotypeDatasetBuilder(
    phenopackets,
    hpo_file="path/to/hp.json",
    hpo_release="2023-10-09",
)
```

---

## Advanced: restricting the feature space to a reference cohort

Sometimes it is useful to evaluate a target on the full dataset while restricting features to those observed in a reference cohort:

```python
# Load full dataset
phenopackets_all = load_phenopackets_by_cohort()
dataset_all = PhenotypeDatasetBuilder(phenopackets_all).build(missing_threshold=1.0)

# Load reference cohort
phenopackets_fbn1 = load_phenopackets_by_cohort(cohorts="FBN1")
dataset_fbn1 = PhenotypeDatasetBuilder(phenopackets_fbn1).build(missing_threshold=1.0)

# Restrict features to those in reference cohort
dataset_all.hpo_data.matrix = dataset_all.hpo_data.matrix[
    dataset_fbn1.hpo_data.matrix.columns
]

dataset_all.hpo_data.relationship_mask = dataset_all.hpo_data.relationship_mask.loc[
    dataset_fbn1.hpo_data.matrix.columns,
    dataset_fbn1.hpo_data.matrix.columns
]
```

This ensures:

* the **population** -> full dataset
* the **target** -> membership in a specific cohort (e.g. `FBN1` vs others)
* the **features** -> restricted to HPO terms observed in the reference cohort

Ontology-aware filtering remains consistent by using the corresponding relationship mask.

---

## Notes

* HPO feature data and disease targets are always constructed
* Variant-condition targets require both `mane_tx_id` and `variant_effect_type`
* If no individuals match the variant condition, the target may contain only zeros
* The resulting dataset can be used directly for correlation and synergy analysis

---

## Next steps

After constructing the dataset, proceed to:

* See **Correlation analysis** -> explore pairwise relationships between HPO terms
* See **Synergy analysis** -> identify higher-order interactions
