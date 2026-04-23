# ppkt2synergy

**ppkt2synergy** is a Python package for correlation and synergy analysis of Human Phenotype Ontology (HPO) annotations in GA4GH phenopacket cohorts.

---

## Installation

```bash
pip install ppkt2synergy
```

---

## Example

```python
from ppkt2synergy import (
    load_phenopackets_by_cohort,
    PhenotypeDatasetBuilder,
    HPOCorrelationAnalyzer,
    CorrelationType,
)
from gpsea.model import VariantEffect

# Load phenopackets
phenopackets = load_phenopackets_by_cohort("FBN1")

# Build dataset
dataset = PhenotypeDatasetBuilder(phenopackets).build(
    mane_tx_id="NM_000138.5",
    variant_effect_type=VariantEffect.MISSENSE_VARIANT,
)

# Run correlation analysis
analyzer = HPOCorrelationAnalyzer(dataset)
analyzer.compute_correlation_matrix(
    correlation_type=CorrelationType.SPEARMAN
)
```

---

## Overview

This package enables the identification of pairwise associations and higher-order interactions between phenotypic features, helping to uncover biologically meaningful patterns in rare disease data.

---

## Features

* Correlation analysis of HPO features (Spearman, Kendall, Phi)
* Synergy analysis to detect non-additive interactions between phenotypic features with respect to a target variable (e.g., variant effects or disease)
* Support for GA4GH phenopacket data
* Structured dataset construction from phenotypic profiles
* Visualization utilities (e.g., correlation heatmaps)

---

## Quickstart

```python
from ppkt2synergy import SynergyAnalyzer

synergy = SynergyAnalyzer(dataset)
synergy.compute_synergy_matrix(
    target_type="variant_condition",
    target_name="missense_variant"
)
```

For a complete workflow and advanced options, see the documentation.

---


