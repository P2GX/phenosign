# ppkt2synergy

**ppkt2synergy** is a Python library for investigating correlation and synergy in cohorts of GA4GH Phenopackets.

---

## Overview

This package provides tools to analyze relationships between phenotypic features encoded using the Human Phenotype Ontology (HPO).

It supports:

* pairwise correlation analysis between HPO terms
* detection of higher-order interactions (synergy) with respect to a target variable (e.g., variant effects or disease)
* construction of structured datasets from phenopacket cohorts

---

## Quick Example

```python
from ppkt2synergy import load_phenopackets_by_cohort,PhenotypeDatasetBuilder, HPOCorrelationAnalyzer, CorrelationType,

phenopackets = load_phenopackets_by_cohort("FBN1")
dataset = PhenotypeDatasetBuilder(phenopackets).build()

analyzer = HPOCorrelationAnalyzer(dataset)
analyzer.compute_correlation_matrix(
    correlation_type=CorrelationType.SPEARMAN
)
```

---

## Getting Started

* See **Setup** for installation instructions
* See **Tutorial** for a complete analysis workflow

---

## Documentation Structure

* **Setup** – installation instructions
* **Tutorial** – step-by-step workflow
* **API** – automatically generated reference documentation
