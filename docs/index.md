# ppkt2synergy

**ppkt2synergy** is a Python library for analyzing correlations and synergy in [GA4GH Phenopacket](https://www.ga4gh.org/product/phenopackets/) cohorts. It provides tools to systematically process [Human Phenotype Ontology](https://hpo.jax.org/) (HPO) features, identify pairwise associations, and detect higher-order interactions relevant to diseases or variant conditions.

---

## Overview

Phenotypic features often co-occur or interact in complex ways across individuals. Understanding these relationships can provide insights into disease mechanisms, genotype-phenotype associations, and combinatorial biomarkers.

**ppkt2synergy** enables:

* Construction of structured datasets from phenopacket cohorts
* Pairwise correlation analysis of HPO terms
* Detection of higher-order feature interactions (synergy) with respect to disease or variant conditions
* Interactive visualization of correlation and synergy heatmaps

---

## Quick Example

```python
from ppkt2synergy import (
    load_phenopackets_by_cohort,
    PhenotypeDatasetBuilder,
    HPOCorrelationAnalyzer,
    CorrelationType
)

# Load phenopackets from a cohort
phenopackets = load_phenopackets_by_cohort("FBN1")

# Build dataset
dataset = PhenotypeDatasetBuilder(phenopackets).build()

# Compute pairwise correlation
analyzer = HPOCorrelationAnalyzer(dataset)
analyzer.compute_correlation_matrix(correlation_type=CorrelationType.SPEARMAN)
```
This minimal example demonstrates the main workflow: **load phenopackets** → **build dataset** → **compute correlations**.

---

## Getting Started

* See **Installation** for installation instructions for ppkt2synergy.
* See **Tutorial** for a complete analysis workflow for correlation and synergy analysis.
* See **Usage** for detailed examples covering dataset construction, correlation computation, and synergy analysis.


