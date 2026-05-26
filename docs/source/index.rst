ppkt2synergy
============

**ppkt2synergy** is a Python library for analyzing correlations and synergy in
`GA4GH Phenopacket <https://www.ga4gh.org/product/phenopackets/>`_ cohorts.

It provides tools to systematically process
`Human Phenotype Ontology <https://hpo.jax.org/>`_ (HPO) features,
identify pairwise associations, and detect higher-order interactions
relevant to diseases or variant conditions.



Overview
--------

Phenotypic features often co-occur or interact in complex ways across individuals.
Understanding these relationships can provide insights into disease mechanisms,
genotype-phenotype associations, and combinatorial biomarkers.

**ppkt2synergy enables:**

- Construction of structured datasets from phenopacket cohorts
- Pairwise correlation analysis of HPO terms
- Detection of higher-order feature interactions (synergy)
- Interactive visualization of correlation and synergy heatmaps



Quick Example
-------------

.. code-block:: python

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

This minimal example demonstrates the workflow:

**load phenopackets → build dataset → compute correlations**



.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage/loading
   usage/dataset
   usage/correlation
   usage/synergy

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/ppkt2synergy