Build Dataset
=============

After loading phenopackets, the next step is to construct an analysis-ready dataset.

The ``PhenotypeDatasetBuilder`` converts raw phenopackets into a structured representation
that can be used for correlation and synergy analysis.


Core usage
----------

.. code-block:: python

    from ppkt2synergy import PhenotypeDatasetBuilder
    from gpsea.model import VariantEffect

    builder = PhenotypeDatasetBuilder(phenopackets)

    dataset = builder.build(
        mane_tx_id=["NM_004612.4", "NM_003242.6"],
        variant_effect_type=VariantEffect.MISSENSE_VARIANT,
        missing_threshold=0.9,
    )


Dataset overview
----------------

The resulting dataset contains:

- **phenotype features** (HPO-based representation of individuals)
- **target variables** (e.g., disease labels or variant conditions)
- **individual metadata**

These components are used internally by correlation and synergy analysis.

.. note::

   HPO features and disease targets are always constructed.

   Variant-condition targets require both ``mane_tx_id`` and ``variant_effect_type``.

   If no individuals match the variant condition, the target may contain only zeros.


Key parameters
--------------

- ``mane_tx_id``  
  Defines the MANE transcript(s) used for constructing the variant-condition target.

- ``variant_effect_type``  
  Specifies the variant effect class (e.g., missense variants).

- ``missing_threshold``  
  Controls filtering of HPO terms based on missingness:

  - values close to **1.0** → keep all features
  - lower values → remove features with high missing rates


Advanced usage
--------------

Customizing HPO configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    builder = PhenotypeDatasetBuilder(
        phenopackets,
        hpo_file="path/to/hp.json",
        hpo_release="2023-10-09",
    )


Restricting the feature space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

In some cases, you may want to evaluate a target on the full dataset
while restricting features to those observed in a reference cohort.

This setup allows:

- the **population** to be defined by the full dataset
- the **target** to represent a specific cohort
- the **features** to be restricted to a reference phenotype space


Next steps
----------

After constructing the dataset:

- See :doc:`correlation` to explore phenotype–phenotype relationships
- See :doc:`synergy` to identify condition-dependent interactions