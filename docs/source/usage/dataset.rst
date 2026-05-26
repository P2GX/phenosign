Build Dataset
=============

After loading phenopackets, the next step is to construct an analysis-ready dataset.

The ``PhenotypeDatasetBuilder`` converts raw phenopackets into a structured representation
that can be used for correlation and synergy analysis.


Core usage
----------

.. code-block:: python

    from ppkt2synergy import PhenotypeDatasetBuilder

    builder = PhenotypeDatasetBuilder(phenopackets)

    dataset = builder.build(
        missing_threshold=0.9,
        build_gpsea_cohort=True,
    )


Dataset overview
^^^^^^^^^^^^^^^^

The resulting ``PhenotypeDataset`` contains three components:

- **hpo_data** — binary HPO feature matrix across individuals, with optional
  term relationship mask
- **phenopackets** — the original phenopacket objects, retained for reference
  and downstream computations
- **gpsea_cohort** — a preprocessed GPSEA cohort object for variant-aware analyses
  (present only when ``build_gpsea_cohort=True``)

Two index properties provide convenient access to the matrix dimensions:

.. code-block:: python

    dataset.individual_ids   # pd.Index of subject identifiers (rows)
    dataset.feature_ids      # pd.Index of HPO term identifiers (columns)


Key parameters
^^^^^^^^^^^^^^

``missing_threshold``
  Controls which HPO terms are retained based on how often they are observed
  across individuals.

  ``missing_threshold=0.9`` keeps only HPO terms observed (or explicitly excluded)
  in at least 10% of individuals. Set to ``1.0`` to retain all terms regardless
  of missingness.

``build_gpsea_cohort``
  If ``True``, constructs a GPSEA-compatible cohort object and attaches it to
  ``dataset.gpsea_cohort``. Required for variant-based condition analysis.
  Defaults to ``True``.


Advanced usage
--------------

Customizing HPO configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    builder = PhenotypeDatasetBuilder(
        phenopackets,
        hpo_file="path/to/hp.json",
        hpo_release="2023-10-09",
    )

Next steps
----------

After constructing the dataset:

- See :doc:`correlation` to explore phenotype–phenotype relationships
- See :doc:`synergy` to identify condition-dependent interactions