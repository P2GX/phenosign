Synergy Analysis
================

After constructing a dataset, the ``SynergyAnalyzer`` identifies pairs of HPO features
whose joint effect on a target variable cannot be explained by individual features alone.


What is synergy?
----------------

Synergy measures whether a pair of HPO features provides additional information
about a target compared to individual features.

- **positive synergy** → the combination is more informative than each feature alone  
- **near zero** → features contribute independently  
- **negative synergy** → features are redundant with respect to the target  

Synergy is computed using mutual information with permutation-based significance testing.


Targets
-------

Synergy analysis evaluates phenotype relationships **with respect to a target variable**.

You can inspect available targets with:

.. code-block:: python

    dataset.describe_available_targets()

Targets fall into two categories:

- **built targets**: precomputed targets stored in the dataset, such as ``disease`` or ``variant_condition``
- **metadata-derived targets**: constructed from sample-level annotations, such as ``cohort`` or ``sex``


Built targets
~~~~~~~~~~~~~

Built targets are directly available in the dataset.

.. code-block:: python

    from ppkt2synergy import SynergyAnalyzer

    synergy_analyzer = SynergyAnalyzer(
        dataset=dataset,
        n_permutations=1000,
        min_individuals_for_synergy_calculation=40,
        random_state=40,
    )

    synergy_results = synergy_analyzer.compute_synergy_matrix(
        target_type="disease",
        target_name="Loeys-Dietz syndrome 2",
        n_jobs=32,
    )

    synergy_results.head()


Metadata-derived targets
~~~~~~~~~~~~~~~~~~~~~~~~

Metadata-derived targets are created from sample-level annotations.

A binary target is defined by selecting one class as the positive group:

- **1** → selected class  
- **0** → all other samples  

.. code-block:: python

    synergy_results = synergy_analyzer.compute_synergy_matrix(
        target_type="cohort",
        positive_class="SMAD2",
        n_jobs=32,
    )

    synergy_results.head()

Target requirements
-------------------

.. warning::

   Synergy analysis requires variation in the target:

   - both positive (**1**) and negative (**0**) samples must be present  
   - targets with only a single class cannot be used  

   For example:

   - a single-cohort dataset cannot be used with ``target_type="cohort"``
   - a disease target cannot be used if all samples share the same disease


Key parameters
--------------

- ``n_permutations``  
  Controls the number of permutations used for statistical testing.  
  Higher values improve stability but increase runtime.

- ``min_individuals_for_synergy_calculation``  
  Minimum number of individuals required to evaluate a feature pair.  
  Higher values increase robustness, while lower values allow more pairs to be tested.


Save results
------------

.. code-block:: python

    synergy_analyzer.save_synergy_results(
        synergy_threshold=0.08,
        adj_pval_threshold=0.2,
        output_file="synergy_results.csv",
    )


Visualization
-------------

.. code-block:: python

    fig = synergy_analyzer.plot_synergy_heatmap(
        synergy_threshold=0.08,
        adj_pval_threshold=0.2,
        target_name="Loeys-Dietz syndrome 2",
    )

    fig.show()

    synergy_analyzer.save_synergy_heatmap(
        fig,
        output_file="synergy_heatmap.html",
    )


Thresholds
----------

- ``synergy_threshold`` controls interaction strength  
- ``adj_pval_threshold`` controls statistical significance  

Lower thresholds include more pairs, while higher thresholds focus on stronger
and more reliable interactions.

.. note::

   - Synergy requires a binary target with both positive and negative samples  
   - Built targets are typically the most direct option  
   - Metadata targets enable comparisons such as one cohort vs others  
   - Ontologically related HPO terms may be excluded  


Next steps
----------

Synergy analysis can be combined with correlation analysis to distinguish:

- general phenotype associations (correlation)
- target-dependent interactions (synergy)

You can also compare synergy patterns across different targets or datasets.

See :doc:`correlation`