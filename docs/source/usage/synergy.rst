Synergy Analysis
================

After constructing a dataset, the ``SynergyAnalyzer`` identifies pairs of HPO features
whose joint effect on a condition cannot be explained by individual features alone.


What is synergy?
----------------

Synergy measures whether a pair of HPO features provides additional information
about a condition compared to each feature individually:

- **positive synergy** → the combination is more informative than each feature alone
- **near zero** → features contribute independently
- **negative synergy** → features are redundant with respect to the condition

Synergy is computed using mutual information with permutation-based significance testing.


Inspect the dataset
-------------------

Before defining a condition, inspect the cohort to understand available diseases,
genes, sex distribution, and variant effects:

.. code-block:: python

    diseases_df, sex_df, genes_df, variant_effects_df = dataset.describe_conditions()

``variant_effects_df`` is ``None`` if no GPSEA cohort was built during dataset construction.


Core usage
----------

.. code-block:: python

    from ppkt2synergy import SynergyAnalyzer

    synergy_analyzer = SynergyAnalyzer(
        dataset=dataset,
        n_permutations=1000,
        min_individuals_for_synergy_calculation=40,
        random_state=42,
    )

Key parameters
^^^^^^^^^^^^^^

``n_permutations``
  Number of permutations used for significance testing.
  Higher values improve stability but increase runtime.

``min_individuals_for_synergy_calculation``
  Minimum number of individuals required to evaluate a feature pair.
  Higher values increase robustness; lower values allow more pairs to be tested.

``random_state``
  Seed for reproducible permutation testing.


Defining a condition
--------------------

Synergy analysis requires a **condition** — a binary vector indicating which
individuals belong to the positive group (1) and which do not (0).

Conditions are constructed by passing a **predicate function** to the dataset.
**ppkt2synergy** provides built-in helper functions to generate common predicates.

.. note::

   A condition must have both positive (1) and negative (0) samples present.
   Conditions with only a single class cannot be used for synergy analysis.

.. note::

   The built-in helpers cover common use cases, but you can define any predicate
   as a plain Python function. A phenopacket-level predicate must accept a
   ``Phenopacket`` and return ``True``, ``False``, or ``None`` (unknown):

   .. code-block:: python

      def my_predicate(phenopacket) -> bool | None:
          # your custom logic here
          return True

      condition = dataset.get_condition(my_predicate, name="my_condition")

   For variant-level predicates, the function receives a GPSEA ``Patient`` object
   instead. See the `GPSEA documentation <https://gpsea.readthedocs.io>`_ for details
   on the ``Patient`` data model.


Using built-in helpers
^^^^^^^^^^^^^^^^^^^^^^

The following helpers work at the phenopacket level and are passed to
``dataset.get_condition()``:

.. code-block:: python

    from ppkt2synergy import has_disease, has_sex, has_gene

    # By disease
    condition = dataset.get_condition(
        has_disease("OMIM:154700"),
        name="disease:Marfan syndrome",
    )

    # By sex
    condition = dataset.get_condition(
        has_sex("female"),
        name="sex:female",
    )

    # By gene
    condition = dataset.get_condition(
        has_gene("FBN1"),
        name="gene:FBN1",
    )

The ``name`` parameter is optional but recommended — it labels the condition
and enables caching for repeated queries.


Variant-based conditions
^^^^^^^^^^^^^^^^^^^^^^^^

For variant-level conditions, use ``dataset.get_variant_condition()`` with
helpers that operate on GPSEA ``Patient`` objects.

.. warning::

   Variant-based conditions require a GPSEA cohort.
   Set ``build_gpsea_cohort=True`` when constructing the dataset.

Filter by variant effect on a specific transcript:

.. code-block:: python

    from gpsea.model import VariantEffect
    from ppkt2synergy import has_variant_effect

    condition = dataset.get_variant_condition(
        has_variant_effect(
            transcript_id="NM_000138.5",
            variant_effect=VariantEffect.MISSENSE_VARIANT,
        ),
        condition_name="variant:missense_NM_000138.5",
    )

Filter by variant effect restricted to a specific exon:

.. code-block:: python

    from ppkt2synergy import has_exon_and_variant_effect

    condition = dataset.get_variant_condition(
        has_exon_and_variant_effect(
            transcript_id="NM_000138.5",
            exon=25,
            variant_effect=VariantEffect.MISSENSE_VARIANT,
        ),
        condition_name="variant:missense_exon25_NM_000138.5",
    )

.. warning::

   Synergy analysis requires variability in both conditions and HPO term pairs.
   Each condition must split individuals into at least two groups - if a
   condition has no variation, no synergy will be computed. Each HPO term
   pair must also have variability across individuals; pairs with no variation
   are automatically skipped and will not appear in the results.


Running the analysis
--------------------

Pass a condition to ``compute_synergy_matrix()``:

.. code-block:: python

    synergy_results = synergy_analyzer.compute_synergy_matrix(
        condition=condition,
        n_jobs=-1,
    )

    synergy_results.head()

Set ``n_jobs=-1`` to use all available CPU cores.


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
        condition_name="disease:Marfan syndrome",
    )

    fig.show()

    synergy_analyzer.save_synergy_heatmap(
        fig,
        output_file="synergy_heatmap.html",
    )

``synergy_threshold`` sets the minimum interaction strength;
``adj_pval_threshold`` controls statistical significance.
Lower thresholds include more pairs; higher thresholds focus on stronger
and more reliable interactions.

.. note::

   Ontologically related HPO terms may be excluded from pairwise testing.
   Multiple-testing correction is applied automatically.


Next steps
----------

Synergy analysis can be combined with correlation analysis to distinguish:

- general phenotype associations (correlation)
- condition-dependent interactions (synergy)

See :doc:`correlation` to explore phenotype–phenotype relationships independently of a condition.