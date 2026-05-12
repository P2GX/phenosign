Correlation Analysis
====================

Once a dataset has been constructed, pairwise associations between HPO terms
can be quantified using ``HPOCorrelationAnalyzer``.


What is correlation?
--------------------

Correlation measures the association between two phenotypes:

- **positive values** → phenotypes tend to co-occur  
- **negative values** → phenotypes tend to occur in different individuals  
- **values near zero** → little or no association  


Correlation methods
-------------------

The analyzer supports multiple correlation measures:

- ``SPEARMAN`` (default) — general-purpose rank-based correlation  
- ``KENDALL`` — alternative rank-based measure  
- ``PHI`` — designed for binary data  

For most use cases, the default setting provides a good starting point.


Core usage
----------

.. code-block:: python

    from ppkt2synergy import HPOCorrelationAnalyzer, CorrelationType

    analyzer = HPOCorrelationAnalyzer(
        dataset=dataset,
        min_individuals_for_correlation_test=30,
        min_cooccurrence_count=1,
    )

    results = analyzer.compute_correlation_matrix(
        correlation_type=CorrelationType.SPEARMAN,
        n_jobs=32,
        include_pmids=False,
    )

The main parameters control data filtering, correlation type, and parallelization.


Key parameters
--------------

- ``min_individuals_for_correlation_test``  
  Minimum number of individuals required to evaluate a feature pair.  
  Higher values increase robustness, while lower values allow more pairs to be tested.

- ``min_cooccurrence_count``  
  Minimum level of joint support required for a feature pair.  
  Helps exclude pairs with very limited overlap.

- ``include_pmids``  
  If enabled, associated PMIDs are included in the result table.

- ``n_jobs``  
  Controls parallel execution.


Save results
------------

.. code-block:: python

    analyzer.save_correlation_results(
        abs_threshold=0.55,
        adj_pval_threshold=1.0,
        output_file="correlation_results.csv",
    )


Visualization
-------------

.. code-block:: python

    fig = analyzer.plot_correlation_heatmap_with_significance(
        stats_name="spearman",
        abs_threshold=0.55,
        adj_pval_threshold=0.1,
    )

    fig.show()

    analyzer.save_correlation_heatmap(
        fig,
        output_file="correlation_heatmap.html",
    )


Thresholds
----------

- ``abs_threshold`` controls the minimum correlation strength  
- ``adj_pval_threshold`` controls statistical significance  

Lower thresholds include more pairs, while higher thresholds focus on stronger
and more reliable associations.

.. note::

   - Correlation requires variation in the data (both observed and excluded values)  
   - Ontologically related HPO terms may be excluded from testing  
   - Multiple-testing correction is applied automatically  


Next steps
----------

- See :doc:`synergy` for higher-order interaction analysis