Loading Phenopackets
====================

Loading phenopacket data is the first step in a **ppkt2synergy** workflow.

**ppkt2synergy** is designed to work with phenopacket data from any source.
For convenience, it provides helper functions to retrieve example datasets from the
`Phenopacket Store <https://github.com/monarch-initiative/phenopacket-store>`_.

.. note::

   The loading functions shown below are optional utilities for quick exploration.
   Any data following the
   `GA4GH Phenopacket schema <https://phenopacket-schema.readthedocs.io/en/latest/index.html>`_
   can be used directly with **ppkt2synergy**.


Using your own phenopacket data
-------------------------------

In most real-world use cases, you will provide your own phenopacket data.

.. code-block:: python

    # Example: load your own phenopackets
    phenopackets = [...]  # list of Phenopacket objects

As long as your data follows the Phenopacket schema, it can be passed directly into downstream steps such as dataset construction and analysis.


Load from the Phenopacket Store
-------------------------------

The following utilities allow you to retrieve datasets from the Phenopacket Store for testing or exploratory analysis.

.. note::

   The specified cohort or disease identifiers must exist in the Phenopacket Store.
   Refer to the
   `repository <https://github.com/monarch-initiative/phenopacket-store>`_
   to browse available datasets.


Load by cohort
~~~~~~~~~~~~~~

Load phenopackets associated with a single cohort (e.g., a gene-based cohort).

.. code-block:: python

    from ppkt2synergy import load_phenopackets_by_cohort

    phenopackets = load_phenopackets_by_cohort("TGFBR1")
    print(f"Loaded {len(phenopackets)} phenopackets")

Combine phenopackets from multiple cohorts into a single list:

.. code-block:: python

    multi_cohort_names = ["TGFBR1", "TGFBR2", "SMAD3"]

    phenopackets = load_phenopackets_by_cohort(multi_cohort_names)
    print(f"Loaded {len(phenopackets)} phenopackets")

Load all available phenopackets:

.. code-block:: python

    phenopackets_all = load_phenopackets_by_cohort()
    print(f"Loaded {len(phenopackets_all)} phenopackets from all available cohorts")



Load by disease
~~~~~~~~~~~~~~~

Retrieve phenopackets associated with a specific disease identifier.

.. code-block:: python

    from ppkt2synergy import load_phenopackets_by_disease

    phenopackets = load_phenopackets_by_disease("OMIM:614816")
    print(f"Loaded {len(phenopackets)} phenopackets")

Aggregate phenopackets for multiple diseases:

.. code-block:: python

    phenopackets = load_phenopackets_by_disease([
        "OMIM:614816",
        "OMIM:610168",
    ])
    print(f"Loaded {len(phenopackets)} phenopackets")



Reproducibility
---------------

By default, **ppkt2synergy** uses the latest available release of the Phenopacket Store.

To ensure reproducible analyses, you can specify a particular version:

.. code-block:: python

    phenopackets = load_phenopackets_by_cohort(
        "TGFBR1",
        ppkt_store_version="0.1.23"
    )



Next step
---------

Once phenopackets are loaded, proceed to dataset construction:

See :doc:`dataset` for instructions on how to create a **ppkt2synergy** dataset from your phenopacket data.