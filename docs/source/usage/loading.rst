Loading Phenopackets
====================

**ppkt2synergy** is designed to work with any phenopacket data conforming to the
`GA4GH Phenopacket schema <https://phenopacket-schema.readthedocs.io/en/latest/index.html>`_.

For convenience, it also provides helper utilities to retrieve example datasets from the
`Phenopacket Store <https://github.com/monarch-initiative/phenopacket-store>`_
for quick exploration and testing.


Using your own phenopacket data
-------------------------------

Pass your own phenopacket objects directly into downstream steps.
The example below shows how to load phenopackets from a local ``.json`` file:

.. code-block:: python

    from pathlib import Path
    from phenopackets import Phenopacket
    from google.protobuf.json_format import Parse

    phenopackets = [
        Parse(p.read_text(), Phenopacket())
        for p in Path("data/").glob("*.json")
    ]
    
Any list of ``Phenopacket`` objects can be passed directly into dataset construction and analysis.


Load from the Phenopacket Store
-------------------------------

The following utilities download datasets from the Phenopacket Store.
Browse available cohort names and disease identifiers in the
`repository <https://github.com/monarch-initiative/phenopacket-store>`_
before loading.

.. tip::

   These helpers are intended for exploration and testing.


Load by cohort
^^^^^^^^^^^^^^

Load phenopackets for one or more gene-based cohorts:

.. code-block:: python

    from ppkt2synergy import load_phenopackets_by_cohort

    # Single cohort
    phenopackets = load_phenopackets_by_cohort("TGFBR1")

    # Multiple cohorts — returns a single combined list
    phenopackets = load_phenopackets_by_cohort(["TGFBR1", "TGFBR2", "SMAD3"])

    print(f"Loaded {len(phenopackets)} phenopackets")

.. warning::

   Calling ``load_phenopackets_by_cohort()`` with no arguments downloads the entire
   Phenopacket Store, which may take several minutes.


Load by disease
^^^^^^^^^^^^^^^

Load phenopackets for one or more disease identifiers:

.. code-block:: python

    from ppkt2synergy import load_phenopackets_by_disease

    # Single disease
    phenopackets = load_phenopackets_by_disease("OMIM:614816")

    # Multiple diseases — returns a single combined list
    phenopackets = load_phenopackets_by_disease(["OMIM:614816", "OMIM:610168"])

    print(f"Loaded {len(phenopackets)} phenopackets")


Pinning a release for reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, **ppkt2synergy** uses the latest available Phenopacket Store release.
To ensure a reproducible analysis, pin a specific version:

.. code-block:: python

    phenopackets = load_phenopackets_by_cohort(
        "TGFBR1",
        ppkt_store_version="0.1.23",
    )

The ``ppkt_store_version`` parameter is supported by both
``load_phenopackets_by_cohort`` and ``load_phenopackets_by_disease``.


Next step
---------

Once phenopackets are loaded, proceed to :doc:`dataset construction <dataset>`.