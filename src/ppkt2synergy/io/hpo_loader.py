from typing import IO
import logging
import hpotk

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _load_hpo(
    file: IO | str | None = None,
    release: str | None = None
) -> hpotk.MinimalOntology:
    """
    Load a minimal Human Phenotype Ontology (HPO) instance.

    The ontology is loaded from ``file`` if provided, otherwise from the
    specified ``release``. If neither file nor release is provided, the latest available release is loaded by default.

    Parameters
    ----------
    file : IO | str | None, optional
        A local file path, remote URL (str), or an opened file-like object (IO), pointing to an HPO ontology file (e.g. ``hp.json``).
    release : str | None, optional
        HPO release tag, for example ``"v2023-10-09"``, used when ``file`` is ``None``.

    Returns
    -------
    hpotk.MinimalOntology
        Loaded HPO ontology.
    """

    if file is not None:
        ontology = hpotk.load_minimal_ontology(file)
        logger.info("Loaded HPO ontology from file: %s", file)
        return ontology
    
    # Fall back to the ontology store for release-based or latest loading.
    store = hpotk.configure_ontology_store()
    
    if release is not None:
        ontology = store.load_minimal_hpo(release=release)
        logger.info("Loaded HPO ontology from release: %s", release)
        return ontology

    ontology = store.load_minimal_hpo()
    logger.info("Loaded HPO ontology from the latest available release.")
    return ontology