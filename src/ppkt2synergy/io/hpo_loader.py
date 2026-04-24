import hpotk
from typing import IO
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def load_hpo(
    file: IO | str | None = None,
    release: str | None = None
) -> hpotk.MinimalOntology:
    """
    Load a minimal Human Phenotype Ontology (HPO) instance.

    The ontology is loaded from ``file`` if provided, otherwise from the
    specified ``release``, and finally from the latest available release.

    Parameters
    ----------
    file : IO | str | None, optional
        Path, URL, or file-like object pointing to an HPO ontology file.
    release : str | None, optional
        HPO release identifier, for example ``"v2023-10-09"``.

    Returns
    -------
    hpotk.MinimalOntology
        Loaded HPO ontology.

    Examples
    --------
    >>> load_hpo(file="hp.json")
    >>> load_hpo(file="https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-10-09/hp.json")
    >>> load_hpo(release="v2023-10-09")
    >>> load_hpo()
    """

    if file is not None:
        ontology = hpotk.load_minimal_ontology(file)
        logger.info("Loaded HPO ontology from file: %s", file)
        return ontology

    store = hpotk.configure_ontology_store()
    
    if release is not None:
        ontology = store.load_minimal_hpo(release=release)
        logger.info("Loaded HPO ontology from release: %s", release)
        return ontology

    ontology = store.load_minimal_hpo()
    logger.info("Loaded HPO ontology from the latest available release.")
    return ontology
   
    
