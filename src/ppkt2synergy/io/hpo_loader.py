import hpotk
from typing import IO
import logging
logger = logging.getLogger(__name__)

def load_hpo(
        file: IO | str | None = None,
        release: str | None = None
    ) -> hpotk.MinimalOntology:
    """
    Loads the HPO ontology from a localfile or a specified release, with fallback.

    The function first attempts to load the ontology from the provided `file`. 
    If that fails or no file is provided, it will try to load the ontology 
    from the specified `release`. If neither is provided, the latest HPO release is fetched.

    Args:
        file: IO | str | None , optional
            Path or file object for the local HPO ontology file.  
        release: str | None, optional:
            Specific HPO release version to load (e.g., '2024-06-01'). 

    Returns:
        hpotk.MinimalOntology: 
            The loaded HPO ontology object.

    Example:
        # Load from local file first, fallback to latest release if file fails
        ontology = load_hpo(file="path/to/hpo.json")

        # Load from a specific release
        ontology = load_hpo(release="2025-10-22")

        # Load the latest release if neither file nor release is provided
        ontology = load_hpo()
    """

    if file is not None:
        try:
            return hpotk.load_minimal_ontology(file)
        except (IOError, FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load HPO ontology from file: {e}")
    
    store = hpotk.configure_ontology_store()
    if release is not None:
        try:
            return store.load_minimal_hpo(release=release)
        except Exception as e:
            logger.warning(f"Failed to load HPO ontology for release '{release}': {e}")
    try:
        ontology = store.load_minimal_hpo()
        logger.info("Loaded HPO ontology from latest release")
        return ontology
    except Exception as e:
        raise RuntimeError("Failed to load HPO ontology from any source") from e
    
