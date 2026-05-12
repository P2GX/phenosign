from collections.abc import Sequence

import phenopackets as ppkt
from ppktstore.registry import configure_phenopacket_registry
import logging

logger = logging.getLogger(__name__)

def load_phenopackets_by_cohort(
    cohorts: str | Sequence[str] | None = None,
    ppkt_store_version: str | None = None,
) -> list[ppkt.Phenopacket]:
    """
    Load phenopackets from the Phenopacket Store by cohort.

    Parameters
    ----------
    cohorts : str | Sequence[str] | None, optional
        Cohort name, e.g. `FBN1`or a sequence of cohort names to load. If ``None``,
        all available cohorts are loaded.
    ppkt_store_version : str | None, optional
        Phenopacket Store release tag, for example ``"0.1.23"``.
        If ``None``, the latest release is used.

    Returns
    -------
    list[ppkt.Phenopacket]
        Loaded phenopackets.
    """
    registry = configure_phenopacket_registry()

    with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
        available = [c.name for c in ps.cohorts()]

        if cohorts is None:
            cohort_names = available
        elif isinstance(cohorts, str):
            cohort_names = [cohorts]
        elif isinstance(cohorts, Sequence): 
            cohort_names = list(cohorts)
        else:
            raise TypeError(
                "`cohorts` must be str, sequence[str], or None, "
                f"(got {type(cohorts).__name__})"
            )
        
        invalid = [c for c in cohort_names if c not in available]
        if invalid:
            raise ValueError(f"Cohorts not found in store: {invalid}")
        
        cohort_names = list(dict.fromkeys(cohort_names))
        phenopackets = []
        for cohort_name in cohort_names:
            for phenopacket in ps.iter_cohort_phenopackets(cohort_name):
                phenopackets.append(phenopacket)
    return phenopackets
    
    
def load_phenopackets_by_disease(
    diseases: str | Sequence[str],
    ppkt_store_version: str | None = None,
) -> list[ppkt.Phenopacket]:
    """
    Load phenopackets with observed disease identifiers matching one or
    more requested diseases.

    Parameters
    ----------
    diseases : str | Sequence[str]
        Disease identifier or a sequence of identifiers in CURIE format
        (e.g., ``"OMIM:614816"``).
    ppkt_store_version : str | None, optional
        Phenopacket Store release tag, for example ``"0.1.23"``.
        If ``None``, the latest release is used.

    Returns
    -------
    list[ppkt.Phenopacket]
        Phenopackets that contain at least one matching observed disease.
    """
    
    if isinstance(diseases, str):
        if not diseases.strip():
            raise ValueError("`diseases` must not be empty or only whitespace.")
        disease_list = [diseases]
    elif isinstance(diseases, Sequence):
        disease_set = {d for d in diseases if d.strip()}  
        if not disease_set:
            raise ValueError("`diseases` must not be empty or only whitespace.")
    else:
        raise TypeError("`diseases` must be str or sequence[str].")

    registry = configure_phenopacket_registry()
    matched: list[ppkt.Phenopacket] = []
    found_ids: set[str] = set()

    with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
        for cohort in ps.cohorts():
            for phenopacket in ps.iter_cohort_phenopackets(cohort.name):
                for disease in phenopacket.diseases:
                    if not disease.term or not disease.term.id:
                        continue
                    if getattr(disease, "excluded", False):
                        continue

                    did = disease.term.id
                    if did in disease_set:
                        matched.append(phenopacket)
                        found_ids.add(did)
                        break

    missing = [d for d in disease_list if d not in found_ids]
    if missing:
        logger.warning(
            "No phenopackets matched disease identifier(s): %s."
            "This may indicate incorrect identifiers or absence in the dataset.",
            missing,
        )

    return matched

    