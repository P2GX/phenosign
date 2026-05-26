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

    Raises
    ------
    TypeError
        If `cohorts` is not of the expected type.
    ValueError
        If any specified cohort is not found in the store.
    """
    registry = configure_phenopacket_registry()

    with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
        available_cohorts = [c.name for c in ps.cohorts()]

        if cohorts is None:
            cohort_names = available_cohorts
        elif isinstance(cohorts, str):
            cohort_names = [cohorts]
        elif isinstance(cohorts, Sequence): 
            cohort_names = list(cohorts)
        else:
            raise TypeError(
                "`cohorts` must be str, sequence[str], or None, "
                f"(got {type(cohorts).__name__})"
            )
        
        cohort_names = list(dict.fromkeys(cohort_names))

        invalid = [c for c in cohort_names if c not in available_cohorts]
        if invalid:
            raise ValueError(f"Cohorts not found in store: {invalid}")
        
        phenopackets = [
            p
            for cohort_name in cohort_names
            for p in ps.iter_cohort_phenopackets(cohort_name)
        ]
        
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
        (e.g., ``"OMIM:614816"``). Must not be empty or whitespace.
    ppkt_store_version : str | None, optional
        Phenopacket Store release tag, for example ``"0.1.23"``.
        If ``None``, the latest release is used.

    Returns
    -------
    list[ppkt.Phenopacket]
        Phenopackets that contain at least one matching observed disease.

    Raises
    ------
    TypeError
        If `diseases` is not str or sequence of str.
    ValueError
        If `diseases` is empty or only whitespace.
    """
    
    if isinstance(diseases, str):
        diseases = [diseases]
    elif not isinstance(diseases, Sequence):
        raise TypeError("`diseases` must be str or sequence[str].")

    disease_list = [d.strip() for d in diseases if d.strip()]
    if not disease_list:
        raise ValueError("`diseases` must not be empty or only whitespace.")
    disease_set = set(disease_list)

    registry = configure_phenopacket_registry()
    matched: list[ppkt.Phenopacket] = []
    found_ids: set[str] = set()

    with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
        for cohort in ps.cohorts():
            for phenopacket in ps.iter_cohort_phenopackets(cohort.name):
                for disease in phenopacket.diseases:
                    did = getattr(disease.term, "id", None)
                    if not did or getattr(disease, "excluded", False):
                        continue

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