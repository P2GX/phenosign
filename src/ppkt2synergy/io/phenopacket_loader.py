from dataclasses import dataclass
import phenopackets as ppkt
from typing import List
from ppktstore.registry import configure_phenopacket_registry


@dataclass(slots=True)
class EnrichedPhenopacket:
    """Phenopacket with cohort-level metadata."""
    phenopacket: ppkt.Phenopacket
    cohort: str

    @property
    def id(
            self
        ) -> str:
        """Get patient ID from the underlying phenopacket."""
        return self.phenopacket.id



def load_phenopackets(
    cohorts: str | List[str] | None = None,
    ppkt_store_version: str | None = None,
) -> List[EnrichedPhenopacket]:
    """
    Load Phenopackets from the configured Phenopacket Store.

    Args:
        cohorts: str | list[str] | None, optional
            Cohort name or list of cohort names to load.
            If None, all cohorts are loaded.  

        ppkt_store_version: str | None, optional
            Release tag of the Phenopacket Store (e.g., `'0.1.23'`).  
            If `None`, the latest release is used.
    Returns:
        List[EnrichedPhenopacket]: 
            A list of EnrichedPhenopacket instances containing the loaded phenopackets and their associated cohort metadata.     
    """
    registry = configure_phenopacket_registry()
    try:
        with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
            if cohorts is None:
                # Load all phenopackets from the store
                cohort_names = [cohort.name for cohort in ps.cohorts()]  
            elif isinstance(cohorts, str):
                cohort_names = [cohorts]
            elif isinstance(cohorts, list) and all(isinstance(c, str) for c in cohorts):
                cohort_names = cohorts
            else:
                raise TypeError(
                    "`cohorts` must be str, list[str], or None "
                    f"(got {type(cohorts).__name__})"
                )
            # --- validate cohort existence ---
            available = {c.name for c in ps.cohorts()}
            invalid = [c for c in cohort_names if c not in available]
            if invalid:
                raise ValueError(f"Unknown cohorts: {invalid}")

            # --- load phenopackets ---
            enrichedphenopackets = []
            for name in cohort_names:
                for ppkt in ps.iter_cohort_phenopackets(name):
                    enrichedphenopackets.append(EnrichedPhenopacket(phenopacket=ppkt, cohort=name))
        return enrichedphenopackets
    except Exception as e:
        raise RuntimeError(
            f"Failed to load phenopackets from store "
            f"(release={ppkt_store_version})"
        ) from e

