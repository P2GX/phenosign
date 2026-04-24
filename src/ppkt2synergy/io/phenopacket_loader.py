from dataclasses import dataclass
from collections.abc import Sequence
import phenopackets as ppkt
from ppktstore.registry import configure_phenopacket_registry

@dataclass(slots=True)
class EnrichedPhenopacket:
    """Phenopacket annotated with cohort membership."""
    phenopacket: ppkt.Phenopacket
    cohort: str

    @property
    def phenopacket_id(
        self
    ) -> str:
        """Return the individual ID from the underlying Phenopacket."""
        return self.phenopacket.id


def load_phenopackets_by_cohort(
    cohorts: str | Sequence[str] | None = None,
    ppkt_store_version: str | None = None,
) -> list[EnrichedPhenopacket]:
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
    list[EnrichedPhenopacket]
        Loaded phenopackets annotated with cohort membership.
    """
    registry = configure_phenopacket_registry()
    try:
        with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
            available = [c.name for c in ps.cohorts()]
            available_set = set(available)

            if cohorts is None:
                cohort_names = available
            elif isinstance(cohorts, str):
                cohort_names = [cohorts]
            elif isinstance(cohorts, Sequence): 
                non_str = [type(c).__name__ for c in cohorts if not isinstance(c, str)]
                if non_str:
                    raise TypeError(
                        "`cohorts` must be a string or a sequence of strings "
                        f"(found non-string elements: {non_str})"
                    )
                cohort_names = list(cohorts)
            else:
                raise TypeError(
                    "`cohorts` must be str, sequence[str], or None "
                    f"(got {type(cohorts).__name__})"
                )

            invalid = [c for c in cohort_names if c not in available_set]
            if invalid:
                raise ValueError(f"Unknown cohorts: {invalid}")

            enriched_phenopackets = []
            for cohort_name in cohort_names:
                for phenopacket in ps.iter_cohort_phenopackets(cohort_name):
                    enriched_phenopackets.append(EnrichedPhenopacket(phenopacket=phenopacket, cohort=cohort_name))
        return enriched_phenopackets
    
    except (TypeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to load phenopackets from store "
            f"(release={ppkt_store_version})"
        ) from e

    
def load_phenopackets_by_disease(
    diseases: str | Sequence[str],
    ppkt_store_version: str | None = None,
) -> list[EnrichedPhenopacket]:
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
    list[EnrichedPhenopacket]
        Phenopackets that contain at least one matching observed disease and annotated with cohort membership.
    """
    if isinstance(diseases, str):
        disease_set = {diseases}
    elif isinstance(diseases, Sequence):
        non_str = [type(d).__name__ for d in diseases if not isinstance(d, str)]
        if non_str:
            raise TypeError(
                "`diseases` must be a string or a sequence of strings "
                f"(found non-string elements: {non_str})"
            )
        disease_set = set(diseases)
    else:
        raise TypeError(
            "`diseases` must be a string or a sequence of strings "
            f"(got {type(diseases).__name__})"
        )

    registry = configure_phenopacket_registry()

    try:
        matched_phenopackets: list[EnrichedPhenopacket] = []

        with registry.open_phenopacket_store(release=ppkt_store_version) as ps:
            for cohort in ps.cohorts():
                for phenopacket in ps.iter_cohort_phenopackets(cohort.name):

                    for disease in phenopacket.diseases:
                        if not disease.term or not disease.term.id:
                            continue
                        if getattr(disease, "excluded", False):
                            continue
                        if disease.term.id in disease_set:
                            matched_phenopackets.append(EnrichedPhenopacket(phenopacket=phenopacket, cohort=cohort.name))
                            break

        return matched_phenopackets
    except(TypeError,ValueError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to load phenopackets from store (release={ppkt_store_version})"
        ) from e