from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PhenopacketRecord:
    """
    Unified representation of a individual for phenotype–genotype analysis.

    Parameters
    ----------
    individual_id : str
        Unique identifier for the individual/sample.
    cohort : str | None, optional
        Cohort or study group name.

    observed_hpo_terms : set[str], default empty
        HPO term IDs (e.g., "HP:0001250") observed in the individual.
    excluded_hpo_terms : set[str], default empty
        HPO term IDs explicitly excluded for the individual.

    observed_diseases : set[str], default empty
        Diagnosed disease identifiers (e.g., OMIM IDs or labels).
    excluded_diseases : set[str], default empty
        Disease identifiers explicitly excluded for the individual.

    sex : str | None, optional
        Biological sex (e.g., "male", "female").
    age : str | None, optional
        Age of the individual (stored as string).

    metadata : dict[str, Any], default empty
        Additional metadata (e.g., PMIDs, sequencing platform).

    Notes
    -----
    This is a lightweight, serializable container used throughout the
    analysis pipeline. It does not perform validation or computation.
    Missing values should be represented as ``None`` or empty containers.
    """
    individual_id: str
    cohort: str | None = None

    observed_hpo_terms: set[str] = field(default_factory=set)
    excluded_hpo_terms: set[str] = field(default_factory=set)

    observed_diseases: set[str] = field(default_factory=set)
    excluded_diseases: set[str] = field(default_factory=set)

    sex: str | None = None
    age: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)