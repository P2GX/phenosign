from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PhenopacketRecord:
    """
    Unified internal representation of a patient for phenotype-genotype analysis.

    This class serves as the canonical data structure used throughout the
    preprocessing and analysis pipeline. It abstracts away the original
    data source format (e.g., Phenopacket) and provides a consistent,
    lightweight representation for downstream processing.

    Notes
    -----
    - All fields are optional except `patient_id`.
    - Missing information should be represented as `None` or empty containers.
    - This object is intentionally simple and serializable.
    - It is not responsible for any computation or validation beyond structure.

    Attributes
    ----------
    patient_id : str
        Unique identifier for the patient/sample.

    cohort : str | None, optional
        Name of the cohort or study group the patient belongs to.
        Used for stratification or optional target construction.

    observed_hpo_terms : set[str]
        Set of HPO term IDs (e.g., "HP:0001250") observed in the patient.
        These correspond to *present* phenotypic features.

    excluded_hpo_terms : set[str]
        Set of HPO term IDs explicitly excluded for the patient.
        These correspond to *negated* phenotypic features.

    observed_diseases : set[str]
        Set of diagnosed disease identifiers (e.g., OMIM IDs or labels).
        Used for constructing disease target matrices.

    excluded_diseases : set[str]
        Set of disease identifiers explicitly excluded for the patient.
        These correspond to *negated* disease diagnoses.

    sex : str | None, optional
        Biological sex of the patient (e.g., "male", "female").
        May be used as metadata or as a target variable in downstream analyses.

    age : str | None, optional
        Age of the patient (format not enforced).
        Stored as string to preserve original representation.

    metadata : dict[str, Any]
        Additional arbitrary metadata associated with the patient.

        Examples:
            - pmids (list of PubMed IDs)
            - sequencing platform
            - clinical notes

        This field is not used directly in analysis but may be used for:
            - visualization
            - reporting
            - downstream filtering
    """
    patient_id: str
    cohort: str | None = None

    observed_hpo_terms: set[str] = field(default_factory=set)
    excluded_hpo_terms: set[str] = field(default_factory=set)

    observed_diseases: set[str] = field(default_factory=set)
    excluded_diseases: set[str] = field(default_factory=set)

    sex: str | None = None
    age: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)