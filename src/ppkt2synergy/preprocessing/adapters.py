from typing import Iterable
import phenopackets as ppkt
import logging

from ..core import PhenopacketRecord
from ..io.phenopacket_loader import EnrichedPhenopacket

logger = logging.getLogger(__name__)

def phenopacket_to_record(
    phenopacket: ppkt.Phenopacket,
    cohort: str | None = None,
) -> PhenopacketRecord:
    """
    Convert a Phenopacket into a PhenopacketRecord.

    Parameters
    ----------
    phenopacket : ppkt.Phenopacket
        Input phenopacket.
    cohort : str | None, optional
        Cohort label associated with the phenopacket.

    Returns
    -------
    PhenopacketRecord
        Standardized internal representation.
    """
    individual_id = phenopacket.id

    observed_terms = set()
    excluded_terms = set()

    for feature in phenopacket.phenotypic_features:
        if not feature.type or not feature.type.id:
            continue

        term_id = feature.type.id
        if getattr(feature, "excluded", False):
            excluded_terms.add(term_id)
        else:
            observed_terms.add(term_id)

    observed_diseases = set()
    excluded_diseases = set()
    for disease in phenopacket.diseases:
        if not disease.term or not disease.term.label:
            continue
        disease_label = disease.term.label
        if getattr(disease, "excluded", False):
            excluded_diseases.add(disease_label)
        else:
            observed_diseases.add(disease_label)

    sex = None
    age = None

    if phenopacket.subject:
        sex_value = getattr(phenopacket.subject, "sex", None)
        sex_map = {
            1: "female",
            2: "male",
        }
        sex = sex_map.get(sex_value, None)
        encounter = getattr(phenopacket.subject, "time_at_last_encounter", None)
        if encounter and getattr(encounter, "age", None):
            age_obj = encounter.age
            age = getattr(age_obj, "iso8601duration", None)

    pmids = []

    if phenopacket.meta_data:
        for ref in phenopacket.meta_data.external_references:
            if hasattr(ref, "id") and ref.id.startswith("PMID:"):
                pmids.append(ref.id.replace("PMID:", ""))

    metadata = {
        "pmids": sorted(set(pmids))
    }

    return PhenopacketRecord(
        individual_id=individual_id,
        cohort=cohort,
        observed_hpo_terms=observed_terms,
        excluded_hpo_terms=excluded_terms,
        observed_diseases=observed_diseases,
        excluded_diseases=excluded_diseases,
        sex=sex,
        age=age,
        metadata=metadata,
    )


def phenopackets_to_records(
    phenopackets: Iterable[ppkt.Phenopacket],
    cohort: str | None = None,
) -> list[PhenopacketRecord]:
    """
    Convert phenopackets into PhenopacketRecord objects.

    Parameters
    ----------
    phenopackets : Iterable[ppkt.Phenopacket]
        Input phenopackets.
    cohort : str | None, optional
        Cohort label assigned to all records.

    Returns
    -------
    list[PhenopacketRecord]
        Converted records.
    """
    records = []
    seen_ids = set()

    for phenopacket in phenopackets:
        phenopacket_id = getattr(phenopacket, "id", None)
        if not phenopacket_id:
            raise ValueError("Phenopacket must have a non-empty `id`.")
        if phenopacket_id in seen_ids:
            logger.warning("Skipping duplicate phenopacket %s", phenopacket_id or "unknown")
            continue
        seen_ids.add(phenopacket_id)
        
        try:
            records.append(phenopacket_to_record(phenopacket, cohort=cohort))
        except Exception as e:
            logger.warning("Skipping phenopacket %s: %s", phenopacket_id or "unknown", e)
    return records


def enriched_phenopackets_to_records(
    enriched: Iterable[EnrichedPhenopacket],
) -> list[PhenopacketRecord]:
    """
    Convert enriched phenopackets into PhenopacketRecord objects.

    Parameters
    ----------
    enriched : Iterable[EnrichedPhenopacket]
        Enriched phenopackets with cohort metadata.

    Returns
    -------
    list[PhenopacketRecord]
        Converted records.
    """
    records = []
    seen_ids = set()

    for item in enriched:
        phenopacket_id = item.phenopacket_id

        if not phenopacket_id:
            raise ValueError("EnrichedPhenopacket contains phenopacket without id")

        if phenopacket_id in seen_ids:
            logger.warning("Skipping duplicate enriched phenopacket %s", phenopacket_id)
            continue
        seen_ids.add(phenopacket_id)

        try:
            records.append(
                phenopacket_to_record(
                    item.phenopacket,
                    cohort=item.cohort,
                )
            )
        except Exception as e:
            logger.warning("Skipping enriched phenopacket %s: %s", phenopacket_id, e)
            
    return records