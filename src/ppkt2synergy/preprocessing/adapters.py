from typing import Iterable
import phenopackets as ppkt
import logging

from ..core import PhenopacketRecord
from ..io.phenopacket_loader import EnrichedPhenopacket

logger = logging.getLogger(__name__)

def phenopacket_to_patient_record(
    phenopacket: ppkt.Phenopacket,
    cohort: str | None = None,
) -> PhenopacketRecord:
    """
    Convert a single Phenopacket into a PhenopacketRecord.

    Args:
        phenopacket : ppkt.Phenopacket
            Input phenopacket.
        cohort : str | None, optional
            Cohort label (if available).

    Returns:
        PhenopacketRecord
            Standardized internal representation.
    """
    patient_id = phenopacket.id

    # -------------------------
    # HPO terms
    # -------------------------
    observed_terms = set()
    excluded_terms = set()

    for f in phenopacket.phenotypic_features:
        if not f.type or not f.type.id:
            continue

        term_id = f.type.id
        if getattr(f, "excluded", False):
            excluded_terms.add(term_id)
        else:
            observed_terms.add(term_id)

    # -------------------------
    # Diseases
    # -------------------------
    observed_diseases = set()
    excluded_diseases = set()
    for d in phenopacket.diseases:
        if not d.term or not d.term.label:
            continue
        disease_label = d.term.label
        if getattr(d, "excluded", False):
            excluded_diseases.add(disease_label)
        else:
            observed_diseases.add(disease_label)
            

    # -------------------------
    # Sex / Age (optional)
    # -------------------------
    sex = None
    age = None

    if phenopacket.subject:
        sex_value = getattr(phenopacket.subject, "sex", None)
        sex_map = {
            0: "unknown",
            1: "female",
            2: "male",
            3: "other",
        }
        sex = sex_map.get(sex_value, None)
        age = getattr(phenopacket.subject, "time_at_last_encounter", None)

    # -------------------------
    # Metadata (PMIDs etc.)
    # -------------------------
    metadata = {}

    pmids = []
    if phenopacket.meta_data:
        for ref in phenopacket.meta_data.external_references:
            if hasattr(ref, "id") and ref.id.startswith("PMID:"):
                pmids.append(ref.id.replace("PMID:", ""))

    if pmids:
        metadata["pmids"] = sorted(set(pmids))

    return PhenopacketRecord(
        patient_id=patient_id,
        cohort=cohort,
        observed_hpo_terms=observed_terms,
        excluded_hpo_terms=excluded_terms,
        observed_diseases=observed_diseases,
        excluded_diseases=excluded_diseases,
        sex=sex,
        age=age,
        metadata=metadata,
    )


# ================================
# Batch Conversion
# ================================

def phenopackets_to_records(
    phenopackets: Iterable[ppkt.Phenopacket],
    cohort: str | None = None,
) -> list[PhenopacketRecord]:
    """
    Convert a list of Phenopackets into PhenopacketRecords.

    Args:
        phenopackets : Iterable[Phenopacket]
            Input phenopackets.
        cohort : str | None
            Cohort label to assign to all records (if available).

    Returns:
        List[PhenopacketRecord]
            List of standardized internal representations.
    -----
    """
    records = []
    seen_ids = set()
    for p in phenopackets:
        if getattr(p, "id", None) in seen_ids:
            logger.warning("Skipping duplicate phenopacket %s", getattr(p, "id", "unknown"))
            continue
        seen_ids.add(getattr(p, "id", None))
        
        try:
            records.append(phenopacket_to_patient_record(p, cohort=cohort))
        except Exception as e:
            logger.warning("Skipping phenopacket %s: %s", getattr(p, "id", "unknown"), e)
    return records


def enriched_phenopackets_to_records(
    enriched: list[EnrichedPhenopacket],
) -> list[PhenopacketRecord]:
    """
    Convert EnrichedPhenopacket objects into PhenopacketRecords.

    Args:
        enriched : List[EnrichedPhenopacket]
                List of enriched phenopackets containing both the original phenopacket
                and additional metadata (e.g., cohort information).

    Returns:
        List[PhenopacketRecord]
                List of standardized patient records derived from the enriched phenopackets.
    """
    records = []
    seen_ids = set()
    for item in enriched:
        if getattr(item.phenopacket, "id", None) in seen_ids:
            logger.warning("Skipping duplicate enriched phenopacket %s", getattr(item, "id", "unknown"))
            continue
        seen_ids.add(getattr(item, "id", None))

        try:
            records.append(
                phenopacket_to_patient_record(
                    item.phenopacket,
                    cohort=item.cohort,
                )
            )
        except Exception as e:
            logger.warning("Skipping enriched phenopacket %s: %s", item.id, e)
    return records