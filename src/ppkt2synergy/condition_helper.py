from __future__ import annotations

from collections.abc import Callable

import phenopackets as ppkt


def has_disease(disease_id: str) -> Callable[[ppkt.Phenopacket], bool | None]:
    """
    Predicate for disease status.

    Returns:
        True  = disease observed
        False = disease explicitly excluded
        None  = disease not mentioned
    """

    def predicate(phenopacket: ppkt.Phenopacket) -> bool | None:

        for disease in getattr(phenopacket, "diseases", []):
            term = getattr(disease, "term", None)
            term_id = getattr(term, "id", None)

            if term_id != disease_id:
                continue

            if not getattr(disease, "excluded", False):
                return True


        return False

    return predicate


def has_sex(sex: str) -> Callable[[ppkt.Phenopacket], bool | None]:
    normalized = sex.lower()

    if normalized not in {"female", "male"}:
        raise ValueError("`sex` must be either 'female' or 'male'.")

    sex_map = {
        1: "female",
        2: "male",
    }

    def predicate(phenopacket: ppkt.Phenopacket) -> bool | None:
        subject = getattr(phenopacket, "subject", None)
        if subject is None:
            return None

        parsed = sex_map.get(getattr(subject, "sex", None))
        if parsed is None:
            return None

        return parsed == normalized

    return predicate


def has_gene(symbol: str) -> Callable[[ppkt.Phenopacket], bool | None]:
    """
    Predicate for gene presence.

    Returns:
        True  = target gene found
        False = genomic information exists but target gene not found
        None  = no genomic information
    """

    def predicate(phenopacket: ppkt.Phenopacket) -> bool | None:
        saw_genomic_information = False

        for interpretation in getattr(phenopacket, "interpretations", []):
            diagnosis = getattr(interpretation, "diagnosis", None)
            if diagnosis is None:
                continue

            for genomic_interpretation in getattr(
                diagnosis,
                "genomic_interpretations",
                [],
            ):
                saw_genomic_information = True

                gene_descriptor = getattr(genomic_interpretation, "gene_descriptor", None)
                if gene_descriptor is not None:
                    if getattr(gene_descriptor, "symbol", None) == symbol:
                        return True

                variant_interpretation = getattr(
                    genomic_interpretation,
                    "variant_interpretation",
                    None,
                )
                if variant_interpretation is None:
                    continue

                variation_descriptor = getattr(
                    variant_interpretation,
                    "variation_descriptor",
                    None,
                )
                gene_context = getattr(variation_descriptor, "gene_context", None)

                if gene_context is not None:
                    if getattr(gene_context, "symbol", None) == symbol:
                        return True

        if saw_genomic_information:
            return False

        return None

    return predicate