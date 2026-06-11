from collections.abc import Callable

import phenopackets as ppkt
from gpsea.model import VariantEffect, Patient


def has_disease(disease_id: str) -> Callable[[ppkt.Phenopacket], bool | None]:
    """
    Generate a predicate to check if a phenopacket matches a specific disease status.

    Parameters
    ----------
    disease_id : str
        The target disease identifier to query (e.g., "OMIM:154700").

    Returns
    -------
    Callable[[phenopackets.Phenopacket], bool | None]
        A predicate function that takes a Phenopacket and returns:
        
        - ``True`` : If the disease is explicitly listed as observed.
        - ``False`` : If the disease is explicitly marked as excluded, or not found.
        - ``None`` : (Reserved for missing disease block context, defaults to False here).
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
    """
    Generate a predicate to verify if a phenopacket matches the designated biological sex.

    Parameters
    ----------
    sex : str
        The biological sex to filter by. Must be either 'female' or 'male' (case-insensitive).

    Returns
    -------
    Callable[[phenopackets.Phenopacket], bool | None]
        A predicate function that takes a Phenopacket and returns:
        
        - ``True`` : If the individual's sex matches the specified criterion.
        - ``False`` : If the individual's sex is explicitly different.
        - ``None`` : If the subject context or sex info is entirely missing/unknown.

    Raises
    ------
    ValueError
        If the input `sex` string is not 'female' or 'male'.
    """
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

        raw_sex = getattr(subject, "sex", None)
        
        # Guard against UNKNOWN_SEX (0) or OTHER_SEX (3)
        if raw_sex in (0, 3) or raw_sex is None:
            return None

        parsed = sex_map.get(raw_sex)

        return parsed == normalized

    return predicate


def has_gene(symbol: str) -> Callable[[ppkt.Phenopacket], bool | None]:
    """
    Generate a predicate to detect the presence of causative variants in a target gene.

    Inspects both the ``gene_descriptor`` block and the ``gene_context`` within 
    the genomic interpretations of a phenopacket.

    Parameters
    ----------
    symbol : str
        The HGNC gene symbol to search for (e.g., "FBN1", "NOTCH1").

    Returns
    -------
    Callable[[phenopackets.Phenopacket], bool | None]
        A predicate function that takes a Phenopacket and returns:
        
        - ``True`` : If a diagnostic variant is found mapped to the target gene symbol.
        - ``False`` : If genomic interpretations exist, but none implicate the target gene.
        - ``None`` : If the phenopacket contains no genomic interpretations/diagnostic data.
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
                if variation_descriptor is None:
                    continue
                
                gene_context = getattr(variation_descriptor, "gene_context", None)

                if gene_context is not None:
                    if getattr(gene_context, "symbol", None) == symbol:
                        return True

        if saw_genomic_information:
            return False

        return None

    return predicate


def has_variant_effect(transcript_id: str, variant_effect: VariantEffect)-> Callable[[Patient], bool | None]:
    """
    Generate a predicate to filter GPSEA Patients by a specific molecular variant effect.

    Evaluates transcript annotations mapped to the specified transcript model identifier.

    Parameters
    ----------
    transcript_id : str
        The target transcript identifier (e.g., "NM_000138.5", "ENST00000316673").
    variant_effect : VariantEffect
        The target GPSEA ``VariantEffect`` enum or object to evaluate (e.g., MISSENSE_VARIANT).

    Returns
    -------
    Callable[[gpsea.model.Patient], bool | None]
        A predicate function that takes a GPSEA Patient and returns:
        
        - ``True`` : If the patient carries a variant with the exact effect on the transcript.
        - ``False`` : If annotations for the transcript exist, but the specified effect is absent.
        - ``None`` : If no annotations for the given transcript id are detected in this patient.
    """

    def predicate(patient: Patient) -> bool | None:

        saw_transcript = False
        has_effect = False

        for variant in patient.variants:
            for txa in variant.tx_annotations:
                if str(txa.transcript_id) != transcript_id:
                    continue

                saw_transcript = True

                effects = {
                    ve.name
                    for ve in txa.variant_effects
                }

                if variant_effect.name in effects:
                    has_effect = True
                    break

            if has_effect:
                break

        if has_effect:
            return True
        elif saw_transcript:
            return False
        else:
            return None

    return predicate


def has_exon_and_variant_effect(transcript_id: str, exon: int, variant_effect: VariantEffect)-> Callable[[Patient], bool | None]:
    """
    Generate a predicate to identify variants spanning both a specific exon and variant effect.

    Useful for granular genotype-phenotype analysis, such as isolating variants localized 
    within hotspot domains (e.g., FBN1 exons 24-32).

    Parameters
    ----------
    transcript_id : str
        The target transcript identifier (e.g., "NM_000138.5").
    exon : int
        The specific exon number expected to be affected (1-based index).
    variant_effect : VariantEffect
        The expected GPSEA ``VariantEffect`` consequence.

    Returns
    -------
    Callable[[gpsea.model.Patient], bool | None]
        A predicate function that takes a GPSEA Patient and returns:
        
        - ``True`` : If a variant disrupts the designated exon AND exhibits the specified effect.
        - ``False`` : If the transcript is tracked but no variant satisfies both criteria simultaneously.
        - ``None`` : If the transcript model itself is not annotated within the patient's variants.
    """

    def predicate(patient: Patient) -> bool | None:
        saw_transcript = False

        for variant in patient.variants:
            for txa in variant.tx_annotations:

                if str(txa.transcript_id) != transcript_id:
                    continue

                saw_transcript = True

                if (
                    txa.affected_exons is not None
                    and exon in txa.affected_exons
                    and variant_effect in txa.variant_effects
                ):
                    return True

        return False if saw_transcript else None

    return predicate
    