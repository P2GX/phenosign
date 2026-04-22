import pytest
import phenopackets as ppkt

from ppkt2synergy import (
    EnrichedPhenopacket,
    load_phenopackets_by_cohort,
    load_phenopackets_by_disease,
)


@pytest.mark.integration
def test_load_single_cohort_success():
    """
    Test loading a single valid cohort.
    """
    cohort_name = "FBN1"
    version = "0.1.23"

    result = load_phenopackets_by_cohort(
        cohorts=cohort_name,
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, EnrichedPhenopacket) for x in result)
    assert all(x.cohort == cohort_name for x in result)
    assert all(hasattr(x, "phenopacket") for x in result)


@pytest.mark.integration
def test_load_multiple_cohorts_success_list():
    """
    Test loading multiple valid cohorts from a list.
    """
    cohorts = ["TGFBR1", "TGFBR2", "SMAD3", "TGFB2", "TGFB3", "SMAD2"]
    version = "0.1.23"

    result = load_phenopackets_by_cohort(
        cohorts=cohorts,
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, EnrichedPhenopacket) for x in result)

    observed_cohorts = {x.cohort for x in result}
    assert observed_cohorts.issubset(set(cohorts))
    assert len(observed_cohorts) > 0


@pytest.mark.integration
def test_load_multiple_cohorts_success_tuple():
    """
    Test loading multiple valid cohorts from a tuple.
    """
    cohorts = ("TGFBR1", "TGFBR2", "SMAD3", "TGFB2", "TGFB3", "SMAD2")
    version = "0.1.23"

    result = load_phenopackets_by_cohort(
        cohorts=cohorts,
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, EnrichedPhenopacket) for x in result)

    observed_cohorts = {x.cohort for x in result}
    assert observed_cohorts.issubset(set(cohorts))
    assert len(observed_cohorts) > 0


@pytest.mark.integration
def test_load_multiple_cohorts_tuple_matches_list():
    """
    Tuple input should behave the same as list input.
    """
    cohorts_list = ["TGFBR1", "TGFBR2", "SMAD3", "TGFB2", "TGFB3", "SMAD2"]
    cohorts_tuple = tuple(cohorts_list)
    version = "0.1.23"

    result_list = load_phenopackets_by_cohort(
        cohorts=cohorts_list,
        ppkt_store_version=version,
    )
    result_tuple = load_phenopackets_by_cohort(
        cohorts=cohorts_tuple,
        ppkt_store_version=version,
    )

    assert isinstance(result_list, list)
    assert isinstance(result_tuple, list)
    assert len(result_list) == len(result_tuple)

    ids_list = sorted(x.phenopacket_id for x in result_list)
    ids_tuple = sorted(x.phenopacket_id for x in result_tuple)
    assert ids_list == ids_tuple


@pytest.mark.integration
def test_load_all_cohorts_success():
    """
    Test loading all cohorts when cohorts=None.
    """
    version = "0.1.23"

    result = load_phenopackets_by_cohort(
        cohorts=None,
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, EnrichedPhenopacket) for x in result)
    assert all(isinstance(x.cohort, str) and x.cohort for x in result)


def test_invalid_cohort_type():
    """
    Test invalid type for cohorts argument.
    """
    with pytest.raises(TypeError) as excinfo:
        load_phenopackets_by_cohort(cohorts=123)

    assert "`cohorts` must be str, sequence[str], or None" in str(excinfo.value)


def test_invalid_cohort_sequence_element_type():
    """
    Test invalid element type inside cohort sequence.
    """
    with pytest.raises(TypeError) as excinfo:
        load_phenopackets_by_cohort(cohorts=["FBN1", 123])

    assert "`cohorts` must be a string or a sequence of strings (found non-string elements: ['int'])" in str(excinfo.value)


def test_invalid_cohort_name():
    """
    Test non-existent cohort name.
    """
    with pytest.raises(ValueError) as excinfo:
        load_phenopackets_by_cohort(cohorts="NOT_EXIST")

    assert "Unknown cohorts:" in str(excinfo.value)
    assert "NOT_EXIST" in str(excinfo.value)


def test_mixed_invalid_cohort_list():
    """
    Test partially invalid cohort list.
    """
    with pytest.raises(ValueError) as excinfo:
        load_phenopackets_by_cohort(cohorts=["FBN1", "INVALID"])

    assert "Unknown cohorts:" in str(excinfo.value)
    assert "INVALID" in str(excinfo.value)


def test_enriched_phenopacket_id_property():
    """
    Test EnrichedPhenopacket.id delegates to underlying phenopacket.id.
    """
    class DummyPhenopacket:
        id = "patient-001"

    enriched = EnrichedPhenopacket(
        phenopacket=DummyPhenopacket(),
        cohort="FBN1",
    )

    assert enriched.phenopacket_id == "patient-001"


@pytest.mark.integration
def test_load_single_disease_returns_list_of_phenopackets():
    """
    Test loading phenopackets for a single disease identifier.

    Note:
        Replace the disease ID below with one that is known to exist
        in your Phenopacket Store release if needed.
    """
    disease_id = "OMIM:154700"
    version = "0.1.23"

    result = load_phenopackets_by_disease(
        diseases=disease_id,
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert all(isinstance(x, ppkt.Phenopacket) for x in result)


@pytest.mark.integration
def test_load_multiple_diseases_list():
    """
    Test loading phenopackets for multiple disease identifiers from a list.

    Note:
        Replace the disease IDs below with IDs known to exist if needed.
    """
    disease_ids = ["OMIM:154700", "OMIM:614816"]
    version = "0.1.23"

    result = load_phenopackets_by_disease(
        diseases=disease_ids,
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert all(isinstance(x, ppkt.Phenopacket) for x in result)


@pytest.mark.integration
def test_load_multiple_diseases_tuple_matches_list():
    """
    Tuple input should behave the same as list input for diseases.
    """
    disease_ids_list = ["OMIM:154700", "OMIM:614816"]
    disease_ids_tuple = tuple(disease_ids_list)
    version = "0.1.23"

    result_list = load_phenopackets_by_disease(
        diseases=disease_ids_list,
        ppkt_store_version=version,
    )
    result_tuple = load_phenopackets_by_disease(
        diseases=disease_ids_tuple,
        ppkt_store_version=version,
    )

    assert isinstance(result_list, list)
    assert isinstance(result_tuple, list)

    ids_list = sorted(x.id for x in result_list)
    ids_tuple = sorted(x.id for x in result_tuple)
    assert ids_list == ids_tuple


@pytest.mark.integration
def test_unknown_disease_returns_empty_list():
    """
    Unknown disease identifier should return an empty list.
    """
    version = "0.1.23"

    result = load_phenopackets_by_disease(
        diseases="OMIM:999999999999",
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert result == []


def test_invalid_disease_type():
    """
    Test invalid type for diseases argument.
    """
    with pytest.raises(TypeError) as excinfo:
        load_phenopackets_by_disease(diseases=123)

    assert "`diseases` must be a string or a sequence of strings" in str(excinfo.value)


def test_invalid_disease_sequence_element_type():
    """
    Test invalid element type inside disease sequence.
    """
    with pytest.raises(TypeError) as excinfo:
        load_phenopackets_by_disease(diseases=["OMIM:154700", 123])

    assert "`diseases` must be a string or a sequence of strings" in str(excinfo.value)