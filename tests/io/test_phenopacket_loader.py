import pytest
from ppkt2synergy import load_phenopackets, EnrichedPhenopacket


@pytest.mark.integration
def test_load_single_cohort_success():
    """
    Test loading a single valid cohort.
    """
    cohort_name = "FBN1"
    version = "0.1.23"

    result = load_phenopackets(
        cohorts=cohort_name,
        ppkt_store_version=version,
    )

    assert isinstance(result, list)
    assert len(result) == 144

    assert isinstance(result[0], EnrichedPhenopacket)

    assert result[0].cohort == cohort_name
    assert hasattr(result[0], "phenopacket")


@pytest.mark.integration
def test_load_multiple_cohorts():
    """
    Test loading multiple cohorts.
    """
    cohorts = ["TGFBR1","TGFBR2","SMAD3","TGFB2","TGFB3","SMAD2"]

    result = load_phenopackets(cohorts=cohorts)

    assert isinstance(result, list)
    assert len(result) == 277
    assert all(isinstance(x, EnrichedPhenopacket) for x in result)


def test_invalid_cohort_type():
    """
    Test invalid type for cohorts argument.
    """
    with pytest.raises(RuntimeError) as excinfo:
        load_phenopackets(cohorts=123)

    assert "Failed to load phenopackets" in str(excinfo.value)


def test_invalid_cohort_name():
    """
    Test non-existent cohort name.
    """
    with pytest.raises(RuntimeError) as excinfo:
        load_phenopackets(cohorts="NOT_EXIST")

    assert "Failed to load phenopackets" in str(excinfo.value)


def test_mixed_invalid_cohort_list():
    """
    Test partially invalid cohort list.
    """
    with pytest.raises(RuntimeError):
        load_phenopackets(cohorts=["FBN1", "INVALID"])

