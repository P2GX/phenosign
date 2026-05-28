import pytest
import phenopackets as ppkt

from ppkt2synergy import (
    load_phenopackets_by_cohort,
    load_phenopackets_by_disease,
)

VERSION = "0.1.23"


# =========================
# cohort tests
# =========================

@pytest.mark.integration
@pytest.mark.parametrize(
    "cohorts",
    [
        "FBN1",
        ["TGFBR1", "TGFBR2"],
        ("TGFBR1", "TGFBR2"),
        None,
    ],
)
def test_load_phenopackets_by_cohort_success(cohorts):
    result = load_phenopackets_by_cohort(
        cohorts=cohorts,
        ppkt_store_version=VERSION,
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, ppkt.Phenopacket) for x in result)


def test_load_phenopackets_by_cohort_invalid_type():
    with pytest.raises(TypeError, match="`cohorts` must be str"):
        load_phenopackets_by_cohort(cohorts=123)


def test_load_phenopackets_by_cohort_invalid_name():
    with pytest.raises(ValueError, match="Cohorts not found"):
        load_phenopackets_by_cohort(cohorts="NOT_EXIST")


def test_load_phenopackets_by_cohort_partial_invalid():
    with pytest.raises(ValueError, match="Cohorts not found"):
        load_phenopackets_by_cohort(
            cohorts=["FBN1", "INVALID"]
        )


# =========================
# disease tests
# =========================

@pytest.mark.integration
@pytest.mark.parametrize(
    "diseases",
    [
        "OMIM:154700",
        ["OMIM:154700", "OMIM:614816"],
        ("OMIM:154700", "OMIM:614816"),
    ],
)
def test_load_phenopackets_by_disease_success(diseases):
    result = load_phenopackets_by_disease(
        diseases=diseases,
        ppkt_store_version=VERSION,
    )

    assert isinstance(result, list)
    assert all(isinstance(x, ppkt.Phenopacket) for x in result)


@pytest.mark.integration
def test_load_phenopackets_by_disease_unknown():
    result = load_phenopackets_by_disease(
        diseases="OMIM:999999999",
        ppkt_store_version=VERSION,
    )

    assert result == []


def test_load_phenopackets_by_disease_invalid_type():
    with pytest.raises(TypeError, match="`diseases` must be str"):
        load_phenopackets_by_disease(diseases=123)


@pytest.mark.parametrize(
    "diseases",
    [
        "",
        "   ",
        [],
        ["", "   "],
    ],
)
def test_load_phenopackets_by_disease_empty(diseases):
    with pytest.raises(ValueError, match="must not be empty"):
        load_phenopackets_by_disease(diseases=diseases)