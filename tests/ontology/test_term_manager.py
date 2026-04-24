import pathlib

import pytest

from ppkt2synergy.ontology import HPOTermManager


TEST_DIR = pathlib.Path(__file__).parent.parent.resolve()
HP_JSON_FILE = TEST_DIR / "data" / "hp.json"


@pytest.fixture
def term_manager():
    return HPOTermManager(hpo_file=str(HP_JSON_FILE))


def test_resolve_term_id_valid(term_manager):
    """
    Valid HPO term IDs should resolve to canonical IDs.
    """
    resolved = term_manager.resolve_term_id("HP:0001250")
    assert isinstance(resolved, str)
    assert resolved == "HP:0001250"


def test_resolve_term_id_invalid(term_manager):
    """
    Invalid HPO term IDs should raise ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        term_manager.resolve_term_id("HP:9999999")

    assert "not found in HPO ontology" in str(excinfo.value)


def test_prepare_terms_skips_invalid_terms(term_manager):
    """
    Invalid terms should be skipped, valid terms retained.
    """
    terms = {"HP:0001250", "HP:9999999"}

    resolved = term_manager.prepare_terms(terms)

    assert isinstance(resolved, set)
    assert "HP:0001250" in resolved
    assert "HP:9999999" not in resolved


def test_prepare_terms_warns_on_invalid_term(term_manager, caplog):
    """
    Invalid terms should trigger a warning during bulk preparation.
    """
    with caplog.at_level("WARNING"):
        resolved = term_manager.prepare_terms({"HP:9999999"})

    assert resolved == set()
    assert "Skipping term" in caplog.text
    assert "HP:9999999" in caplog.text


def test_prepare_terms_populates_id_mapping_cache(term_manager):
    """
    prepare_terms should populate original -> canonical ID mapping.
    """
    term_manager.prepare_terms({"HP:0001250"})

    mapping = term_manager.get_id_mapping()

    assert isinstance(mapping, dict)
    assert mapping["HP:0001250"] == "HP:0001250"


def test_prepare_terms_populates_label_cache(term_manager):
    """
    prepare_terms should populate the label cache for valid terms.
    """
    term_manager.prepare_terms({"HP:0001250"})

    labels = term_manager.get_labels()

    assert isinstance(labels, dict)
    assert "HP:0001250" in labels
    assert isinstance(labels["HP:0001250"], str)
    assert labels["HP:0001250"]


def test_prepare_terms_populates_ancestor_and_descendant_caches(term_manager):
    """
    prepare_terms should populate relationship caches.
    """
    term_manager.prepare_terms({"HP:0020219"})

    ancestors = term_manager.get_ancestors("HP:0020219")
    descendants = term_manager.get_descendants("HP:0020219")

    assert isinstance(ancestors, set)
    assert isinstance(descendants, set)


def test_get_ancestors_contains_expected_parent(term_manager):
    """
    HP:0001250 should be an ancestor of HP:0020219 in the test ontology.
    """
    term_manager.prepare_terms({"HP:0020219"})

    ancestors = term_manager.get_ancestors("HP:0020219")

    assert "HP:0001250" in ancestors


def test_get_descendants_contains_expected_child(term_manager):
    """
    HP:0020219 should be a descendant of HP:0001250 in the test ontology.
    """
    term_manager.prepare_terms({"HP:0001250"})

    descendants = term_manager.get_descendants("HP:0001250")

    assert "HP:0020219" in descendants


def test_get_ancestors_for_unprepared_term_returns_empty_set(term_manager):
    """
    Accessing relationships for an unprepared term should return an empty set.
    """
    ancestors = term_manager.get_ancestors("HP:0001250")
    assert ancestors == set()


def test_get_descendants_for_unprepared_term_returns_empty_set(term_manager):
    """
    Accessing relationships for an unprepared term should return an empty set.
    """
    descendants = term_manager.get_descendants("HP:0001250")
    assert descendants == set()


def test_get_labels_returns_copy(term_manager):
    """
    get_labels should return a copy, not the internal cache object.
    """
    term_manager.prepare_terms({"HP:0001250"})

    labels = term_manager.get_labels()
    labels["FAKE:0000001"] = "fake label"

    labels_again = term_manager.get_labels()
    assert "FAKE:0000001" not in labels_again


def test_get_id_mapping_returns_copy(term_manager):
    """
    get_id_mapping should return a copy, not the internal cache object.
    """
    term_manager.prepare_terms({"HP:0001250"})

    mapping = term_manager.get_id_mapping()
    mapping["FAKE:0000001"] = "HP:0001250"

    mapping_again = term_manager.get_id_mapping()
    assert "FAKE:0000001" not in mapping_again


def test_prepare_terms_accumulates_cache_across_calls(term_manager):
    """
    Repeated prepare_terms calls should accumulate cached entries.
    """
    term_manager.prepare_terms({"HP:0001250"})
    term_manager.prepare_terms({"HP:0020219"})

    labels = term_manager.get_labels()
    mapping = term_manager.get_id_mapping()

    assert "HP:0001250" in labels
    assert "HP:0020219" in labels
    assert "HP:0001250" in mapping
    assert "HP:0020219" in mapping