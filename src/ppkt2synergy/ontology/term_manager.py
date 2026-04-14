from typing import IO
import logging

from ..io import load_hpo

logger = logging.getLogger(__name__)


class HPOTermManager:
    """
    Manage HPO terms and cache ontology relationships.

    This class provides a lightweight interface around the loaded HPO ontology.
    It is responsible for:

    - resolving term IDs to canonical primary IDs
    - caching ancestor relationships
    - caching descendant relationships
    - caching term labels

    Notes
    -----
    - Invalid HPO terms are skipped during bulk preparation with a warning.
    - Individual resolution via `resolve_term_id` raises ValueError if the term
      is not found in the ontology.
    """

    def __init__(
        self,
        hpo_file: str | IO | None = None,
        release: str | None = None,
    ) -> None:
        """
        Args:
            hpo_file : str | IO | None, optional
                Path or file-like object for a local HPO ontology file.
            release : str | None, optional
                Specific HPO release version to load. If None, the latest
                available release is used.
        """
        self.hpo = load_hpo(file=hpo_file, release=release)

        self._ancestor_cache: dict[str, set[str]] = {}
        self._descendant_cache: dict[str, set[str]] = {}
        self._label_cache: dict[str, str] = {}
        self._id_mapping_cache: dict[str, str] = {}

    def resolve_term_id(
        self, 
        term_id: str
    ) -> str:
        """
        Resolve an HPO term ID to its canonical primary ID.

        Args:
            term_id : str
                HPO term ID, e.g. "HP:0001250".

        Returns:
            str
                Canonical HPO term ID.
        Raises:
            ValueError
                If the term ID is not found in the ontology.
        """
        term = self.hpo.get_term(term_id=term_id)
        if term is None:
            raise ValueError(f"Term ID '{term_id}' not found in HPO ontology.")
        return term.identifier.value

    def prepare_terms(
        self, 
        terms: set[str]
    ) -> set[str]:
        """
        Resolve, validate, and cache a set of HPO terms.

        Invalid terms are skipped with a warning.

        Args:
            terms : set[str]
                Input HPO term IDs.

        Returns:
            set[str]
                Valid canonical HPO term IDs.
        """
        resolved_terms: set[str] = set()

        for term in terms:
            try:
                primary_id = self.resolve_term_id(term)
                if primary_id != term:
                    logger.info("Resolved term ID '%s' to primary ID '%s'", term, primary_id)

                resolved_terms.add(primary_id)
                self._id_mapping_cache[term] = primary_id

                if primary_id not in self._label_cache:
                    self._label_cache[primary_id] = self.hpo.get_term(term_id=primary_id).name

            except ValueError as exc:
                logger.warning("%s - skipping term '%s'", exc, term)

        new_terms = resolved_terms - self._ancestor_cache.keys()

        for term in new_terms:
            try:
                self._ancestor_cache[term] = {
                    ancestor.value for ancestor in self.hpo.graph.get_ancestors(term)
                }
            except Exception as exc:
                logger.warning("Failed to fetch ancestors for '%s': %s", term, exc)
                self._ancestor_cache[term] = set()

            try:
                self._descendant_cache[term] = {
                    descendant.value for descendant in self.hpo.graph.get_descendants(term)
                }
            except Exception as exc:
                logger.warning("Failed to fetch descendants for '%s': %s", term, exc)
                self._descendant_cache[term] = set()

        return resolved_terms

    def get_ancestors(self, term: str) -> set[str]:
        """Return cached ancestors of a term."""
        return self._ancestor_cache.get(term, set())

    def get_descendants(self, term: str) -> set[str]:
        """Return cached descendants of a term."""
        return self._descendant_cache.get(term, set())

    def get_labels(self) -> dict[str, str]:
        """Return cached HPO term labels."""
        return dict(self._label_cache)

    def get_id_mapping(self) -> dict[str, str]:
        """Return cached original-to-canonical term ID mappings."""
        return dict(self._id_mapping_cache)