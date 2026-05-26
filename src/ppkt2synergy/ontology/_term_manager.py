from typing import IO
import logging

from ..io.hpo_loader import _load_hpo

logger = logging.getLogger(__name__)


class HPOTermManager:
    """
    Manage HPO term normalization and cached ontology relationships.

    This class provides a lightweight interface to the loaded HPO ontology.
    It resolves term IDs to canonical identifiers and caches ancestors,
    descendants, labels, and original-to-canonical ID mappings.

    Notes
    -----
    Invalid HPO terms are skipped during bulk preparation with a warning.
    Individual term resolution via ``resolve_term_id`` raises ``ValueError``
    if the term is not found in the ontology.
    """

    def __init__(
        self,
        hpo_file: str | IO | None = None,
        release: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        hpo_file : str | IO | None, optional
            Path, URL, or file-like object pointing to an HPO ontology file.
        release : str | None, optional
            HPO release identifier. If ``None``, the latest available
            release is used.
        """
        self._hpo = _load_hpo(file=hpo_file, release=release)

        self._ancestor_cache: dict[str, set[str]] = {}
        self._descendant_cache: dict[str, set[str]] = {}
        self._label_cache: dict[str, str] = {}
        self._id_mapping_cache: dict[str, str] = {}

    @property
    def hpo(self):
        """
        Return the underlying HPO ontology object (read-only).

        Advanced/internal use only.
        """
        return self._hpo

    def resolve_term_id(
        self, 
        term_id: str
    ) -> str:
        """
        Resolve an HPO term ID to its canonical primary ID.

        Parameters
        ----------
        term_id : str
            HPO term ID, for example ``"HP:0001250"``.

        Returns
        -------
        str
            Canonical HPO term ID.

        Raises
        ------
        ValueError
            If the term ID is not found in the ontology.
        """
        term = self._hpo.get_term(term_id=term_id)
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

        Parameters
        ----------
        terms : set[str]
            Input HPO term IDs.

        Returns
        -------
        set[str]
            Valid canonical HPO term IDs.
        """
        resolved_terms: set[str] = set()

        for term in terms:
            try:
                primary_id = self.resolve_term_id(term)
                if primary_id != term:
                    logger.info("Resolved term ID %r to primary ID %r", term, primary_id)

                resolved_terms.add(primary_id)
                self._id_mapping_cache[term] = primary_id

                if primary_id not in self._label_cache:
                    self._label_cache[primary_id] = self._hpo.get_term(term_id=primary_id).name

            except ValueError as exc:
                logger.warning("%s Skipping term %r", exc, term)

        new_terms = resolved_terms - self._ancestor_cache.keys()

        for term in new_terms:
            try:
                self._ancestor_cache[term] = {
                    ancestor.value for ancestor in self._hpo.graph.get_ancestors(term)
                }
            except Exception as exc:
                logger.warning("Failed to fetch ancestors for %r: %s", term, exc)
                self._ancestor_cache[term] = set()

            try:
                self._descendant_cache[term] = {
                    descendant.value for descendant in self._hpo.graph.get_descendants(term)
                }
            except Exception as exc:
                logger.warning("Failed to fetch descendants for %r: %s", term, exc)
                self._descendant_cache[term] = set()

        return resolved_terms

    def get_ancestors(self, term: str) -> set[str]:
        """Return cached ancestor terms for a canonical HPO term ID."""
        return self._ancestor_cache.get(term, set())

    def get_descendants(self, term: str) -> set[str]:
        """Return cached descendant terms for a canonical HPO term ID."""
        return self._descendant_cache.get(term, set())

    def get_labels(self) -> dict[str, str]:
        """Return cached labels for canonical HPO term IDs."""
        return dict(self._label_cache)

    def get_id_mapping(self) -> dict[str, str]:
        """Return cached mappings from original to canonical HPO term IDs."""
        return dict(self._id_mapping_cache)