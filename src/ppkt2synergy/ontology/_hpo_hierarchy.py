from typing import IO
from collections.abc import Sequence
import logging

import numpy as np
import pandas as pd

from ._term_manager import HPOTermManager

logger = logging.getLogger(__name__)


class HPOHierarchyEngine:
    """
    Perform hierarchy-aware operations on HPO feature matrices.

    This class supports propagation of observed and excluded terms through
    the HPO hierarchy, as well as construction of relationship masks for
    pairwise analyses.

    .. note::

    Input matrices are expected to use:
    
        1 = observed
        0 = excluded
        NaN = unknown

    Invalid HPO terms are removed during preprocessing.

    If multiple input columns map to the same canonical HPO ID,
    they are merged using ``max()``, which prioritizes:

    - 1 over 0
    - 0 over NaN

    A warning is emitted if conflicting values (1 and 0) are found
    within duplicated columns for the same sample.
    """

    def __init__(
        self,
        hpo_file: str | IO | None = None,
        release: str | None = None,
    ) -> None:
        self._term_manager = HPOTermManager(hpo_file=hpo_file, release=release)
        
    @property
    def hpo(self):
        """Direct access to underlying HPO ontology (read-only)."""
        return self._term_manager.hpo

    def propagate(
        self, 
        matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Propagate HPO observations and exclusions through the ontology hierarchy.

        Observed terms (``1``) are propagated to ancestor terms, and excluded
        terms (``0``) are propagated to descendant terms.

        Parameters
        ----------
        matrix : pd.DataFrame
            HPO status matrix with individuals as rows and HPO terms as columns.

        Returns
        -------
        pd.DataFrame
            Matrix with propagated values and canonicalized HPO term IDs.
        """
        matrix = matrix.copy()

        original_terms = set(matrix.columns)
        valid_terms = self._term_manager.prepare_terms(original_terms)

        id_mapping = self._term_manager.get_id_mapping()
        matrix = matrix.rename(columns=id_mapping)

        matrix = matrix.loc[:, matrix.columns.isin(valid_terms)]
        matrix = matrix.T.groupby(level=0).max().T

        valid_terms = list(matrix.columns)
        valid_term_set = set(valid_terms)

        for term in valid_terms:
            ancestors = self._term_manager.get_ancestors(term) & valid_term_set
            descendants = self._term_manager.get_descendants(term) & valid_term_set

            observed_mask = matrix[term] == 1
            if observed_mask.any():
                for ancestor in ancestors:
                    conflict_mask = observed_mask & (matrix[ancestor] == 0)
                    if conflict_mask.any():
                        logger.warning(
                            "Conflict %d samples: %s=1 but ancestor %s=0",
                            int(conflict_mask.sum()),
                            term,
                            ancestor,
                        )
                    update_mask = observed_mask & matrix[ancestor].isna()
                    matrix.loc[update_mask, ancestor] = 1

            excluded_mask = matrix[term] == 0
            if excluded_mask.any():
                for descendant in descendants:
                    conflict_mask = excluded_mask & (matrix[descendant] == 1)
                    if conflict_mask.any():
                        logger.warning(
                            "Conflict %d samples: %s=0 but descendant %s=1",
                            int(conflict_mask.sum()),
                            term,
                            descendant,
                        )
                    update_mask = excluded_mask & matrix[descendant].isna()
                    matrix.loc[update_mask, descendant] = 0

        return matrix
    

    def build_relationship_mask(
        self, 
        terms: Sequence[str]
    ) -> pd.DataFrame:
        """
        Build a pairwise relationship mask for HPO terms.

        Parameters
        ----------
        terms : Sequence[str]
            Canonical HPO term IDs (typically obtained after propagation).
            The input order is preserved.

        Returns
        -------
        pd.DataFrame
            Square matrix indexed by HPO term IDs, where ``NaN`` indicates
            related terms (ancestor, descendant, or self) and ``0`` indicates
            unrelated terms.

        .. note::

        This mask can be used to exclude ontology-related term pairs from
        pairwise correlation or synergy analyses.
        """
        terms = list(terms)
        term_to_idx = {t: i for i, t in enumerate(terms)}
        N = len(terms)
        mask = np.zeros((N, N), dtype=float)

        for i, term in enumerate(terms):
            related = self._term_manager.get_ancestors(term) | self._term_manager.get_descendants(term)
            related_idx = [term_to_idx[t] for t in related if t in term_to_idx]
            mask[i, related_idx] = np.nan
            mask[related_idx, i] = np.nan
            mask[i, i] = np.nan

        mask_df = pd.DataFrame(mask, index=terms, columns=terms)
        return mask_df
    

    def get_labels(self) -> dict[str, str]:
        """Return cached HPO term labels."""
        return self._term_manager.get_labels()

    def get_id_mapping(self) -> dict[str, str]:
        """Return cached mapping from original to canonical HPO term IDs."""
        return self._term_manager.get_id_mapping()