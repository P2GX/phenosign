from typing import IO, Sequence
import logging

import numpy as np
import pandas as pd

from .term_manager import HPOTermManager

logger = logging.getLogger(__name__)


class HPOHierarchyEngine:
    """
    Perform hierarchy-aware operations on HPO matrices.

    This class supports:

    - propagation of observed terms (1) to ancestor terms
    - propagation of excluded terms (0) to descendant terms
    - construction of relationship masks for valid pairwise analysis

    Notes
    -----
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
        self.term_manager = HPOTermManager(hpo_file=hpo_file, release=release)

    def propagate(
        self, 
        matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Propagate HPO observations and exclusions through the ontology hierarchy.

        Rules
        -----
        - Observed (1) propagates upward to ancestors.
        - Excluded (0) propagates downward to descendants.

        Args:
            matrix : pd.DataFrame
                HPO status matrix with patients as rows and HPO terms as columns.

        Returns:
            pd.DataFrame
                Matrix with propagated values and canonicalized HPO term IDs.
        """
        matrix = matrix.copy()

        original_terms = set(matrix.columns)
        valid_terms = self.term_manager.prepare_terms(original_terms)

        id_mapping = self.term_manager.get_id_mapping()
        matrix = matrix.rename(columns=id_mapping)

        matrix = matrix.loc[:, matrix.columns.isin(valid_terms)]
        matrix = matrix.T.groupby(level=0).max().T

        valid_terms = list(matrix.columns)
        valid_term_set = set(valid_terms)

        for term in valid_terms:
            ancestors = self.term_manager.get_ancestors(term) & valid_term_set
            descendants = self.term_manager.get_descendants(term) & valid_term_set

            observed_mask = matrix[term] == 1
            if observed_mask.any():
                for ancestor in ancestors:
                    conflict_mask = observed_mask & (matrix[ancestor] == 0)
                    if conflict_mask.any():
                        logger.warning(
                            "[Conflict] %d samples: %s=1 but ancestor %s=0",
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
                            "[Conflict] %d samples: %s=0 but descendant %s=1",
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

        Args:
            terms : Sequence[str]
                HPO term IDs to include.

        Returns:
            pd.DataFrame
                Square matrix indexed by HPO term IDs where:
                - NaN means related (ancestor/descendant/self)
                - 0 means unrelated

        Notes
        -----
        This mask is useful for excluding ontology-related term pairs from
        pairwise correlation or synergy analyses.
        """
        terms = list(terms)
        self.term_manager.prepare_terms(set(terms))

        mask = pd.DataFrame(0.0, index=terms, columns=terms)

        term_set = set(terms)
        for term in terms:
            related = (
                self.term_manager.get_ancestors(term)
                | self.term_manager.get_descendants(term)
            ) & term_set

            for related_term in related:
                mask.loc[term, related_term] = np.nan
                mask.loc[related_term, term] = np.nan

            mask.loc[term, term] = np.nan

        return mask

    def get_labels(self) -> dict[str, str]:
        """Return cached HPO term labels."""
        return self.term_manager.get_labels()

    def get_id_mapping(self) -> dict[str, str]:
        """Return cached original-to-canonical ID mapping."""
        return self.term_manager.get_id_mapping()