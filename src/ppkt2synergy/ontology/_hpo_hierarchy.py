from typing import IO
from collections.abc import Sequence
from collections import defaultdict
import logging

import numpy as np
import pandas as pd

from ._term_manager import HPOTermManager

logger = logging.getLogger(__name__)


class HPOHierarchyEngine:
    """
    Perform hierarchy-aware operations on HPO feature matrices.

    This class supports:
    - Propagation of observed and excluded HPO terms through the ontology hierarchy.
    - Construction of pairwise relationship masks for downstream analyses.
    - Canonicalization of HPO term IDs and merging of duplicate columns.


    Input matrices are expected to use:
    
        1 = observed
        0 = excluded
        NaN = unknown

    Duplicate columns mapping to the same canonical HPO ID are merged using ``max()``, 
    prioritizing 1 > 0 > NaN. Conflicts (1 vs 0) generate warnings.

    Invalid HPO terms are removed during preprocessing.
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

        Observed terms (``1``) are propagated to ancestor terms, and excluded terms (``0``) 
        are propagated to descendant terms. Conflicts (1 vs 0) are logged as warnings
        but the original value is preserved.

        Parameters
        ----------
        matrix : pd.DataFrame
            HPO status matrix with individuals as rows and HPO terms as columns.
            Values should be 1 (observed), 0 (excluded), or NaN (unknown).


        Returns
        -------
        pd.DataFrame
            Matrix with propagated values and canonicalized HPO term IDs.
        """
        if matrix.empty:
            return matrix.copy()
        
        self._check_mapping_conflicts(matrix)

        all_present_terms = set(matrix.columns)
        valid_terms = self._term_manager.prepare_terms(all_present_terms)
        id_mapping = self._term_manager.get_id_mapping()

        matrix = matrix.rename(columns=id_mapping)
        matrix = matrix.T.groupby(level=0).max().T
        matrix = matrix.loc[:, matrix.columns.isin(valid_terms)]
        present_terms = set(matrix.columns)

        ancestors_lookup = {t: self._term_manager.get_ancestors(t) for t in present_terms}
        descendants_lookup = {t: self._term_manager.get_descendants(t) for t in present_terms}

        new_ones = {}
        new_zeros = {}

        for term in present_terms:
            if term not in matrix.columns:
                continue
                
            observed_mask = matrix[term] == 1
            if observed_mask.any():
                for ancestor in ancestors_lookup.get(term, []):
                    if ancestor == term:
                        continue
                    new_ones[ancestor] = new_ones.get(ancestor, observed_mask) | observed_mask

            excluded_mask = matrix[term] == 0
            if excluded_mask.any():
                for descendant in descendants_lookup.get(term, []):
                    if descendant == term:
                        continue
                    new_zeros[descendant] = new_zeros.get(descendant, excluded_mask) | excluded_mask

        for term in present_terms:
            mask_1 = new_ones.get(term, None)
            mask_0 = new_zeros.get(term, None)
            
            orig_1 = matrix[term] == 1
            orig_0 = matrix[term] == 0

            if mask_1 is not None:
                conflict_a = mask_1 & orig_0
                if conflict_a.any():
                    logger.warning(
                        "HPO propagation conflict: %d sample(s) have a descendant term propagating "
                        "OBSERVED (1) to '%s', but it is explicitly marked as EXCLUDED (0). Preserving 0.",
                        int(conflict_a.sum()), term
                    )
                update_mask_1 = mask_1 & matrix[term].isna()
                matrix.loc[update_mask_1, term] = 1

            if mask_0 is not None:
                conflict_b = mask_0 & orig_1
                if conflict_b.any():
                    logger.warning(
                        "HPO propagation conflict: %d sample(s) have an ancestor term propagating "
                        "EXCLUDED (0) to '%s', but it is explicitly marked as OBSERVED (1). Preserving 1.",
                        int(conflict_b.sum()), term
                    )
                update_mask_0 = mask_0 & matrix[term].isna()
                matrix.loc[update_mask_0, term] = 0

        return matrix
    

    def _check_mapping_conflicts(self, matrix: pd.DataFrame) -> None:
        """
        Check for many-to-one HPO term mapping conflicts in the matrix before merging.

        This method scans the input DataFrame prior to column renaming and grouping. 
        It identifies historical, obsolete, or alternative HPO terms that map to the 
        same canonical target term but possess conflicting truth values (both 0 and 1) 
        for the same individual sample.

        Parameters
        ----------
        matrix : pd.DataFrame
            HPO status matrix with individuals as rows and HPO terms as columns.
            Values must be 1 (observed), 0 (excluded), or NaN (unknown).

        Returns
        -------
        None
        """
        id_mapping = self._term_manager.get_id_mapping()
        
        target_to_sources = defaultdict(list)
        for source, target in id_mapping.items():
            if source in matrix.columns:
                target_to_sources[target].append(source)
                
        multi_mappings = {
            tgt: srcs for tgt, srcs in target_to_sources.items() if len(srcs) > 1
        }
        
        if not multi_mappings:
            return  

        for target_id, source_ids in multi_mappings.items():
            sub_df = matrix[source_ids]
            
            has_one = (sub_df == 1).any(axis=1)
            has_zero = (sub_df == 0).any(axis=1)
            conflict_mask = has_one & has_zero
            
            if conflict_mask.any():
                conflict_samples = matrix.index[conflict_mask].tolist()
                
                logger.warning(
                    "Pre-merging conflict detected! The HPO terms %s all map to '%s'. "
                    "However, for sample(s) %s, these terms have conflicting values (both 0 and 1). "
                    "Subsequent grouping will silently privilege OBSERVED (1).",
                    source_ids,
                    target_id,
                    conflict_samples if len(conflict_samples) <= 5 else f"{conflict_samples[:5]}... (Total: {len(conflict_samples)})"
                )
    

    def build_relationship_mask(
        self, 
        terms: Sequence[str]
    ) -> pd.DataFrame:
        """
        Build a pairwise relationship mask for HPO terms.

        Related terms (ancestor, descendant, or self) are marked with NaN,
        while unrelated terms are marked with 0. This mask can be used to
        exclude ontology-related term pairs in correlation or synergy analyses.

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