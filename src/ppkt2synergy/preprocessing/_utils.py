from typing import Dict, Set, Union, IO, Sequence
from ..io import load_hpo
import pandas as pd
import numpy as np
import logging
logger= logging.getLogger(__name__)

class HPOTermManager:
    """
    Manage HPO terms and cache their hierarchical relationships.

    This class provides utilities for:
    - Resolving HPO term IDs to their canonical (primary) IDs
    - Caching ancestor and descendant relationships
    - Avoiding repeated ontology graph queries
    """

    def __init__(
            self, 
            hpo_file: str| IO | None = None, 
            release: str | None = None
        ):
        """
        Initialize the HPO term manager.

        Args:
        hpo_file : str or IO, optional
            Path or file-like object for a local HPO ontology file.
        release : str, optional
            Specific HPO release version to load.
            If None, the latest release is used.
        """
        self.hpo = load_hpo(file = hpo_file, release = release)

        # Cache for ontology relationships
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self._descendant_cache: Dict[str, Set[str]] = {}
        self._label_cache: Dict[str, str] = {}  # Cache for resolved term IDs
        self._id_mapping_cache: Dict[str, str] = {}  # Cache for term ID resolutions

    
    def _resolve_term_id(
            self, 
            term_id: str
        ) -> str:
        """
        Resolve an HPO term ID to its canonical primary ID.

        Args:
            term_id : str
                HPO term ID (e.g., "HP:0001250").

        Returns:
            str:
                Canonical HPO term ID.
        """
        term = self.hpo.get_term(term_id=term_id)
        if not term:
            raise ValueError(
                f"Term ID '{term_id}' not found in HPO ontology."
            )
        
        return term.identifier.value
        
    
    def prepare_terms(
            self, 
            terms: Set[str]
        ) -> Set[str]:
        """
        Resolve, validate, and cache HPO terms.
        Invalid terms are skipped with a warning.

        Args:
            terms: set[str]
                Input HPO term IDs.

        Returns:
            set[str]:
                Valid canonical HPO term IDs.
        """
        resolved_terms = set()
        for term in terms:
            try:
                primary_id = self._resolve_term_id(term)
                if primary_id != term:
                    logger.info(f"Resolved term ID '{term}' to primary ID '{primary_id}'")
                resolved_terms.add(primary_id)
                self._id_mapping_cache[term] = primary_id
                if primary_id not in self._label_cache:
                    self._label_cache[primary_id] = self.hpo.get_term(term_id=primary_id).name
            except ValueError as e:
                logger.warning(f"Warning: {e} — skipping term '{term}'")

        new_terms = resolved_terms - self._ancestor_cache.keys()
        for term in new_terms:
            try:
                self._ancestor_cache[term] = {a.value for a in self.hpo.graph.get_ancestors(term)}
            except Exception as e:
                logger.warning(f"Warning: {e}")
                self._ancestor_cache[term] = set()
            
            try:
                self._descendant_cache[term] = {d.value for d in self.hpo.graph.get_descendants(term)}
            except Exception as e:
                logger.warning(f"Warning: {e}")
                self._descendant_cache[term] = set()
        return resolved_terms

    def get_ancestors(self, term: str) -> Set[str]:
        """Return cached ancestors of a term."""
        return self._ancestor_cache.get(term, set())

    def get_descendants(self, term: str) -> Set[str]:
        """Return cached descendants of a term."""
        return self._descendant_cache.get(term, set())
    
    def get_labels(self) -> Dict[str, str]:
        """Return cached labels."""
        return self._label_cache
    
    def get_id_mapping(self) -> Dict[str, str]:
        """Return cached term ID mappings."""
        return self._id_mapping_cache

class HPOHierarchyEngine:
    """
    Provides hierarchical operations on HPO term matrices using an HPOTermManager.

    This class offers:
    - Propagation of observed/excluded HPO term values along the ontology hierarchy.
      * Observed (1) values propagate to all ancestor terms.
      * Excluded (0) values propagate to all descendant terms.
    - Construction of hierarchical relationship masks between HPO terms.
    
    It ensures that only valid HPO terms (present in the ontology) are considered.
    Invalid terms are automatically removed during processing.
    """
    def __init__(self, 
                 hpo_file: str | IO | None = None, 
                 release: str | None = None):
        self.tm = HPOTermManager(hpo_file=hpo_file, release=release)

    def propagate_hpo_hierarchy(
            self, 
            matrix: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Applies hierarchical propagation to an HPO term matrix.

        Propagation logic:
        - A value of 1 (observed) propagates to all ancestors of the corresponding HPO term.
        - A value of 0 (excluded) propagates to all descendants.

        If a term is not found in the ontology, it is removed from the matrix.

        Example (before propagation):
            HP:0012759 (Neurological abnormality)
            ├── HP:0001250 (Seizures)
            │   └── HP:0004322 (Focal seizures)

            Matrix:
                         HP:0004322  HP:0001250  HP:0012759
            Patient_1         1         NaN         NaN
            Patient_2         NaN        0          NaN

        Example (after propagation):
                         HP:0004322  HP:0001250  HP:0012759
            Patient_1         1         1          1
            Patient_2         0         0          NaN

        Args:
            matrix (pd.DataFrame): 
                Input matrix with raw HPO observations/exclusions..

        Returns:
            pd.DataFrame: 
                Matrix with propagated values.
        """
        matrix = matrix.copy()

        terms_old = set(matrix.columns)
        valid_terms = self.tm.prepare_terms(terms_old)

        id_mapping = self.tm.get_id_mapping()
        matrix = matrix.rename(columns=id_mapping)

        matrix = matrix.loc[:, matrix.columns.isin(valid_terms)] # delete invalid terms if not in ontology
        matrix = matrix.T.groupby(level=0).max().T # delete duplicate columns if multiple input terms map to same primary term

        valid_terms = list(matrix.columns)

        for term in valid_terms:
            # ---- Propagate observed (1) values upwards ----
            ancestors = self.tm.get_ancestors(term)
            valid_ancestors = ancestors & set(valid_terms)
            if valid_ancestors:
                mask = matrix[term] == 1
                for anc in valid_ancestors:
                    conflict_mask = mask & (matrix[anc] == 0)
                    if conflict_mask.any():
                        logger.warning(
                            f"[Conflict] {conflict_mask.sum()} samples: {term}=1 but {anc}=0"
                        )
                    # Only assign where target is NaN
                    update_mask = mask & (matrix[anc].isna())
                    matrix.loc[update_mask, anc] = 1

            # ---- Propagate excluded (0) values downwards ----
            descendants = self.tm.get_descendants(term)
            valid_descendants = descendants & set(valid_terms)
            if valid_descendants:
                mask = matrix[term] == 0
                for desc in valid_descendants:
                    conflict_mask = mask & (matrix[desc] == 1)
                    if conflict_mask.any():
                        logger.warning(
                            f"[Conflict] {conflict_mask.sum()} samples: {term}=0 but {desc}=1"
                        )
                    # Only assign where target is NaN
                    update_mask = mask & (matrix[desc].isna())
                    matrix.loc[update_mask, desc] = 0
        return matrix    
    
    def build_relationship_mask(
            self, 
            terms: Union[pd.Index, Sequence[str]]
        ) -> pd.DataFrame:
        """
        Build a hierarchical relationship mask for a set of HPO terms.

        The returned matrix has:
        - NaN where terms are related (ancestor/descendant relationship, including self-relations).
        - 0 where terms are unrelated.

        Args:
            terms (pd.Index or Sequence[str]):
                List of HPO term IDs to include in the mask.

        Returns:
            pd.DataFrame:
                Square matrix (terms x terms) with NaN for related terms
                and 0 for unrelated terms.

        Example:
            Terms: [HP:0001250, HP:0004322, HP:0012759]

            Mask:
                           HP:0001250  HP:0004322  HP:0012759
            HP:0001250        NaN         NaN          NaN
            HP:0004322        NaN         NaN          NaN
            HP:0012759        NaN         NaN          NaN
        """
        terms = list(terms)
        # Initialize a matrix filled with 0s; it will be updated to 1 or NaN later.
        mask = pd.DataFrame(0, index=terms, columns=terms, dtype=float)
        self.tm.prepare_terms(set(terms))  # Ensure terms are valid and cache relationships
        for term in terms:
            if term not in terms:
                continue
            related = (self.tm.get_ancestors(term) | self.tm.get_descendants(term)) & set(terms)
            for rel in related:
                mask.loc[term, rel] = np.nan
                mask.loc[rel, term] = np.nan
            mask.loc[term, term] = np.nan

        return mask
    
    def get_labels(self) -> Dict[str, str]:
        """Return cached labels."""
        return self.tm.get_labels()
