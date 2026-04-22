import logging
import pandas as pd

from ..core import HpoFeatureData
from ..core import PhenopacketRecord
from ..ontology import HPOHierarchyEngine

logger = logging.getLogger(__name__)

class HpoFeatureBuilder:
    """
    Build HPO feature data from standardized individual records.

    This builder constructs the raw HPO matrix, propagates observed and
    excluded terms through the ontology hierarchy, filters terms by
    missingness, and generates label mappings and relationship masks.

    Notes
    -----
    The HPO matrix uses a three-state representation:

    - ``1`` for observed terms
    - ``0`` for explicitly excluded terms
    - ``NaN`` for unknown terms
    """
    def __init__(
        self,
        records: list[PhenopacketRecord],
        hpo_hierarchy: HPOHierarchyEngine
    ):
        """
        Parameters
        ----------
        records : list[PhenopacketRecord]
            Standardized individual records.
        hpo_hierarchy : HPOHierarchyEngine
            Hierarchy engine used for HPO propagation and mask construction.
        """
        if not records:
            raise ValueError("records cannot be empty")

        self.records = records
        self.hpo_engine = hpo_hierarchy

        self.individual_index = pd.Index(
            [record.individual_id for record in self.records],
            name="individual_id",
        )


    def build_raw_matrix(
        self
    ) -> pd.DataFrame:
        """
        Build the raw HPO status matrix from individual records.

        Returns
        -------
        pd.DataFrame
            HPO status matrix with individuals as rows and HPO terms as
            columns. Values are ``1`` for observed terms, ``0`` for excluded
            terms, and ``NaN`` for unknown terms.
        """
        all_terms = set()

        for record in self.records:
            all_terms.update(record.observed_hpo_terms)
            all_terms.update(record.excluded_hpo_terms)

        all_terms = sorted(all_terms)
        data: dict[str, dict[str, float]] = {}

        for record in self.records:
            conflict = record.observed_hpo_terms & record.excluded_hpo_terms
            if conflict:
                logger.warning(
                    "Individual %s has conflicting HPO annotations: %s",
                    record.individual_id,
                    sorted(conflict),
                )

            row = {term: float("nan") for term in all_terms}

            for term in record.observed_hpo_terms:
                row[term] = 1.0

            for term in record.excluded_hpo_terms:
                row[term] = 0.0

            data[record.individual_id] = row

        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.reindex(self.individual_index)

        return df
    

    @staticmethod
    def filter_by_missingness(
        matrix: pd.DataFrame,
        missing_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Filter HPO terms by the allowed proportion of missing values.

        Parameters
        ----------
        matrix : pd.DataFrame
            HPO status matrix.
        missing_threshold : float, default=1.0
            Maximum allowed proportion of missing values per column.
            Must satisfy ``0 <= missing_threshold <= 1``.

        Returns
        -------
        pd.DataFrame
            Filtered HPO matrix.

        Raises
        ------
        ValueError
            If ``missing_threshold`` is outside ``[0, 1]``.
        """
        if not 0.0 <= missing_threshold <= 1.0:
            raise ValueError("missing_threshold must be between 0 and 1")

        min_non_missing = int((1 - missing_threshold) * len(matrix))
        filtered = matrix.dropna(axis=1, thresh=min_non_missing)

        return filtered


    def build(
        self,
        missing_threshold: float = 1.0,
    ) -> HpoFeatureData:
        """
        Build HPO feature data from the input records.

        Parameters
        ----------
        missing_threshold : float, default=1.0
            Maximum allowed proportion of missing values per HPO term.

        Returns
        -------
        HpoFeatureData
            Final HPO feature container.

        Raises
        ------
        ValueError
            If no valid HPO terms remain after filtering.
        """
        raw_matrix = self.build_raw_matrix()
        if raw_matrix.shape[1] == 0:
            raise ValueError(
                "No HPO terms found in input records. "
                "Ensure records contain observed or excluded HPO terms."
            )
        propagated_matrix = self.hpo_engine.propagate(raw_matrix)
        filtered_matrix = self.filter_by_missingness(
            propagated_matrix,
            missing_threshold=missing_threshold,
        )

        if filtered_matrix.empty:
            raise ValueError(
                "No HPO terms remain after propagation and missingness filtering. "
                f"(missing_threshold={missing_threshold}, "
                f"n_raw_terms={raw_matrix.shape[1]}, "
                f"n_after_propagation={propagated_matrix.shape[1]}). "
                "Possible causes include invalid terms removed during preprocessing "
                "or overly strict filtering."
            )
        
        label_mapping = self.hpo_engine.get_labels()
        relationship_mask = self.hpo_engine.build_relationship_mask(
            filtered_matrix.columns
        )

        return HpoFeatureData(
            matrix=filtered_matrix,
            label_mapping=label_mapping,
            relationship_mask=relationship_mask,
        )