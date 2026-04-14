import logging
import pandas as pd

from ..core import HpoFeatureData
from ..core import PhenopacketRecord
from ..ontology import HPOHierarchyEngine

logger = logging.getLogger(__name__)

class HpoFeatureBuilder:
    """
    Build HPO feature data from standardized patient records.

    This builder is responsible for:

    - constructing the raw HPO status matrix
    - propagating HPO observations/exclusions through the ontology hierarchy
    - filtering terms by missingness threshold
    - building the HPO relationship mask
    - collecting HPO label mappings

    Notes
    -----
    The HPO matrix uses a 3-state representation:

    - 1   = observed
    - 0   = explicitly excluded
    - NaN = unknown
    """
    def __init__(
        self,
        records: list[PhenopacketRecord],
        hpo_hierarchy: HPOHierarchyEngine
    ):
        if not records:
            raise ValueError("records cannot be empty")

        self.records = records
        self.hpo_engine = hpo_hierarchy

        self.patient_index = pd.Index(
            [record.patient_id for record in self.records],
            name="patient_id",
        )

        if self.patient_index.has_duplicates:
            raise ValueError("Patient IDs must be unique in records")

    # ------------------------------------------------------------------
    # Raw HPO matrix
    # ------------------------------------------------------------------

    def build_raw_matrix(
        self
    ) -> pd.DataFrame:
        """
        Build raw HPO status matrix from patient records.

        Returns:
            pd.DataFrame
                Rows = patients
                Columns = HPO terms
                Values:
                    1   = observed
                    0   = explicitly excluded
                    NaN = unknown
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
                    "Patient %s has conflicting HPO annotations: %s",
                    record.patient_id,
                    sorted(conflict),
                )

            row = {term: float("nan") for term in all_terms}

            for term in record.observed_hpo_terms:
                row[term] = 1.0

            for term in record.excluded_hpo_terms:
                row[term] = 0.0

            data[record.patient_id] = row

        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.reindex(self.patient_index)

        return df

    # ------------------------------------------------------------------
    # Missingness filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_missingness(
        matrix: pd.DataFrame,
        missing_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Filter HPO terms by allowed missing-value proportion.

        Args:
            matrix : pd.DataFrame
                HPO status matrix.
            missing_threshold : float, default=1.0
                Maximum allowed proportion of missing values per column.
                Must satisfy 0 <= missing_threshold <= 1.

                Examples:
                - 1.0 keeps all columns
                - 0.9 removes columns with >90% missing values
                - 0.0 keeps only columns with no missing values

        Returns:
            pd.DataFrame
                Filtered matrix.

        Raises:
            ValueError
                If threshold is outside [0, 1].
        """
        if not 0.0 <= missing_threshold <= 1.0:
            raise ValueError("missing_threshold must be between 0 and 1")

        min_non_missing = int((1 - missing_threshold) * len(matrix))
        filtered = matrix.dropna(axis=1, thresh=min_non_missing)

        return filtered

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------

    def build(
        self,
        missing_threshold: float = 1.0,
    ) -> HpoFeatureData:
        """
        Build final HPO feature data.

        Parameters:
            missing_threshold : float, default=1.0
                Maximum allowed proportion of missing values per HPO term.

        Returns:
            HpoFeatureData
                Final HPO feature container.

        Raises:
            ValueError
                If no valid HPO terms remain after filtering.
        """
        raw_matrix = self.build_raw_matrix()

        propagated_matrix = self.hpo_engine.propagate(raw_matrix)
        filtered_matrix = self.filter_by_missingness(
            propagated_matrix,
            missing_threshold=missing_threshold,
        )

        if filtered_matrix.empty:
            raise ValueError(
                "No valid HPO terms remain after filtering. "
                "Please adjust the missing_threshold."
            )

        relationship_mask = self.hpo_engine.build_relationship_mask(
            filtered_matrix.columns
        )
        label_mapping = self.hpo_engine.get_labels()

        return HpoFeatureData(
            matrix=filtered_matrix,
            label_mapping=label_mapping,
            relationship_mask=relationship_mask,
        )