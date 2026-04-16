from dataclasses import dataclass, field
import pandas as pd

@dataclass
class HpoFeatureData:
    """
    Container for patient HPO feature data used in downstream analysis.

    Attributes
    ----------
    matrix : pd.DataFrame
        Binary matrix of HPO term observations
        (rows=patients, columns=HPO terms).
        Values: 
        - 1=observed
        - 0=excluded
        - NaN=unknown.

    label_mapping : dict[str, str]
        Mapping from HPO term IDs to human-readable labels.

    relationship_mask : pd.DataFrame | None, optional
         Square matrix describing hierarchical relationships between HPO terms
        (terms x terms):
        - NaN = related (ancestor/descendant/self)
        - 0 = unrelated
    """
    matrix: pd.DataFrame
    label_mapping: dict[str, str] = field(default_factory=dict)
    relationship_mask: pd.DataFrame | None = None

    def __post_init__(self):
        if not isinstance(self.matrix, pd.DataFrame):
            raise TypeError("matrix must be a pandas DataFrame")

        if self.matrix.columns.has_duplicates:
            raise ValueError("matrix columns must be unique")

        if self.matrix.index.has_duplicates:
            raise ValueError("matrix index must be unique")

        if self.relationship_mask is not None and not self.relationship_mask.empty:
            terms_set = set(self.matrix.columns)
            mask_rows_set = set(self.relationship_mask.index)
            mask_cols_set = set(self.relationship_mask.columns)

            missing_rows = terms_set - mask_rows_set
            extra_rows = mask_rows_set - terms_set
            missing_cols = terms_set - mask_cols_set
            extra_cols = mask_cols_set - terms_set

            if missing_rows or extra_rows or missing_cols or extra_cols:
                raise ValueError(
                    "relationship_mask rows/columns do not match matrix columns.\n"
                    f"Missing rows: {missing_rows}\n"
                    f"Extra rows: {extra_rows}\n"
                    f"Missing cols: {missing_cols}\n"
                    f"Extra cols: {extra_cols}"
                )

    @property
    def feature_names(self) -> pd.Index:
        """Return HPO feature names (matrix columns)."""
        return self.matrix.columns

    @property
    def sample_ids(self) -> pd.Index:
        """Return sample IDs (matrix index)."""
        return self.matrix.index