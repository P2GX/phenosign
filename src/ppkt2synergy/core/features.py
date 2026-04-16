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
        
        # Validate matrix values: only 0, 1, NaN
        invalid = ~self.matrix.isin([0, 1]) & self.matrix.notna()
        if invalid.any().any():
            bad_vals = pd.unique(self.matrix[invalid].values.ravel())
            raise ValueError(
                f"matrix must contain only 0, 1, or NaN. Found: {bad_vals.tolist()}"
            )

        # --- relationship_mask checks ---
        if self.relationship_mask is None:
            return

        mask = self.relationship_mask

        if not isinstance(mask, pd.DataFrame):
            raise TypeError("relationship_mask must be a pandas DataFrame")

        if mask.shape[0] != mask.shape[1]:
            raise ValueError("relationship_mask must be a square matrix")

        if mask.index.has_duplicates or mask.columns.has_duplicates:
            raise ValueError("relationship_mask index/columns must be unique")

        # --- index/column match ---
        if not (mask.index.equals(self.matrix.columns) and
                mask.columns.equals(self.matrix.columns)):
            raise ValueError("relationship_mask must match matrix columns exactly")

        # --- mask value check (0 / NaN) ---
        invalid_mask = ~mask.isin([0]) & mask.notna()
        if invalid_mask.any().any():
            bad_vals = pd.unique(mask[invalid_mask].values.ravel())
            raise ValueError(
                f"relationship_mask must contain only 0 or NaN. Found: {bad_vals.tolist()}"
            )

    @property
    def feature_names(self) -> pd.Index:
        """Return HPO feature names (matrix columns)."""
        return self.matrix.columns

    @property
    def sample_ids(self) -> pd.Index:
        """Return sample IDs (matrix index)."""
        return self.matrix.index