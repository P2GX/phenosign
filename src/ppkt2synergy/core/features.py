from dataclasses import dataclass, field
import pandas as pd

@dataclass
class HpoFeatureData:
    """
    Container for individual-level HPO feature data.

    Parameters
    ----------
    matrix : pd.DataFrame
        Binary HPO feature matrix with individuals as rows and HPO terms
        as columns. Values must be ``1`` (observed), ``0`` (excluded),
        or ``NaN`` (unknown).
    label_mapping : dict[str, str], optional
        Mapping from HPO term IDs to human-readable labels.
    relationship_mask : pd.DataFrame | None, optional
        Square mask over HPO terms with the same index and columns as
        ``matrix.columns``. Related term pairs are encoded as ``NaN``
        and unrelated pairs as ``0``.
    """
    matrix: pd.DataFrame
    label_mapping: dict[str, str] = field(default_factory=dict)
    relationship_mask: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, pd.DataFrame):
            raise TypeError(f"`matrix` must be a pandas DataFrame, got {type(self.matrix).__name__}.")
        
        if self.matrix.shape[0] == 0 or self.matrix.shape[1] == 0:
            raise ValueError("`matrix` must not be empty.")

        if self.matrix.columns.has_duplicates:
            duplicated_columns = (
                self.matrix.columns[self.matrix.columns.duplicated()]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"`matrix` columns must be unique. Duplicated columns: {duplicated_columns}."
            )

        if self.matrix.index.has_duplicates:
            duplicated_index = (
                self.matrix.index[self.matrix.index.duplicated()]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"`matrix` index must be unique. Duplicated index values: {duplicated_index}."
            )
        
        invalid = ~self.matrix.isin([0, 1]) & self.matrix.notna()
        if invalid.any().any():
            bad_vals = [
                v for v in pd.unique(self.matrix[invalid].values.ravel())
                if pd.notna(v)
            ]
            raise ValueError(
                "`matrix` must contain only 0, 1, or NaN. "
                f"Found invalid values: {bad_vals[:10]}."
            )
        

        if not isinstance(self.label_mapping, dict):
            raise TypeError(
                f"`label_mapping` must be a dict[str, str], got {type(self.label_mapping).__name__}."
            )

        if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in self.label_mapping.items()
        ):
            raise TypeError("`label_mapping` must map strings to strings.")
        

        if self.relationship_mask is None:
            return

        mask = self.relationship_mask

        if not isinstance(mask, pd.DataFrame):
            raise TypeError(
                f"`relationship_mask` must be a pandas DataFrame, got {type(mask).__name__}."
            )

        if not (
            mask.index.equals(self.matrix.columns)
            and mask.columns.equals(self.matrix.columns)
        ):
            raise ValueError(
                "`relationship_mask` must have the same index and columns as "
                "`matrix.columns`, in the same order."
            )

        invalid_mask = ~mask.isin([0]) & mask.notna()
        if invalid_mask.any().any():
            bad_vals = [
                v for v in pd.unique(mask[invalid_mask].values.ravel())
                if pd.notna(v)
            ]
            raise ValueError(
                "`relationship_mask` must contain only 0 or NaN. "
                f"Found invalid values: {bad_vals[:10]}."
            )

        if not mask.equals(mask.T):
            raise ValueError("`relationship_mask` must be symmetric.")
        
    @property
    def feature_names(self) -> pd.Index:
        """Return HPO feature names (matrix columns)."""
        return self.matrix.columns

    @property
    def sample_ids(self) -> pd.Index:
        """Return sample IDs (matrix index)."""
        return self.matrix.index