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

    Properties
    ----------
    feature_names : pd.Index
        HPO term IDs (matrix columns).
    sample_ids : pd.Index
        Individual identifiers (matrix index).
    """

    matrix: pd.DataFrame
    label_mapping: dict[str, str] = field(default_factory=dict)
    relationship_mask: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self._validate_matrix()
        self._validate_label_mapping()
        self._validate_relationship_mask()


    def _validate_matrix(self) -> None:
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
        
    def _validate_label_mapping(self) -> None:
        if not isinstance(self.label_mapping, dict):
            raise TypeError(
                f"`label_mapping` must be a dict[str, str], got {type(self.label_mapping).__name__}."
            )

        if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in self.label_mapping.items()
        ):
            raise TypeError("`label_mapping` must map strings to strings.")
        
    def _validate_relationship_mask(self) -> None:
        if self.relationship_mask is None:
            return

        mask = self.relationship_mask

        if not isinstance(mask, pd.DataFrame):
            raise TypeError(
                f"`relationship_mask` must be a pandas DataFrame, got {type(mask).__name__}."
            )

        if set(mask.index) != set(self.matrix.columns) or set(mask.columns) != set(self.matrix.columns):
            raise ValueError("`relationship_mask` must contain the same HPO terms as matrix columns.")

        self.relationship_mask = mask.loc[self.matrix.columns, self.matrix.columns]

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
    def feature_ids(self) -> pd.Index:
        """HPO feature IDs (matrix columns)."""
        return self.matrix.columns

    @property
    def individual_ids(self) -> pd.Index:
        """Individual IDs (matrix index)."""
        return self.matrix.index