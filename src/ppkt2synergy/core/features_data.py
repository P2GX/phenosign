from dataclasses import dataclass, field

import pandas as pd
import numpy as np

@dataclass
class HpoFeatureData:
    """
    Container for individual-level HPO feature data.

    Stores a binary HPO feature matrix together with optional
    term labels and ontology relationship masks.

    Matrix values follow the convention:

        1 = observed
        0 = excluded
        NaN = unknown
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
        
        matrix_values = self.matrix.values
        invalid = (matrix_values != 0) & (matrix_values != 1) & (~np.isnan(matrix_values))
        if invalid.any():
            bad_vals = [
                v for v in np.unique(matrix_values[invalid]) if not np.isnan(v)
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

        mask = mask.loc[self.matrix.columns, self.matrix.columns]
        self.relationship_mask = mask

        mask_values = mask.values
        invalid_mask = (mask_values != 0) & (~np.isnan(mask_values))
        if invalid_mask.any():
            bad_vals = [
                v for v in np.unique(mask_values[invalid_mask]) if not np.isnan(v)
            ]
            raise ValueError(
                "`relationship_mask` must contain only 0 or NaN. "
                f"Found invalid values: {bad_vals[:10]}."
            )

        if not mask.index.equals(mask.columns) or not mask.equals(mask.T):
            raise ValueError("`relationship_mask` must be perfectly symmetric in both shape and labels.")
        

    @property
    def feature_ids(self) -> pd.Index:
        """HPO feature IDs (matrix columns)."""
        return self.matrix.columns


    @property
    def individual_ids(self) -> pd.Index:
        """Individual IDs (matrix index)."""
        return self.matrix.index