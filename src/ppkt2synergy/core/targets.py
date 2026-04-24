from dataclasses import dataclass
import pandas as pd

@dataclass
class TargetData:
    """
    Container for target matrices used in downstream analysis.

    Parameters
    ----------
    disease_matrix : pd.DataFrame
        Binary disease target matrix with individuals as rows and diseases
        as columns. Values must be ``1`` (observed), ``0`` (excluded),
        or ``NaN`` (unknown).
    variant_condition_matrix : pd.DataFrame | None, optional
        Binary variant-condition target matrix with individuals as rows and
        variant conditions as columns. Values must be ``1``, ``0``, or ``NaN``.
    """
    disease_matrix: pd.DataFrame 
    variant_condition_matrix: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        matrices = {
            "disease_matrix": self.disease_matrix,
            "variant_condition_matrix": self.variant_condition_matrix,
        }

        non_null = {k: v for k, v in matrices.items() if v is not None}

        if not non_null:
            return

        first_name, first_df = next(iter(non_null.items()))

        if not isinstance(first_df, pd.DataFrame):
            raise TypeError(f"`{first_name}` must be a pandas DataFrame.")

        ref_index = first_df.index

        for name, df in non_null.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"`{name}` must be a pandas DataFrame.")

            if not ref_index.equals(df.index):
                raise ValueError(f"`{name}` index must match `{first_name}`.")

            if df.index.has_duplicates:
                raise ValueError(f"`{name}` index must be unique.")

            if df.columns.has_duplicates:
                raise ValueError(f"`{name}` columns must be unique.")

            if (~df.isin([0, 1]) & df.notna()).any().any():
                raise ValueError(f"`{name}` must contain only 0, 1, or NaN.")