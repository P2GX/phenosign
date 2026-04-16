from dataclasses import dataclass
import pandas as pd

@dataclass
class TargetData:
    """
    Container for target/condition matrices used in downstream analysis.
    """
    disease_matrix: pd.DataFrame 
    variant_condition_matrix: pd.DataFrame | None = None

    def __post_init__(self):
        matrices = {
            "disease_matrix": self.disease_matrix,
            "variant_condition_matrix": self.variant_condition_matrix,
        }
        # --- keep only non-null matrices ---
        non_null = {name: df for name, df in matrices.items() if df is not None}

        if not non_null:
            return

        # --- basic type + index alignment ---
        first_name, first_df = next(iter(non_null.items()))

        if not isinstance(first_df, pd.DataFrame):
            raise TypeError(f"{first_name} must be a pandas DataFrame")

        ref_index = first_df.index

        for name, df in non_null.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")

            if not ref_index.equals(df.index):
                raise ValueError(
                    f"{name} index does not match {first_name} index"
                )
            
            if df.index.has_duplicates:
                raise ValueError(f"{name} index must be unique")
            
            if df.columns.has_duplicates:
                raise ValueError(f"{name} columns must be unique")

            # --- value check: only 0 / 1 / NaN ---
            invalid = ~df.isin([0, 1]) & df.notna()
            if invalid.any().any():
                bad_vals = pd.unique(df[invalid].values.ravel())
                raise ValueError(
                    f"{name} must contain only 0, 1, or NaN. "
                    f"Found: {bad_vals.tolist()}"
                )