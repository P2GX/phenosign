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
        non_null = {name: df for name, df in matrices.items() if df is not None}

        if not non_null:
            return

        first_name, first_df = next(iter(non_null.items()))
        ref_index = first_df.index

        for name, df in non_null.items():
            if not ref_index.equals(df.index):
                missing = set(ref_index) - set(df.index)
                extra = set(df.index) - set(ref_index)
                raise ValueError(
                    f"{name} index does not match {first_name} index.\n"
                    f"Missing samples: {missing}\n"
                    f"Extra samples: {extra}"
                )