from dataclasses import dataclass, field
import pandas as pd

@dataclass
class HpoFeatureMatrix:
    """
    Container for patient HPO feature matrices.

    Attributes
    ----------
    hpo_matrix : pd.DataFrame
        Binary matrix of HPO term observations
        (rows=patients, columns=HPO terms).
        Values: 1=observed, 0=excluded, NaN=unknown.

    label_mapping : dict[str, str]
        Mapping from HPO term IDs to human-readable labels.

    patient_info_df : pd.DataFrame
        Patient-level metadata (e.g., 'pmids', 'age', 'sex').

    hpo_relationship_mask : pd.DataFrame | None, optional
        Matrix indicating hierarchical relationships between HPO terms
        (terms x terms), NaN if ancestor/descendant, 0 otherwise.
    """
    hpo_matrix: pd.DataFrame
    label_mapping: dict[str, str] = field(default_factory=dict)
    patient_info_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    hpo_relationship_mask: pd.DataFrame | None = None

    def __post_init__(self):
        # ------------------- Align patient_info_df -------------------
        if not self.hpo_matrix.empty and not self.patient_info_df.empty:
            if not self.hpo_matrix.index.equals(self.patient_info_df.index):
                try:
                    self.patient_info_df = self.patient_info_df.reindex(self.hpo_matrix.index)
                except Exception:
                    raise ValueError(
                        "Cannot align patient_info_df index with hpo_matrix index."
                    )

        # ------------------- Align hpo_relationship_mask -------------------
        if self.hpo_relationship_mask is not None:
            if not self.hpo_relationship_mask.empty:
                terms_set = set(self.hpo_matrix.columns)
                mask_rows_set = set(self.hpo_relationship_mask.index)
                mask_cols_set = set(self.hpo_relationship_mask.columns)
                missing_rows = terms_set - mask_rows_set
                extra_rows = mask_rows_set - terms_set
                missing_cols = terms_set - mask_cols_set
                extra_cols = mask_cols_set - terms_set
                if missing_rows or extra_rows or missing_cols or extra_cols:
                    raise ValueError(
                        f"hpo_relationship_mask rows/columns do not match hpo_matrix columns.\n"
                        f"Missing rows: {missing_rows}\nExtra rows: {extra_rows}\n"
                        f"Missing cols: {missing_cols}\nExtra cols: {extra_cols}"
                    )


@dataclass
class TargetMatrix:
    """
    Container for disease and variant target matrices.

    Attributes
    ----------
    disease_matrix : pd.DataFrame
        Binary matrix of disease diagnoses (rows=patients, columns=diseases).

    cohort_info_df : pd.DataFrame| None, optional
        Cohort-level metadata (rows must match disease_matrix index).

    variant_effect_df : pd.DataFrame | None, optional
        Optional binary matrix of variant effects (rows match disease_matrix index).
    """
    disease_matrix: pd.DataFrame
    cohort_info_df: pd.DataFrame | None = None
    variant_effect_df: pd.DataFrame | None = None

    def __post_init__(self):
        # ------------------- Strict alignment for cohort_info_df -------------------
        if not self.disease_matrix.empty and not self.cohort_info_df.empty:
            if not self.disease_matrix.index.equals(self.cohort_info_df.index):
                missing = set(self.disease_matrix.index) - set(self.cohort_info_df.index)
                extra = set(self.cohort_info_df.index) - set(self.disease_matrix.index)
                raise ValueError(
                    f"cohort_info_df index does not match disease_matrix index.\n"
                    f"Missing patients: {missing}\nExtra patients: {extra}"
                )

        # ------------------- Strict alignment for variant_effect_df -------------------
        if self.variant_effect_df is not None:
            if not self.disease_matrix.index.equals(self.variant_effect_df.index):
                missing = set(self.disease_matrix.index) - set(self.variant_effect_df.index)
                extra = set(self.variant_effect_df.index) - set(self.disease_matrix.index)
                raise ValueError(
                    f"variant_effect_df index does not match disease_matrix index.\n"
                    f"Missing patients: {missing}\nExtra patients: {extra}"
                )
    
