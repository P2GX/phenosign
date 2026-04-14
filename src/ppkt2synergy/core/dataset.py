from dataclasses import dataclass, field
from typing import Sequence
import pandas as pd
import json

from .features import HpoFeatureData
from .targets import TargetData
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhenotypeDataset:
    """
    Unified dataset for phenotype-based analysis.

    Contains:
    - HPO feature data
    - target/condition data
    - sample-level metadata
    """
    hpo_data: HpoFeatureData
    targets: TargetData = field(default_factory=TargetData)
    sample_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        ref_index = self.hpo_data.matrix.index

        if self.targets.disease_matrix is not None:
            if not ref_index.equals(self.targets.disease_matrix.index):
                raise ValueError("disease_matrix index does not match hpo matrix index.")

        if self.targets.variant_condition_matrix is not None:
            if not ref_index.equals(self.targets.variant_condition_matrix.index):
                raise ValueError("variant_condition_matrix index does not match hpo matrix index.")

        if not self.sample_metadata.empty:
            self.sample_metadata = self.sample_metadata.reindex(ref_index)

    def get_target(
        self,
        target_type: str,
        target_name: str | None = None,
        positive_class: str | None = None,
    ):
        """
        Retrieve a target for downstream analysis.

        This method supports two types of targets:

        1. Built targets (precomputed matrices)
        - "disease"
        - "variant_condition"

        2. Metadata-derived targets (on-the-fly binary encoding)
        - e.g. "sex", "cohort"

        Args:
            target_type : str
                Type of target to retrieve. One of:
                - "disease"
                - "variant_condition"
                - any column name in `sample_metadata` (e.g. "sex", "cohort")

            target_name : str | None, optional
                Column name within built target matrices.
                Required when `target_type` is:
                - "disease"
                - "variant_condition"

            positive_class : str | None, optional
                Required for metadata-derived targets.
                Defines the positive class for binary encoding:
                - 1 → sample_metadata[target_type] == positive_class
                - 0 → otherwise

        Examples:
        >>> dataset.get_target("disease", target_name="Loeys-Dietz syndrome")
        >>> dataset.get_target("variant_condition", target_name="missense_variant")
        >>> dataset.get_target("sex", positive_class="MALE")
        >>> dataset.get_target("cohort", positive_class="TGFBR1")
        """
        # -------------------------
        # Built targets
        # -------------------------
        if target_type == "disease":
            df = self.targets.disease_matrix
            if df is None:
                raise ValueError("Target 'disease' is not available.")

            if target_name is not None:
                if target_name not in df.columns:
                    raise ValueError(f"{target_name} not found in disease targets.")
                return df[target_name]

            return df

        elif target_type == "variant_condition":
            df = self.targets.variant_condition_matrix
            if df is None:
                raise ValueError("Target 'variant_condition' is not available.")

            if target_name is not None:
                if target_name not in df.columns:
                    raise ValueError(f"{target_name} not found in variant_condition targets.")
                return df[target_name]

            return df

        elif target_type in self.sample_metadata.columns:
            if positive_class is None:
                raise ValueError(
                    f"`positive_class` must be provided for metadata target '{target_type}'."
                )
            
            values = self.sample_metadata[target_type].dropna().unique()
            if positive_class not in values:
                raise ValueError(
                    f"Invalid positive_class '{positive_class}'. Available: {sorted(values)}"
                )

            return (self.sample_metadata[target_type] == positive_class).astype(float)

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def describe_available_targets(
        self
    ) -> dict:
        """
        Describe all available targets and how to use them.

        Returns
        -------
        dict
            {
                "built_targets": {
                    "disease": {
                        "available_target_names": [...]
                    },
                    "variant_condition": {
                        "available_target_names": [...]
                    }
                },
                "metadata_targets": {
                    "sex": {
                        "available_positive_classes": [...]
                    },
                    "cohort": {
                        "available_positive_classes": [...]
                    }
                }
            }
        """
        result = {
            "built_targets": {},
            "metadata_targets": {},
        }

        # -------------------------
        # Built targets
        # -------------------------
        if self.targets.disease_matrix is not None:
            result["built_targets"]["disease"] = {
                "available_target_names": list(self.targets.disease_matrix.columns)
            }

        if self.targets.variant_condition_matrix is not None:
            result["built_targets"]["variant_condition"] = {
                "available_target_names": list(
                    self.targets.variant_condition_matrix.columns
                )
            }

        # -------------------------
        # Metadata targets
        # -------------------------
        for col in ["sex", "cohort"]:
            if col in self.sample_metadata.columns:
                values = sorted(
                    v for v in self.sample_metadata[col].dropna().unique().tolist()
                )
                result["metadata_targets"][col] = {
                    "available_positive_classes": values
                }

        # -------------------------
        # Warning
        # -------------------------
        if not result["built_targets"] and not result["metadata_targets"]:
            logger.warning(
                "No available targets found. "
                "Make sure targets are built and metadata contains valid columns."
            )

        print(json.dumps(result, indent=2))
    
    def get_pmids(
        self, 
        patient_ids: Sequence[str]
    )-> pd.Series:
        return self.sample_metadata.loc[patient_ids, "pmids"]

    def describe(self) -> str:
        """
        Return a short human-readable summary of the dataset.

        Returns:
            str
                Text summary of dataset dimensions and available targets.
        """
        n_samples = self.hpo_data.matrix.shape[0]
        n_features = self.hpo_data.matrix.shape[1]

        lines = [
            "PhenotypeDataset",
            f"- samples: {n_samples}",
            f"- HPO features: {n_features}",
        ]

        if self.targets.disease_matrix is not None:
            lines.append(
                f"- disease targets: {self.targets.disease_matrix.columns.tolist()}"
            )

        if self.targets.variant_condition_matrix is not None:
            lines.append(
                f"- variant condition targets: {self.targets.variant_condition_matrix.columns.tolist()}"
            )

        if not self.sample_metadata.empty:
            lines.append(
                f"- sample metadata columns: {self.sample_metadata.columns.tolist()}"
            )

        return "\n".join(lines)
    
