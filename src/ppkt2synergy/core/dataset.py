from dataclasses import dataclass, field
from typing import Sequence
import pandas as pd
import json
import logging

from .features import HpoFeatureData
from .targets import TargetData

logger = logging.getLogger(__name__)


@dataclass
class PhenotypeDataset:
    """
    Unified dataset for phenotype-based analysis.

    Parameters
    ----------
    hpo_data : HpoFeatureData
        HPO feature data.
    targets : TargetData | None, optional
        Target matrices for downstream analysis. If ``None``, an empty
        target container is created.
    individual_metadata : pd.DataFrame, optional
        Metadata indexed by individual identifiers.
    """
    hpo_data: HpoFeatureData
    targets: TargetData | None = None
    individual_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        ref_index = self.hpo_data.matrix.index

        if self.targets.disease_matrix is not None:
            if not ref_index.equals(self.targets.disease_matrix.index):
                 raise ValueError(
                    "`targets.disease_matrix` index must match `hpo_data.matrix.index`."
                )

        if self.targets.variant_condition_matrix is not None:
            if not ref_index.equals(self.targets.variant_condition_matrix.index):
                raise ValueError("`targets.variant_condition_matrix` index must match `hpo_data.matrix.index`.")

        if not isinstance(self.individual_metadata, pd.DataFrame):
            raise TypeError(
                "`individual_metadata` must be a pandas DataFrame."
            )

        if not self.individual_metadata.empty:
            self.individual_metadata = self.individual_metadata.reindex(ref_index)


    def get_target(
        self,
        target_type: str,
        target_name: str | None = None,
        positive_class: str | None = None,
    ) -> pd.DataFrame | pd.Series:
        """
        Retrieve a target for downstream analysis.

        Parameters
        ----------
        target_type : str
            Target type. Supported built targets are ``"disease"`` and
            ``"variant_condition"``. Metadata-derived targets can also be
            retrieved using a column name from ``individual_metadata``.
        target_name : str | None, optional
            Column name within a built target matrix. If omitted, the full
            target matrix is returned.
        positive_class : str | None, optional
            Positive class used to binarize a metadata column. Required for
            metadata-derived targets.

        Returns
        -------
        pd.DataFrame | pd.Series
            Target matrix or target vector.

        Raises
        ------
        ValueError
            If the requested target is unavailable or invalid.

        Examples:
        >>> dataset.get_target("disease", target_name="Loeys-Dietz syndrome")
        >>> dataset.get_target("variant_condition", target_name="missense_variant")
        >>> dataset.get_target("sex", positive_class="MALE")
        >>> dataset.get_target("cohort", positive_class="TGFBR1")
        """
        if target_type == "disease":
            df = self.targets.disease_matrix
            if df is None or df.empty:
                raise ValueError("Target type `'disease'` is not available.")

            if target_name is None:
                raise ValueError("`target_name` must be provided for target type `'disease'`.")

            if target_name not in df.columns:
                raise ValueError(f"Target `{target_name}` not found in `disease_matrix`.")
            return df[target_name]

        if target_type == "variant_condition":
            df = self.targets.variant_condition_matrix
            if df is None or df.empty:
                raise ValueError("Target type `'variant_condition'` is not available.")

            if target_name is None:
                raise ValueError(
                    "`target_name` must be provided for target type `'variant_condition'`."
                )

            if target_name not in df.columns:
                raise ValueError(
                    f"Target `{target_name}` not found in `variant_condition_matrix`."
                )
            return df[target_name]

        if target_type in self.individual_metadata.columns:
            if positive_class is None:
                raise ValueError(
                    f"`positive_class` must be provided for metadata target `{target_type}`."
                )

            values = self.individual_metadata[target_type].dropna().unique().tolist()
            if positive_class not in values:
                raise ValueError(
                    f"Invalid `positive_class` {positive_class!r} for `{target_type}`. "
                    f"Available values: {sorted(values)}."
                )

            col = self.individual_metadata[target_type]
            result = (col == positive_class).astype(float)
            result[col.isna()] = float("nan")
            return result

        raise ValueError(f"Unknown target type: {target_type!r}.")
    

    def describe_available_targets(
        self
    ):
        """
        Describe available built and metadata-derived targets.
        """
        result = {
            "built_targets": {},
            "metadata_targets": {},
        }

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

        for col in ["sex", "cohort"]:
            if col in self.individual_metadata.columns:
                values = sorted(
                    v for v in self.individual_metadata[col].dropna().unique().tolist()
                )
                result["metadata_targets"][col] = {
                    "available_positive_classes": values
                }

        if not result["built_targets"] and not result["metadata_targets"]:
            logger.warning(
                "No available targets found. Make sure targets are built and metadata contains valid columns."
            )

        print(json.dumps(result, indent=2))

    
    def get_pmids(
        self, 
        individual_ids: Sequence[str]
    )-> pd.Series:
        """
        Retrieve PMIDs for the specified individuals.

        Parameters
        ----------
        individual_ids : Sequence[str]
            Individual identifiers.

        Returns
        -------
        pd.Series
            PMID entries indexed by individual identifier.
        """
        return self.individual_metadata.loc[individual_ids, "pmids"]
    

    def describe(self) -> str:
        """
        Return a short summary of the dataset.

        Returns
        -------
        str
            Human-readable dataset summary.
        """
        n_individuals = self.hpo_data.matrix.shape[0]
        n_features = self.hpo_data.matrix.shape[1]

        lines = [
            "PhenotypeDataset",
            f"- individuals: {n_individuals}",
            f"- HPO features: {n_features}",
        ]

        if self.targets.disease_matrix is not None and not self.targets.disease_matrix.empty:
            lines.append(
                f"- disease targets: {self.targets.disease_matrix.columns.tolist()}"
            )

        if (
            self.targets.variant_condition_matrix is not None
            and not self.targets.variant_condition_matrix.empty
        ):
            lines.append(
                "- variant condition targets: "
                f"{self.targets.variant_condition_matrix.columns.tolist()}"
            )

        if not self.individual_metadata.empty:
            lines.append(
                f"- individual metadata columns: {self.individual_metadata.columns.tolist()}"
            )

        return "\n".join(lines)
    
