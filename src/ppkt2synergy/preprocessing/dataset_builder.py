import phenopackets as ppkt
import pandas as pd
from typing import IO
from gpsea.model import VariantEffect

from .feature_builder import HpoFeatureBuilder
from .target_builder import TargetDataBuilder
from .adapters import phenopackets_to_records, enriched_phenopackets_to_records
from ..core import PhenotypeDataset
from ..ontology import HPOHierarchyEngine
from ..io import EnrichedPhenopacket


class PhenotypeDatasetBuilder:
    """
    Build an analysis-ready PhenotypeDataset from phenopacket inputs.

    This builder coordinates:
    - conversion of raw phenopackets into internal patient records
    - construction of HPO feature data
    - construction of target data
    - construction of sample metadata

    Notes
    -----
    - HPO feature data are always built.
    - Disease targets are always built.
    - Variant condition targets are built only when both
      `variant_effect_type` and `mane_tx_id` are provided.
    - Sample metadata may include:
      - cohort
      - pmids
      - sex
      - age
    """
    def __init__(
        self,
        phenopackets: list[ppkt.Phenopacket] | list[EnrichedPhenopacket],
        hpo_file: str | IO | None = None,
        hpo_release: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        phenopackets : list[ppkt.Phenopacket] | list[EnrichedPhenopacket]
            Input phenopackets.
        hpo_file : str | IO | None, optional
            Local HPO file.
        hpo_release : str | None, optional
            HPO release version.
        """
        if not phenopackets:
            raise ValueError("phenopackets cannot be empty")

        if all(isinstance(p, ppkt.Phenopacket) for p in phenopackets):
            self.raw_phenopackets = phenopackets
            self.records = phenopackets_to_records(phenopackets)

        elif all(isinstance(p, EnrichedPhenopacket) for p in phenopackets):
            self.raw_phenopackets = [item.phenopacket for item in phenopackets]
            self.records = enriched_phenopackets_to_records(phenopackets)

        else:
            raise TypeError(
                "phenopackets must be a list of Phenopacket or EnrichedPhenopacket"
            )

        self.hpo_hierarchy = HPOHierarchyEngine(
            hpo_file=hpo_file,
            release=hpo_release,
        )

    def build_sample_metadata(
        self
    ) -> pd.DataFrame:
        """
        Build sample-level metadata.

        Returns:
            pd.DataFrame
                Rows = patients
                Columns may include:
                - cohort
                - pmids
                - sex
                - age
            """
        data = {}

        for r in self.records:
            data[r.patient_id] = {
                "cohort": r.cohort,
                "sex": r.sex,
                "age": r.age,
                "pmids": r.metadata.get("pmids"),
            }

        df = pd.DataFrame.from_dict(data, orient="index")
        return df

    def build(
        self,
        variant_effect_type: VariantEffect | None = None,
        mane_tx_id: str | list[str] | None = None,
        missing_threshold: float = 1.0,
    ) -> PhenotypeDataset:
        """
        Build the final dataset containing feature matrices and label vectors.

        Args:
            variant_effect_type : VariantEffect | None, optional
                The type of variant effect to consider.
            mane_tx_id : str | list[str] | None, optional
                The MANE transcript ID(s) to consider.
            missing_threshold : float, default=1.0
                Maximum allowed proportion of missing values per HPO term.

        Returns:          
            PhenotypeDataset
                Final dataset containing:
                - hpo feature data
                - target data
                - sample metadata
      
        """
        # -------------------------
        # HPO features
        # -------------------------
        feature_builder = HpoFeatureBuilder(records=self.records, hpo_hierarchy=self.hpo_hierarchy)
        hpo_data = feature_builder.build(missing_threshold=missing_threshold)

        # -------------------------
        # Targets
        # -------------------------
        target_builder = TargetDataBuilder(
            records=self.records,
            raw_phenopackets=self.raw_phenopackets,
            hpo_hierarchy=self.hpo_hierarchy,
        )

        targets = target_builder.build(
            variant_effect_type=variant_effect_type,
            mane_tx_id=mane_tx_id,
        )

        # -------------------------
        # sample metadata
        # -------------------------
        sample_metadata = self.build_sample_metadata()

        # -------------------------
        # Final dataset
        # -------------------------
        return PhenotypeDataset(
            hpo_data=hpo_data,
            targets=targets,
            sample_metadata=sample_metadata,
        )