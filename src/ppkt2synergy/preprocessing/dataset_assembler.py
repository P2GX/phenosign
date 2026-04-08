from typing import List, IO, Tuple
import phenopackets as ppkt
from .matrix_builder import PhenopacketMatrixBuilder
from ..io.phenopacket_loader import EnrichedPhenopacket
from .matrices import HpoFeatureMatrix, TargetMatrix
from gpsea.model import VariantEffect
from ._utils import HPOHierarchyEngine

class PhenopacketDatasetAssembler:
    """
    Assembles HPO observation matrices and disease/target matrices from phenopackets.

    This class processes phenopacket data to generate:
    - HPO term observation matrices with hierarchical propagation.
    - Disease/target matrices with optional variant effect filtering.
    - Patient metadata and cohort information.

    Features:
    - Filters HPO terms based on missing value threshold.
    - Propagates observed/excluded terms through the HPO hierarchy.
    - Builds variant effect matrix if variant effect type and MANE transcript ID(s) are provided.

    Example:
        from ppkt2synergy import load_phenopackets
        from gpsea.model import VariantEffect

        phenopackets = load_phenopackets('FBN1')

        assembler = PhenopacketDatasetAssembler(phenopackets)
        hpo_matrix, target_matrix = assembler.build(
            variant_effect_type=VariantEffect.MISSENSE_VARIANT,
            mane_tx_id="ENST00000369535",
            threshold=0.3
        )
    """

    def __init__(
        self,
        phenopackets: List[EnrichedPhenopacket] | List[ppkt.Phenopacket],
        hpo_file: IO | str | None = None,
        release: str | None = None
        ):

        self.hpo_engine = HPOHierarchyEngine(hpo_file, release)
        self.builder = PhenopacketMatrixBuilder(
            phenopackets=phenopackets,
            hpo_hierarchy=self.hpo_engine
        )

    def build(
            self,
            variant_effect_type: VariantEffect| None = None,
            mane_tx_id: str | List[str] | None = None,
            threshold: float = 1.0, 
        ) -> Tuple[HpoFeatureMatrix, TargetMatrix]:
        """
        Builds HPO feature matrix and disease/target matrix from phenopackets.

        Args:
            variant_effect_type (VariantEffect, optional):
                Specific variant effect type to include in target matrix.
            mane_tx_id (str or List[str], optional):
                MANE transcript ID(s) for filtering variant effects.
            threshold (float, default=1.0):
                Maximum allowed proportion of NaN values per HPO term column.
                Columns exceeding this threshold are dropped. threshold must be between 0 and 1.

        Returns:
            tuple[HpoFeatureMatrix, TargetMatrix]:
                - HpoFeatureMatrix: filtered HPO term observation matrix with patient info
                  and hierarchical relationship mask
                - TargetMatrix: disease matrix, cohort info, and optional variant effect data

        Raises:
            ValueError: if threshold not in [0,1] or no valid HPO terms remain after filtering.
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"NaN threshold {threshold} must be between 0 and 1") 
        
        hpo_matrix = self.builder.generate_hpo_matrix()
        hpo_matrix_filtered = hpo_matrix.dropna(axis=1, thresh=int((1-threshold) * len(hpo_matrix)))
       
        if hpo_matrix_filtered.empty:
            raise ValueError("No valid terms found. Adjust threshold.")

        disease_matrix = self.builder.generate_disease_matrix()
        patient_info_df = self.builder.extract_patient_metadata()
        cohort_info_df = getattr(self.builder, 'cohorts', None)

        variant_effect_df = None
        if variant_effect_type is not None and mane_tx_id is not None:
            variant_effect_df = self.builder.build_variant_matrix(variant_effect_type=variant_effect_type, mane_tx_id=mane_tx_id)  
        hpo_relationship_mask = self.hpo_engine.build_relationship_mask(hpo_matrix_filtered.columns)
        hpo_label_mapping = self.hpo_engine.get_labels()
        return (
                    HpoFeatureMatrix(
                        hpo_matrix=hpo_matrix_filtered,
                        label_mapping=hpo_label_mapping,
                        patient_info_df=patient_info_df,
                        hpo_relationship_mask=hpo_relationship_mask
                    ),
                    TargetMatrix(
                        disease_matrix=disease_matrix,
                        cohort_info_df=cohort_info_df,
                        variant_effect_df=variant_effect_df
                    )
                )
                   




