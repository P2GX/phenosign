import pandas as pd
import phenopackets as ppkt
from ..io.phenopacket_loader import EnrichedPhenopacket
from typing import List, Union,  Callable
from gpsea.preprocessing import configure_caching_cohort_creator, load_phenopackets
from gpsea.analysis.clf import monoallelic_classifier
from gpsea.model import VariantEffect
from gpsea.analysis.predicate import variant_effect, anyof
from ._utils import HPOHierarchyEngine
import logging
logger = logging.getLogger(__name__)

class PhenopacketMatrixBuilder:
    """
    Generates structured matrices from phenopacket data for downstream analysis.
    """

    def __init__(
            self, 
            phenopackets: List[EnrichedPhenopacket] | List[ppkt.Phenopacket], 
            hpo_hierarchy: HPOHierarchyEngine,         
        ):
        """
        Args:
            phenopackets (List[EnrichedPhenopacket]): 
                A list of EnrichedPhenopacket instances. 
            hpo_hierarchy (HPOHierarchyEngine): 
                An instance of HPOHierarchyEngine for hierarchical processing.

        ValueError: If `phenopackets` is empty or if HPO file fails to load.
        """
        if isinstance(phenopackets,list) and all(isinstance(p, ppkt.Phenopacket) for p in phenopackets):
            self.phenopackets = phenopackets
            self.patient_index = pd.Index([p.id for p in phenopackets], name="patient_id")
        elif isinstance(phenopackets, list) and all(isinstance(p, EnrichedPhenopacket) for p in phenopackets):
            self.phenopackets = [ppkt.phenopacket for ppkt in phenopackets]
            self.patient_index = pd.Index([p.id for p in phenopackets], name="patient_id")
            self.cohorts = pd.DataFrame(
            {'cohort': [ppkt.cohort for ppkt in phenopackets]},
            index=self.patient_index
            )
        else:
            raise ValueError("Phenopackets must be a list of either ppkt.Phenopacket or EnrichedPhenopacket instances.")

        self.hpo_hierarchy = hpo_hierarchy
        


    def _generate_status_matrix(
            self, 
            feature_extractor: Callable,  
        ) -> pd.DataFrame:
        """
        Internal method to generate a binary matrix from any phenopacket feature set (e.g., HPO terms, diseases).

        Args:
            feature_extractor (Callable): 
                Function that extracts (id, label, value) tuples from each phenopacket.

        Returns:
            pd.DataFrame:
                - Binary matrix of feature statuses (1 for observed, 0 for excluded, NaN for missing)
        """
        feature_ids, status_data = set(), {}
            
        for phenopacket in self.phenopackets:
            status_data[phenopacket.id] = {}
            for f_id, value in feature_extractor(phenopacket):
                feature_ids.add(f_id)
                status_data[phenopacket.id][f_id] = value

        matrix = pd.DataFrame.from_dict(
            status_data, 
            orient='index', 
            columns=sorted(feature_ids)
            ).reindex(self.patient_index)

        
        return matrix
      
        

    def generate_hpo_matrix(
            self, 
        ) -> pd.DataFrame:
        """
        Constructs a binary matrix indicating the presence or exclusion of HPO terms for each patient.

        Structure of the resulting matrix:
        - Rows: Patient IDs
        - Columns: HPO term IDs (e.g., HP:0004322)
        - Values:
            - 1 → Term is observed in the patient
            - 0 → Term is explicitly excluded
            - NaN → No information
        Propagation (if enabled):
        - Observed terms (1) propagate to all ancestors in the ontology
        - Excluded terms (0) propagate to all descendants

        Args:
            propagate_hierarchy (bool): (default: True)
                If True, applies hierarchical propagation. 

        Returns:
            pd.DataFrame: 
                - Binary matrix of HPO term statuses
        """
        matrix = self._generate_status_matrix(
            feature_extractor=lambda ppkt: [
                (f.type.id, 0 if f.excluded else 1) for f in ppkt.phenotypic_features
            ])
        matrix = self.hpo_hierarchy.propagate_hpo_hierarchy(matrix)
        return matrix
    

    def generate_disease_matrix(
            self, 
        ) -> pd.DataFrame:
        """
        Constructs a binary matrix indicating whether each patient has been diagnosed with specific diseases.

        Structure of the resulting matrix:
        - Rows: Patient IDs
        - Columns: Disease IDs (e.g., OMIM:101600)
        - Values:
            - 1 → Patient has been diagnosed with this disease
            - 0 → No diagnosis recorded (default)

        Returns:
            pd.DataFrame: 
                - Binary matrix of disease statuses
        """
        matrix =self._generate_status_matrix(
            feature_extractor=lambda ppkt: [
                (f.term.label, 0 if f.excluded else 1) for f in ppkt.diseases
            ]) 
        return matrix
    

    def extract_patient_metadata(
            self, 
        ) -> pd.DataFrame:
        """
        Constructs an annotation matrix for patients, containing metadata such as PMIDs
        (from externalReferences). Can be extended with cohort, age, etc.

        Rows: patient IDs
        Columns: Metadata attributes (pmids, ...)
        Values:
            - pmids → List of PubMed IDs (object type column)
        Returns:
            pd.DataFrame: Annotation matrix with patient metadata.
        """
        annotations = {}
        for ppkt in self.phenopackets:
            row = {}

            # PMIDs
            pmids = []
            meta = ppkt.meta_data
            for ref in meta.external_references:
                if hasattr(ref, "id") and ref.id.startswith("PMID:"):
                    pmids.append(ref.id.replace("PMID:", ""))
            row["pmids"] = sorted(set(pmids))           
            annotations[ppkt.id] = row

        annotation_matrix = pd.DataFrame.from_dict(
            annotations, orient="index"
        ).reindex(self.patient_index)

        return annotation_matrix


    def build_variant_matrix(
            self,
            variant_effect_type: VariantEffect,
            mane_tx_id: Union[str, List[str]]
        ) -> pd.DataFrame:

        """
        Processes variant effect annotations and creates a classification matrix.

        If both `variant_effect_type` and `mane_tx_id` are provided, assigns:
        - 1 → Patients matching the given variant effect.
        - 0 → Other patients.

        Args:
            variant_effect_type (VariantEffect):
                Target variant effect class to evaluate.
            mane_tx_id (str or List[str]):
                MANE transcript ID(s) to filter variant effects.

        Returns:
            pd.DataFrame:
                - Binary matrix of variant effect classification.
        """
        if not isinstance(variant_effect_type, VariantEffect):
            raise TypeError("variant_effect_type must be a VariantEffect enum")
        if isinstance(mane_tx_id, list) and not all(isinstance(tx, str) for tx in mane_tx_id):
            raise TypeError("mane_tx_id must be str or List[str]")
        label = str(variant_effect_type)

        cohort_creator = configure_caching_cohort_creator(self.hpo_hierarchy.tm.hpo)
        cohort, _ = load_phenopackets(phenopackets=self.phenopackets, cohort_creator=cohort_creator)

        tx_list = [mane_tx_id] if isinstance(mane_tx_id, str) else mane_tx_id
        predicates = [variant_effect(variant_effect_type, tx_id=tx) for tx in tx_list]
        predicate = anyof(predicates)

        clf = monoallelic_classifier(
            a_predicate=predicate,
            b_predicate=~predicate,
            a_label=label,
            b_label="other"
        )

        cohort_map = {p._labels.meta_label: p for p in cohort.all_patients}  # id -> patient
        data = []
        for pid in self.patient_index:
            p = cohort_map.get(pid, None)
            cat = clf.test(p) if p is not None else None
            data.append(1 if cat is not None and cat.category.name == label else 0)

        variant_effects_matrix = pd.DataFrame(
            data=data,
            index=self.patient_index,
            columns=[label]
        )

        if variant_effects_matrix[label].sum() == 0:
            logger.warning(f"Warning: The column '{label}' in variant_effects_matrix is all zeros. Please check the corresponding mane_tx_id.")
    
        return variant_effects_matrix
    
   



