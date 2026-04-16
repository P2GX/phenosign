import pandas as pd
import numpy as np

from ..core import PhenopacketRecord
from ..core import TargetData
from ..ontology import HPOHierarchyEngine
import phenopackets as ppkt

from gpsea.preprocessing import configure_caching_cohort_creator, load_phenopackets
from gpsea.analysis.clf import monoallelic_classifier
from gpsea.model import VariantEffect
from gpsea.analysis.predicate import variant_effect, anyof

import logging

logger = logging.getLogger(__name__)

class TargetDataBuilder:
    """
    Build structured target matrices from patient-level records.

    This builder constructs:

    - disease_matrix
    - variant_condition_matrix

    These matrices are intended for downstream statistical analysis
    (e.g. correlation, synergy).

    Notes
    -----
    - Output matrices are binary (1 / 0) with patients as rows.
    - All matrices are aligned by patient_id index.
    """

    def __init__(
            self, 
            records: list[PhenopacketRecord],
            raw_phenopackets: list[ppkt.Phenopacket] | None = None,
            hpo_hierarchy: HPOHierarchyEngine | None = None,
        ):
        if not records:
            raise ValueError("records cannot be empty")

        self.records = records
        self.hpo = hpo_hierarchy.term_manager.hpo if hpo_hierarchy else None
        self.phenopackets = raw_phenopackets if raw_phenopackets else None
        self.patient_index = pd.Index(
            [r.patient_id for r in records],
            name="patient_id"
        )

    # ------------------------------------------------------------------
    # Disease
    # ------------------------------------------------------------------

    def build_disease_matrix(
        self,
        fill_unknown_with_zero: bool = True,
    ) -> pd.DataFrame:
        """
        Build disease status matrix.

        Args:
            fill_unknown_with_zero : bool
            If True, unknown diseases are encoded as 0.
            If False, unknown diseases are left as NaN.

        Returns:
            pd.DataFrame
                Rows = patients
                Columns = diseases
                Values:
                    1   = observed (diagnosed)
                    0   = not observed as positive target (including explicitly excluded and unknown)
        """
        all_diseases = set()

        # Collect all disease terms
        for r in self.records:
            all_diseases.update(r.observed_diseases)
            all_diseases.update(r.excluded_diseases)

        all_diseases = sorted(all_diseases)

        data = {}

        for r in self.records:
            # Start with unknown
            if fill_unknown_with_zero:
                row = {d: 0.0 for d in all_diseases}
            else:
                row = {d: np.nan for d in all_diseases}
                for d in r.excluded_diseases:
                    row[d] = 0.0

            # Observed diseases → 1
            for d in r.observed_diseases:
                row[d] = 1.0

            

            data[r.patient_id] = row

        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.reindex(self.patient_index)

        return df

    # ------------------------------------------------------------------
    # Variant condition
    # ------------------------------------------------------------------

    def build_variant_condition_matrix(
        self,
        variant_effect_type: list[VariantEffect],
        mane_tx_id: str | list[str],
    ) -> pd.DataFrame:
        """
        Build a binary variant condition matrix using gpsea classifiers.

        Args:
            variant_effect_type : lsit[VariantEffect]
                Variant effect class to evaluate.
            mane_tx_id : str | list[str]
                Transcript ID(s) used to filter the variant effect.

        Returns:
            pd.DataFrame
                Binary matrix (patients x 1):
                1 = matches requested variant condition
                0 = does not match
        """
        if self.phenopackets is None:
            raise ValueError(
                "Raw phenopackets are required to build variant_condition_matrix."
            )
        if self.hpo is None:
            raise ValueError(
                "HPO ontology object is required to build variant_condition_matrix."
            )

        if not isinstance(variant_effect_type, VariantEffect):
            raise TypeError("variant_effect_type must be a VariantEffect enum")

        if isinstance(mane_tx_id, list):
            if not all(isinstance(tx, str) for tx in mane_tx_id):
                raise TypeError("mane_tx_id must be str or list[str]")
            tx_list = mane_tx_id
        elif isinstance(mane_tx_id, str):
            tx_list = [mane_tx_id]
        else:
            raise TypeError("mane_tx_id must be str or list[str]")

        label = str(variant_effect_type)

        cohort_creator = configure_caching_cohort_creator(self.hpo)
        cohort, _ = load_phenopackets(
            phenopackets=self.phenopackets,
            cohort_creator=cohort_creator,
        )

        predicates = [variant_effect(variant_effect_type, tx_id=tx) for tx in tx_list]
        predicate = anyof(predicates)

        clf = monoallelic_classifier(
            a_predicate=predicate,
            b_predicate=~predicate,
            a_label=label,
            b_label="other",
        )

        cohort_map = {p._labels.meta_label: p for p in cohort.all_patients}

        values = []
        for pid in self.patient_index:
            patient = cohort_map.get(pid)
            category = clf.test(patient) if patient is not None else None
            values.append(
                1.0 if category is not None and category.category.name == label else 0.0
            )

        df = pd.DataFrame(
            data=values,
            index=self.patient_index,
            columns=[label],
        )

        if df[label].sum() == 0:
            logger.warning(
                "The variant condition column '%s' is all zeros. "
                "Please check the provided mane_tx_id / variant_effect_type.",
                label,
            )

        return df

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------

    def build(
        self,
        variant_effect_type: VariantEffect | None = None,
        mane_tx_id: str | list[str] | None = None,
    ) -> TargetData:
        """
        Build TargetData container.

        Args:
            variant_effect_type : VariantEffect | None, optional
                The type of variant effect to consider. If None, the variant
                condition matrix will not be built.
            mane_tx_id : str | list[str] | None, optional
                The MANE transcript ID(s) to consider. If None, the variant
                condition matrix will not be built.

        Returns:
            TargetData
        """
        disease_matrix = (
            self.build_disease_matrix(fill_unknown_with_zero=True)
        )
        variant_matrix = None

        if variant_effect_type is not None or mane_tx_id is not None:
            # enforce both or none
            if variant_effect_type is None or mane_tx_id is None:
                logger.warning(
                    "Variant condition not built because both parameters "
                    "`variant_effect_type` and `mane_tx_id` were not provided."
                )
                variant_matrix = None
            else:
                variant_matrix = self.build_variant_condition_matrix(
                    variant_effect_type=variant_effect_type,
                    mane_tx_id=mane_tx_id,
                )

        return TargetData(
            disease_matrix=disease_matrix,
            variant_condition_matrix=variant_matrix,
        )