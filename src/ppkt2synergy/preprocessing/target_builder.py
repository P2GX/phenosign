import pandas as pd
import numpy as np
import phenopackets as ppkt
from collections.abc import Sequence

from gpsea.preprocessing import configure_caching_cohort_creator, load_phenopackets
from gpsea.analysis.clf import monoallelic_classifier
from gpsea.model import VariantEffect
from gpsea.analysis.predicate import variant_effect, anyof

from ..core import PhenopacketRecord
from ..core import TargetData
from ..ontology import HPOHierarchyEngine

import logging

logger = logging.getLogger(__name__)

class TargetDataBuilder:
    """
    Build target matrices from standardized individual records.

    This builder constructs disease and variant-condition target matrices
    for downstream analyses such as correlation and synergy testing.

    Notes
    -----
    Output matrices are binary and indexed by individual identifiers.
    """
    def __init__(
            self, 
            records: Sequence[PhenopacketRecord],
            raw_phenopackets: Sequence[ppkt.Phenopacket] | None = None,
            hpo_hierarchy: HPOHierarchyEngine | None = None,
        ):
        """
        Parameters
        ----------
        records : Sequence[PhenopacketRecord]
            Standardized individual records.
        raw_phenopackets : Sequence[ppkt.Phenopacket] | None, optional
            Raw phenopackets used for variant-condition classification.
        hpo_hierarchy : HPOHierarchyEngine | None, optional
            HPO hierarchy engine providing access to the ontology.
        """
        if not records:
            raise ValueError("records cannot be empty")

        self.records = records
        self.hpo = hpo_hierarchy.term_manager.hpo if hpo_hierarchy else None
        self.phenopackets = raw_phenopackets if raw_phenopackets else None
        self.individual_index = pd.Index(
            [r.individual_id for r in records],
            name="individual_id"
        )

    def build_disease_matrix(
        self,
        fill_unknown_with_zero: bool = True,
    ) -> pd.DataFrame:
        """
        Build the disease status matrix.

        Parameters
        ----------
        fill_unknown_with_zero : bool, default=True
            If ``True``, unknown diseases are encoded as ``0``. If ``False``,
            unknown diseases are left as ``NaN``.

        Returns
        -------
        pd.DataFrame
            Disease status matrix with individuals as rows and diseases as
            columns. Values are ``1`` for observed diseases, ``0`` for
            excluded diseases, and optionally ``NaN`` for unknown diseases.
        """
        all_diseases = set()

        for r in self.records:
            all_diseases.update(r.observed_diseases)
            all_diseases.update(r.excluded_diseases)

        all_diseases = sorted(all_diseases)
        data = {}

        for r in self.records:
            if fill_unknown_with_zero:
                row = {d: 0.0 for d in all_diseases}
            else:
                row = {d: np.nan for d in all_diseases}
                for d in r.excluded_diseases:
                    row[d] = 0.0

            for d in r.observed_diseases:
                row[d] = 1.0   

            data[r.individual_id] = row

        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.reindex(self.individual_index)

        return df

    def build_variant_condition_matrix(
        self,
        variant_effect_type: VariantEffect,
        mane_tx_id: str | Sequence[str],
    ) -> pd.DataFrame:
        """
        Build a binary variant-condition matrix using gpsea classifiers.

        Parameters
        ----------
        variant_effect_type : VariantEffect
            Variant effect to classify.
        mane_tx_id : str | Sequence[str]
            MANE transcript identifier or a sequence of identifiers used to define
            the variant condition.

        Returns
        -------
        pd.DataFrame
            Binary matrix with individuals as rows and a single variant
            condition column.

        Raises
        ------
        ValueError
            If raw phenopackets or the HPO ontology are unavailable.
        TypeError
            If the input types are invalid.
        """
        if self.phenopackets is None:
            raise ValueError(
                "Raw phenopackets are required to build 'variant_condition_matrix'."
            )
        if self.hpo is None:
            raise ValueError(
                "HPO ontology object is required to build 'variant_condition_matrix'."
            )

        if not isinstance(variant_effect_type, VariantEffect):
            raise TypeError( "`variant_effect_type` must be a `VariantEffect` value.")

        if isinstance(mane_tx_id, str):
            tx_list = [mane_tx_id]
        elif isinstance(mane_tx_id, Sequence):
            if not all(isinstance(tx, str) for tx in mane_tx_id):
                raise TypeError("mane_tx_id must be str or a sequence of strings")
            tx_list = list(mane_tx_id)
        else:
            raise TypeError("mane_tx_id must be str or a sequence of strings")

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
        for pid in self.individual_index:
            patient = cohort_map.get(pid)
            category = clf.test(patient) if patient is not None else None
            values.append(
                1.0 if category is not None and category.category.name == label else 0.0
            )

        df = pd.DataFrame(
            data=values,
            index=self.individual_index,
            columns=[label],
        )

        if df[label].sum() == 0:
            logger.warning(
                "The variant condition column %r contains only zeros. "
                "Please check `mane_tx_id` and `variant_effect_type`.",
                label,
            )

        return df


    def build(
        self,
        variant_effect_type: VariantEffect | None = None,
        mane_tx_id: str | Sequence[str] | None = None,
    ) -> TargetData:
        """
        Build the target data container.

        Parameters
        ----------
        variant_effect_type : VariantEffect | None, optional
            Variant effect used to build the variant-condition matrix.
        mane_tx_id : str | Sequence[str] | None, optional
            MANE transcript identifier or identifiers used to build the
            variant-condition matrix.

        Returns
        -------
        TargetData
            Target data container with disease and optional
            variant-condition matrices.
        """
        disease_matrix = (
            self.build_disease_matrix(fill_unknown_with_zero=True)
        )
        variant_matrix = None

        if variant_effect_type is not None or mane_tx_id is not None:
            if variant_effect_type is None or mane_tx_id is None:
                logger.warning(
                    "Variant-condition matrix was not built because both "
                    "`variant_effect_type` and `mane_tx_id` must be provided."
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