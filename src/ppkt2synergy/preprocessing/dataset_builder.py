from typing import IO
from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd
import phenopackets as ppkt
import numpy as np

from ..core.features import HpoFeatureData
from ..core import PhenotypeDataset
from ..ontology import HPOHierarchyEngine

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class _HpoObservation:
    individual_id: str
    observed_terms: frozenset[str]
    excluded_terms: frozenset[str]

class PhenotypeDatasetBuilder:
    """
    Build an analysis-ready ``PhenotypeDataset`` from phenopacket inputs.

    Responsibilities:
    - validate phenopacket IDs early
    - build HPO feature matrix
    - apply HPO propagation
    - filter by missingness
    - build HPO labels and relationship mask
    - optionally attach a prebuilt GPSEA cohort
    """
    def __init__(
        self,
        phenopackets: list[ppkt.Phenopacket],
        *,
        hpo_file: str | IO | None = None,
        hpo_release: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        phenopackets : list[ppkt.Phenopacket]
            Input phenopackets.
        hpo_file : str | IO | None, optional
            Local HPO file.
        hpo_release : str | None, optional
            HPO release version.
        """
        self._validate_phenopackets(phenopackets)
        self.phenopackets = phenopackets
        self._hpo_engine = HPOHierarchyEngine(
            hpo_file=hpo_file,
            release=hpo_release,
        )

    def build(
        self,
        missing_threshold: float = 1.0,
        build_gpsea_cohort: bool = False,
    ) -> PhenotypeDataset:
        """
        Parse phenopackets and assemble a ``PhenotypeDataset`

        Parameters
        ----------
        missing_threshold : float
            Maximum allowed proportion of missing values per HPO term (0-1).
        build_gpsea_cohort : bool
            Whether to build a GPSEA cohort.

        Returns
        -------
        PhenotypeDataset
            Contains HPO feature data, metadata dictionary, raw phenopackets,
            optionally condition vector and targets.
        """
        observations = self._parse_hpo_observations(self.phenopackets)
        raw_matrix = self._build_raw_hpo_matrix(observations)

        if raw_matrix.shape[1] == 0:
            raise ValueError(
                "No HPO terms found in phenopackets. "
                "Check `phenotypic_features`."
            )

        propagated_matrix = self._hpo_engine.propagate(raw_matrix)

        filtered_matrix = self._filter_by_missingness(
            propagated_matrix,
            missing_threshold=missing_threshold,
        )

        if filtered_matrix.shape[1] == 0:
            raise ValueError(
                "No HPO terms remain after propagation and missingness filtering. "
                f"missing_threshold={missing_threshold}, "
                f"n_raw_terms={raw_matrix.shape[1]}, "
                f"n_after_propagation={propagated_matrix.shape[1]}."
            )

        hpo_data = HpoFeatureData(
            matrix=filtered_matrix,
            label_mapping=self._hpo_engine.get_labels(),
            relationship_mask=self._hpo_engine.build_relationship_mask(
                filtered_matrix.columns,
            ),
        )

        gpsea_cohort = None

        if build_gpsea_cohort:
            from gpsea.preprocessing import configure_caching_cohort_creator, load_phenopackets 
            cohort_creator = configure_caching_cohort_creator(self._hpo_engine.hpo) 
            gpsea_cohort, _ = load_phenopackets(phenopackets=self.phenopackets, cohort_creator=cohort_creator, )

        return PhenotypeDataset(
            hpo_data=hpo_data,
            phenopackets=self.phenopackets,
            gpsea_cohort= gpsea_cohort,
        )
    
    @staticmethod
    def _validate_phenopackets(
        phenopackets: list[ppkt.Phenopacket]
    ) -> None:
        if not phenopackets:
            raise ValueError("phenopackets cannot be empty")

        if not all(isinstance(p, ppkt.Phenopacket) for p in phenopackets):
            raise TypeError("`phenopackets` must contain only Phenopacket objects.")
        
        seen: set[str] = set()

        for phenopacket in phenopackets:
            individual_id = getattr(phenopacket, "id", None)

            if not individual_id:
                raise ValueError("Every phenopacket must have a non-empty `id`.")

            individual_id = str(individual_id)

            if individual_id in seen:
                raise ValueError(
                    f"Duplicate phenopacket id detected: {individual_id!r}. "
                    "Phenopacket ids define dataset row identities and must be unique."
                )

            seen.add(individual_id)

    @staticmethod
    def _parse_hpo_observations(
        phenopackets: list[ppkt.Phenopacket]
    ) -> list[_HpoObservation]:
        
        observations: list[_HpoObservation] = []

        for phenopacket in phenopackets:
            individual_id = phenopacket.id

            observed_terms: set[str] = set()
            excluded_terms: set[str] = set()

            for f in phenopacket.phenotypic_features:
                if not getattr(f, "type", None) or not getattr(f.type, "id", None):
                    continue
                term_id = f.type.id
                if getattr(f, "excluded", False):
                    excluded_terms.add(term_id)
                else:
                    observed_terms.add(term_id)

            for feature in getattr(phenopacket, "phenotypic_features", []):
                term = getattr(feature, "type", None)
                term_id = getattr(term, "id", None)

                if not term_id:
                    continue

                if getattr(feature, "excluded", False):
                    excluded_terms.add(term_id)
                else:
                    observed_terms.add(term_id)

            conflict = observed_terms & excluded_terms
            if conflict:
                logger.warning(
                    "Individual %s has conflicting HPO annotations. "
                    "Observed takes precedence. Conflicts: %s",
                    individual_id,
                    sorted(conflict),
                )
                excluded_terms -= conflict

            observations.append(
                _HpoObservation(
                    individual_id=individual_id,
                    observed_terms=frozenset(observed_terms),
                    excluded_terms=frozenset(excluded_terms),
                )
            )

        return observations

    @staticmethod
    def _build_raw_hpo_matrix(
        observations: list[_HpoObservation],
    ) -> pd.DataFrame:
        """
        Construct the raw HPO status matrix from individual records.

        Returns
        -------
        pd.DataFrame
            HPO status matrix with individuals as rows and HPO terms as
            columns. Values are ``1`` for observed terms, ``0`` for excluded
            terms, and ``NaN`` for unknown terms.
        """
        terms = sorted(
            set().union(
                *(obs.observed_terms | obs.excluded_terms for obs in observations)
            )
        )

        individual_ids = [obs.individual_id for obs in observations]

        matrix = np.full(
            shape=(len(observations), len(terms)),
            fill_value=np.nan,
            dtype=np.float32,
        )

        term_to_col = {term: col for col, term in enumerate(terms)}

        for row, obs in enumerate(observations):
            for term in obs.excluded_terms:
                matrix[row, term_to_col[term]] = 0.0

            for term in obs.observed_terms:
                matrix[row, term_to_col[term]] = 1.0

        return pd.DataFrame(
            matrix,
            index=pd.Index(individual_ids, name="individual_id"),
            columns=pd.Index(terms, name="hpo_id"),
        )
    

    @staticmethod
    def _filter_by_missingness(
        matrix: pd.DataFrame,
        missing_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Drop columns whose missing-value proportion exceeds the threshold.

        Parameters
        ----------
        matrix : pd.DataFrame
            HPO status matrix.
        missing_threshold : float, default=1.0
            Maximum allowed proportion of missing values per column.
            Must satisfy ``0 <= missing_threshold <= 1``.

        Returns
        -------
        pd.DataFrame
            Filtered HPO matrix.

        Raises
        ------
        ValueError
            If ``missing_threshold`` is outside ``[0, 1]``.
        """
        if not isinstance(missing_threshold, (int, float)):
            raise TypeError("`missing_threshold` must be numeric.")

        if not 0.0 <= float(missing_threshold) <= 1.0:
            raise ValueError("`missing_threshold` must be between 0 and 1.")

        n_samples = matrix.shape[0]
        min_non_missing = int(np.ceil((1.0 - missing_threshold) * n_samples))
        min_non_missing = max(min_non_missing, 1)

        return matrix.dropna(axis=1, thresh=min_non_missing)

    def _build_gpsea_cohort(
        self
    ) -> Any:
        try:
            from gpsea.preprocessing import (
                configure_caching_cohort_creator,
                load_phenopackets,
            )
        except ImportError as exc:
            raise ImportError(
                "`build_gpsea_cohort=True` requires GPSEA to be installed."
            ) from exc

        cohort_creator = configure_caching_cohort_creator(self._hpo_engine.hpo)

        gpsea_cohort, _ = load_phenopackets(
            phenopackets=self.phenopackets,
            cohort_creator=cohort_creator,
        )

        return gpsea_cohort

        