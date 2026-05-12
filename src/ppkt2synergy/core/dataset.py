import phenopackets as ppkt
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from gpsea.model import VariantEffect, Cohort

import logging

from .features import HpoFeatureData

logger = logging.getLogger(__name__)


@dataclass
class PhenotypeDataset:
    """
    Unified dataset for phenotype-based analysis.

    Parameters
    ----------
    hpo_data : HpoFeatureData
        HPO feature data.
    phenopackets : list[ppkt.Phenopacket], optional
        Original phenopackets. Retained for reference and downstream condition computation.
    phenopacket_by_id : dict[str, ppkt.Phenopacket], optional
        Parsed phenopacket records indexed by individual ID. Contains structured metadata extracted from phenopackets, such as observed/excluded HPO terms and other relevant fields.
    """
    hpo_data: HpoFeatureData
    phenopackets: list[ppkt.Phenopacket] = field(default_factory=list)
    _phenopacket_by_id: dict[str, ppkt.Phenopacket] = field(init=False, repr=False, default_factory=dict)
    gpsea_cohort: Cohort | None = None
    _condition_cache: dict[str, pd.Series] = field(init=False, repr=False,default_factory=dict)
    _summary_cache: dict[str, pd.DataFrame] = field(init=False, repr=False, default_factory=dict)
    _variant_effect_by_individual: dict[str, list[tuple[str, tuple[str, ...]]]] | None = field(init=False,repr=False,default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.hpo_data, HpoFeatureData):
            raise TypeError("`hpo_data` must be an HpoFeatureData instance.")
        
        self._phenopacket_by_id = self._build_phenopacket_index(self.phenopackets)

    @property
    def individual_ids(self) -> pd.Index:
        return self.hpo_data.individual_ids

    @property
    def feature_ids(self) -> pd.Index:
        return self.hpo_data.feature_ids
    
    @staticmethod
    def _build_phenopacket_index(
        phenopackets: list[ppkt.Phenopacket],
    ) -> dict[str, ppkt.Phenopacket]:
        
        indexed: dict[str, ppkt.Phenopacket] = {}
        for phenopacket in phenopackets:

            individual_id = phenopacket.id
            indexed[individual_id] = phenopacket

        return indexed
    
    def describe_conditions(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        diseases_df = self.list_diseases()
        sex_df = self.describe_sex()
        genes_df = self.list_genes()
        variant_effects_df = None
        if self.gpsea_cohort is not None:
            variant_effects_df = self.variant_effect_summary()
        
        return diseases_df, sex_df, genes_df, variant_effects_df
        

    def list_diseases(self, *, use_cache: bool = True) -> pd.DataFrame:
        cache_key = "list_diseases"

        if use_cache and cache_key in self._summary_cache:
            return self._summary_cache[cache_key].copy()

        rows: dict[str, dict] = {}

        for phenopacket in self.phenopackets:
            seen_observed: set[str] = set()
            seen_excluded: set[str] = set()

            for disease in getattr(phenopacket, "diseases", []):
                term = getattr(disease, "term", None)
                disease_id = getattr(term, "id", None)

                if not disease_id:
                    continue

                label = getattr(term, "label", None)
                excluded = bool(getattr(disease, "excluded", False))

                if disease_id not in rows:
                    rows[disease_id] = {
                        "disease_id": disease_id,
                        "label": label,
                        "observed_n": 0,
                        "excluded_n": 0,
                    }

                if excluded:
                    seen_excluded.add(disease_id)
                else:
                    seen_observed.add(disease_id)

            for disease_id in seen_observed:
                rows[disease_id]["observed_n"] += 1

            for disease_id in seen_excluded - seen_observed:
                rows[disease_id]["excluded_n"] += 1

        df = pd.DataFrame(rows.values())

        if df.empty:
            df = pd.DataFrame(
                columns=["disease_id", "label", "observed_n", "excluded_n"]
            )
        else:
            df = df.sort_values(
                ["observed_n", "excluded_n", "disease_id"],
                ascending=[False, False, True],
            ).reset_index(drop=True)

        if use_cache:
            self._summary_cache[cache_key] = df.copy()

        return df

    def describe_sex(self, *, use_cache: bool = True) -> pd.DataFrame:
        cache_key = "describe_sex"

        if use_cache and cache_key in self._summary_cache:
            return self._summary_cache[cache_key].copy()

        sex_map = {
            1: "female",
            2: "male",
        }

        counts: dict[str, int] = {
            "female": 0,
            "male": 0,
            "unknown": 0,
        }

        for phenopacket in self.phenopackets:
            subject = getattr(phenopacket, "subject", None)
            if subject is None:
                counts["unknown"] += 1
                continue

            sex = sex_map.get(getattr(subject, "sex", None), "unknown")
            counts[sex] = counts.get(sex, 0) + 1

        df = pd.DataFrame(
            [{"sex": sex, "n_individuals": n} for sex, n in counts.items()]
        )

        if use_cache:
            self._summary_cache[cache_key] = df.copy()

        return df
    
    def list_genes(self, *, use_cache: bool = True) -> pd.DataFrame:
        cache_key = "list_genes"

        if use_cache and cache_key in self._summary_cache:
            return self._summary_cache[cache_key].copy()

        counts: dict[str, int] = {}

        for phenopacket in self.phenopackets:
            for gene_symbol in self._extract_gene_symbols(phenopacket):
                counts[gene_symbol] = counts.get(gene_symbol, 0) + 1

        df = pd.DataFrame(
            [
                {"gene_symbol": gene_symbol, "n_individuals": n}
                for gene_symbol, n in counts.items()
            ]
        )

        if df.empty:
            df = pd.DataFrame(columns=["gene_symbol", "n_individuals"])
        else:
            df = df.sort_values(
                ["n_individuals", "gene_symbol"],
                ascending=[False, True],
            ).reset_index(drop=True)

        if use_cache:
            self._summary_cache[cache_key] = df.copy()

        return df
    
    @staticmethod
    def _extract_gene_symbols(phenopacket: ppkt.Phenopacket) -> set[str]:
        genes: set[str] = set()

        for interpretation in getattr(phenopacket, "interpretations", []):
            diagnosis = getattr(interpretation, "diagnosis", None)
            if diagnosis is None:
                continue

            for genomic_interpretation in getattr(
                diagnosis,
                "genomic_interpretations",
                [],
            ):
                gene_descriptor = getattr(genomic_interpretation, "gene_descriptor", None)
                if gene_descriptor is not None:
                    symbol = getattr(gene_descriptor, "symbol", None)
                    if symbol:
                        genes.add(symbol)

                variant_interpretation = getattr(
                    genomic_interpretation,
                    "variant_interpretation",
                    None,
                )
                if variant_interpretation is None:
                    continue

                variation_descriptor = getattr(
                    variant_interpretation,
                    "variation_descriptor",
                    None,
                )
                gene_context = getattr(variation_descriptor, "gene_context", None)

                if gene_context is not None:
                    symbol = getattr(gene_context, "symbol", None)
                    if symbol:
                        genes.add(symbol)

        return genes
    
    def variant_effect_summary(
        self,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Summarize GPSEA variant effects by transcript.

        Rows:
            transcript IDs

        Columns:
            variant effects

        Values:
            number of variants with that effect on the transcript.
        """
        if self.gpsea_cohort is None:
            raise ValueError("No GPSEA cohort available.")

        cache_key = "variant_effect_summary"

        if use_cache and cache_key in self._summary_cache:
            return self._summary_cache[cache_key].copy()

        counters = self.gpsea_cohort.variant_effect_count_by_tx()

        summary = (
            pd.DataFrame.from_dict(counters, orient="index")
            .fillna(0)
            .astype(int)
        )

        if not summary.empty:
            summary = summary.sort_index()
            summary.index.name = "transcript_id"
            summary.columns.name = None

        if use_cache:
            self._summary_cache[cache_key] = summary.copy()

        return summary

    def get_condition(
        self,
        predicate: Callable[[ppkt.Phenopacket], bool | None],
        *,
        name: str | None = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Convert a phenopacket predicate into a binary condition vector.

        Returns
        -------
        pd.Series
            index = hpo_data.matrix.index
            values:
                1.0 = True
                0.0 = False
                NaN = unknown
        """

        if use_cache and name is not None and name in self._condition_cache:
            return self._condition_cache[name].copy()

        condition = pd.Series(
            data=np.nan,
            index=self.individual_ids,
            dtype="float64",
            name=name,
        )

        for individual_id in self.individual_ids:
            phenopacket = self._phenopacket_by_id[str(individual_id)]

            try:
                value = predicate(phenopacket)
            except Exception as exc:
                raise RuntimeError(
                    f"Condition predicate failed for individual "
                    f"{individual_id!r}."
                ) from exc

            if value is None:
                condition.loc[individual_id] = np.nan
            elif isinstance(value, (bool, np.bool_)):
                condition.loc[individual_id] = float(value)
            else:
                raise TypeError(
                    "Condition predicate must return bool or None. "
                    f"Got {type(value).__name__} for individual "
                    f"{individual_id!r}."
                )

        if use_cache and name is not None:
            self._condition_cache[name] = condition.copy()

        return condition
    
    def get_variant_effect_condition(
        self,
        *,
        transcript_id: str,
        variant_effect: VariantEffect,
    ) -> pd.Series:
        """
        Build an individual-level condition vector for a transcript-aware variant effect.

        Values
        ------
        1.0
            The individual has at least one variant with `effect` on `transcript_id`.

        0.0
            The individual has at least one variant annotated on `transcript_id`,
            but none has `effect`.

        NaN
            The individual has no variant annotation on `transcript_id`.

        Notes
        -----
        This is intentionally individual-level, unlike
        `variant_effect_summary()`, which summarizes variant counts.
        """
        if self.gpsea_cohort is None:
            raise ValueError(
                "No GPSEA cohort available. Variant effect condition requires "
                "a preprocessed GPSEA cohort."
            )

        condition_name = f"variant_effect:{transcript_id}:{variant_effect}"

        if condition_name in self._condition_cache:
            return self._condition_cache[condition_name].copy()

        condition = pd.Series(
            data=np.nan,
            index=self.individual_ids,
            dtype="float64",
            name=condition_name,
        )

        cohort_map = {str(p._labels.meta_label): p for p in self.gpsea_cohort.all_patients}

        for individual_id in self.individual_ids:
            patient = cohort_map.get(str(individual_id))

            if patient is None:
                condition.loc[individual_id] = np.nan
                continue

            saw_transcript = False
            has_effect = False

            for variant in patient.variants:
                for txa in variant.tx_annotations:
                    if str(txa.transcript_id) != transcript_id:
                        continue

                    saw_transcript = True

                    effects = {
                        variant_effect.name
                        for variant_effect in txa.variant_effects
                    }

                    if variant_effect.name in effects:
                        has_effect = True
                        break

                if has_effect:
                    break

            if has_effect:
                condition.loc[individual_id] = 1.0
            elif saw_transcript:
                condition.loc[individual_id] = 0.0
            else:
                condition.loc[individual_id] = np.nan

        if condition_name not in self._condition_cache:
            self._condition_cache[condition_name] = condition.copy()

        return condition

    def get_pmids(
        self, 
        individual_ids: Sequence[str]
    ) -> pd.Series:
        result: dict[str, list[str]] = {}

        for individual_id in individual_ids:
            key = str(individual_id)
            phenopacket = self._phenopacket_by_id.get(key)
            result[key] = [] if phenopacket is None else self._extract_pmids(phenopacket)

        return pd.Series(result, name="pmids", dtype="object")
    
    @staticmethod
    def _extract_pmids(phenopacket: ppkt.Phenopacket) -> list[str]:
        pmids: list[str] = []

        meta_data = getattr(phenopacket, "meta_data", None)
        if meta_data is None:
            return pmids

        for ref in getattr(meta_data, "external_references", []):
            ref_id = getattr(ref, "id", None)
            if isinstance(ref_id, str) and ref_id.startswith("PMID:"):
                pmids.append(ref_id.removeprefix("PMID:"))

        return sorted(set(pmids))
    

    
