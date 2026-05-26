import phenopackets as ppkt
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from gpsea.model import Patient, Cohort

import logging

from .features import HpoFeatureData

logger = logging.getLogger(__name__)


@dataclass
class PhenotypeDataset:
    """
    Unified dataset for phenotype-based analysis.

    This class provides individual-level access to phenotype data, 
    summary statistics, and GPSEA variant effects. It supports caching 
    for repeated queries and allows building binary conditions based 
    on predicates.

    Parameters
    ----------
    hpo_data : HpoFeatureData
        HPO feature data container with binary feature matrix, labels, 
        and optional relationship mask.
    phenopackets : list[ppkt.Phenopacket], optional
        Original phenopackets corresponding to individuals. Retained for 
        reference and downstream computations. Default is an empty list.
    gpsea_cohort : Cohort, optional
        Preprocessed GPSEA cohort object for variant-aware analyses.
        Default is None.
    """
    hpo_data: HpoFeatureData
    phenopackets: list[ppkt.Phenopacket] = field(default_factory=list)
    _phenopacket_by_id: dict[str, ppkt.Phenopacket] = field(init=False, repr=False, default_factory=dict)
    gpsea_cohort: Cohort | None = None
    _condition_cache: dict[str, pd.Series] = field(init=False, repr=False,default_factory=dict)
    _summary_cache: dict[str, pd.DataFrame] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.hpo_data, HpoFeatureData):
            raise TypeError("`hpo_data` must be an HpoFeatureData instance.")
        
        self._phenopacket_by_id = self._build_phenopacket_index(self.phenopackets)

    @property
    def individual_ids(self) -> pd.Index:
        """Individual IDs (matrix index)."""
        return self.hpo_data.individual_ids

    @property
    def feature_ids(self) -> pd.Index:
        """HPO feature IDs (matrix columns)."""
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
        """
        Generate summary tables describing key cohort-level conditions.

        This method provides an overview of diseases, sex distribution,
        gene annotations, and variant effects (if a GPSEA cohort is available).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]

            - diseases_df : pd.DataFrame
                Summary of observed and excluded diseases across individuals.
                Columns: 'disease_id', 'label', 'observed_n', 'excluded_n'.

            - sex_df : pd.DataFrame
                Summary of sex distribution across individuals.
                Columns: 'sex', 'n_individuals'.

            - genes_df : pd.DataFrame
                Summary of gene annotations across individuals.
                Columns: 'gene_symbol', 'n_individuals'.

            - variant_effects_df : pd.DataFrame or None
                Summary of GPSEA variant effects by transcript.
                Returns None if `gpsea_cohort` is not set.

        Notes
        -----
        - Each DataFrame is independent and can be used for downstream analysis.
        - Variant effects summary is only available when a GPSEA cohort has been provided.
        """
        diseases_df = self.list_diseases()
        sex_df = self.describe_sex()
        genes_df = self.list_genes()
        variant_effects_df = None
        if self.gpsea_cohort is not None:
            variant_effects_df = self.variant_effect_summary()
        
        return diseases_df, sex_df, genes_df, variant_effects_df
        

    def list_diseases(self, *, use_cache: bool = True) -> pd.DataFrame:
        """
        Summarize observed diseases across individuals.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use cached summary if available. Default is ``True``.

        Returns
        -------
        pd.DataFrame

            Columns:

            - disease_id :
                Disease identifier (e.g., OMIM or ORPHA ID).

            - label :
                Human-readable disease label.

            - n_individuals :
                Number and percentage of individuals with the disease.
        """
        cache_key = "list_diseases"

        if use_cache and cache_key in self._summary_cache:
            return self._summary_cache[cache_key].copy()

        total_individuals = len(self.phenopackets)

        disease_labels: dict[str, str] = {}
        observed_counts: dict[str, int] = {}

        for phenopacket in self.phenopackets:
            seen_observed_in_patient: set[str] = set()

            for disease in getattr(phenopacket, "diseases", []):
                term = getattr(disease, "term", None)
                disease_id = getattr(term, "id", None)

                if not disease_id:
                    continue

                if bool(getattr(disease, "excluded", False)):
                    continue

                if disease_id not in disease_labels:
                    disease_labels[disease_id] = getattr(term, "label", None) or "Unknown Label"

                seen_observed_in_patient.add(disease_id)

            for disease_id in seen_observed_in_patient:
                observed_counts[disease_id] = observed_counts.get(disease_id, 0) + 1

        df = pd.DataFrame(
            [
                {
                    "disease_id": dis_id, 
                    "label": disease_labels[dis_id], 
                    "observed_raw": count
                }
                for dis_id, count in observed_counts.items()
            ]
        )

        if df.empty:
            df = pd.DataFrame(columns=["disease_id", "label", "n_individuals"])
        else:
            df = df.sort_values(
                ["observed_raw", "disease_id"],
                ascending=[False, True]
            ).reset_index(drop=True)

            denom = total_individuals if total_individuals > 0 else 1
            percentage = (df["observed_raw"] / denom) * 100

            df["n_individuals"] = (
                df["observed_raw"].astype(str) +
                " (" + percentage.round(1).astype(str) + "%)"
            )

            df = df.drop(columns=["observed_raw"])
        return df.set_index("disease_id")


    def describe_sex(self, *, use_cache: bool = True) -> pd.DataFrame:
        """
        Summarize sex distribution across individuals.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use cached summary if available. Default is True.

        Returns
        -------
        pd.DataFrame

            Columns:

            - sex :
                One of ``female``, ``male``, or ``unknown``.

            - n_individuals :
                Count of individuals with this sex.
        """
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

        total_individuals = len(self.phenopackets)

        df = pd.DataFrame(
            [
                {"sex": sex, "n_individuals_raw": n} 
                for sex, n in counts.items()
            ]
        )

        denom = total_individuals if total_individuals > 0 else 1
        percentage = (df["n_individuals_raw"] / denom) * 100

        df["n_individuals"] = (
            df["n_individuals_raw"].astype(str) + 
            " (" + percentage.round(1).astype(str) + "%)"
        )

        df = df.drop(columns=["n_individuals_raw"])

        return df.set_index("sex")
    

    def list_genes(self, *, use_cache: bool = True) -> pd.DataFrame:
        """
        List gene symbols annotated in the cohort.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use cached summary if available. Default is ``True``.

        Returns
        -------
        pd.DataFrame

            Columns:

            - gene_symbol :
                HGNC gene symbol.

            - n_individuals :
                Number and percentage of individuals carrying variants in the gene.
        """
        cache_key = "list_genes"

        if use_cache and cache_key in self._summary_cache:
            return self._summary_cache[cache_key].copy()

        total_individuals = len(self.phenopackets)

        counts: dict[str, int] = {}

        for phenopacket in self.phenopackets:
            unique_genes = set(self._extract_gene_symbols(phenopacket))
            for gene_symbol in unique_genes:
                counts[gene_symbol] = counts.get(gene_symbol, 0) + 1

        df = pd.DataFrame(
            [
                {"gene_symbol": gene_symbol, "n_individuals_raw": n}
                for gene_symbol, n in counts.items()
            ]
        )

        if df.empty:
            df = pd.DataFrame(columns=["gene_symbol", "n_individuals"])
        else:
            df = df.sort_values(
                ["n_individuals_raw", "gene_symbol"],
                ascending=[False, True],
            ).reset_index(drop=True)

            denom = total_individuals if total_individuals > 0 else 1
            percentage = (df["n_individuals_raw"] / denom) * 100

            df["n_individuals"] = (
                df["n_individuals_raw"].astype(str) + " (" + percentage.round(1).astype(str) + "%)"
            )

            df = df.drop(columns=["n_individuals_raw"])

        return df.set_index("gene_symbol")
    
    
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

        Calculates variant effect distributions for each transcript.
        Returns a transposed matrix where each cell contains both
        absolute counts and percentages.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use a cached summary if available. Default is ``True``.

        Returns
        -------
        pd.DataFrame

            Rows:
                Variant effect types (e.g., ``MISSENSE``, ``NONSENSE``)

            Columns:
                Transcript IDs

            Values:
                Strings formatted as ``"count (percentage%)"``

        Example:
            ``15 (75.0%)``

        Raises
        ------
        ValueError
            If no GPSEA cohort has been loaded.
        """
        if self.gpsea_cohort is None:
            raise ValueError("No GPSEA cohort available.")

        cache_key = "variant_effect_summary_fractional"

        if use_cache and cache_key in self._summary_cache:
            return self._summary_cache[cache_key].copy()

        counters = self.gpsea_cohort.variant_effect_count_by_tx()

        preferred_tx_ids = set()
        for patient in self.gpsea_cohort.all_patients:
            for variant in patient.variants:
                for txa in variant.tx_annotations:
                    if txa.is_preferred:
                        preferred_tx_ids.add(txa.transcript_id)

        counters = {tx_id: counters[tx_id] for tx_id in preferred_tx_ids if tx_id in counters}

        raw_df = (
            pd.DataFrame.from_dict(counters, orient="index")
            .fillna(0)
            .astype(int)
        )

        if not raw_df.empty:
            raw_df = raw_df.sort_index()
            
            total_per_tx = raw_df.sum(axis=1)
            percentage_df = raw_df.divide(total_per_tx, axis=0) * 100
            
            summary = raw_df.astype(str) + " (" + percentage_df.round(1).astype(str) + "%)"
            
            summary = summary.fillna("0 (0.0%)")
            
            summary = summary.T
            
            summary.index.name = "variant_effect"
            summary.columns.name = "transcript_id"
        else:
            summary = pd.DataFrame()

        return summary
    

    def get_condition(
        self,
        predicate: Callable[[ppkt.Phenopacket], bool | None],
        *,
        name: str | None = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Convert a phenopacket predicate into an individual-level binary condition.

        Parameters
        ----------
        predicate : Callable[[ppkt.Phenopacket], bool | None]
            Function that maps a phenopacket to True, False, or None (unknown).
        name : str | None, optional
            Name of the condition to store in cache. Default is None.
        use_cache : bool, optional
            Whether to use cached condition if available. Default is True.

        Returns
        -------
        pd.Series
            Index: individual IDs
            Values:
            - 1.0 = True
            - 0.0 = False
            - NaN = unknown

        Raises
        ------
        TypeError
            If predicate returns a non-bool/non-None value.
        RuntimeError
            If predicate raises an exception for a specific individual.
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
    
    
    def get_variant_condition(
        self,
        predicate: Callable[[Patient], bool | None],
        *,
        condition_name: str | None = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Build a binary condition vector for transcript-aware variant effects.

        Parameters
        ----------
        predicate : Callable[[Patient], bool | None]
            Function that maps a GPSEA Patient to True, False, or None.
        condition_name : str | None, optional
            Name of the condition to store in cache. Default is None.
        use_cache : bool, optional
            Whether to use cached condition if available. Default is True.

        Returns
        -------
        pd.Series
            Index: individual IDs
            Values:
            - 1.0 : at least one variant with the effect
            - 0.0 : variant present but effect absent
            - NaN : no variant annotation or unknown

        Raises
        ------
        ValueError
            If GPSEA cohort is not available.
        """
        if self.gpsea_cohort is None:
            raise ValueError(
                "No GPSEA cohort available. Variant condition requires "
                "a preprocessed GPSEA cohort."
            )


        if use_cache and condition_name in self._condition_cache:
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

            value = predicate(patient)

            if value is None:
                condition.loc[individual_id] = np.nan
            else:
                condition.loc[individual_id] = float(value)

        if use_cache and condition_name not in self._condition_cache:
            self._condition_cache[condition_name] = condition.copy()

        return condition
    

    def get_pmids(
        self, 
        individual_ids: Sequence[str]
    ) -> pd.Series:
        """
        Retrieve PubMed IDs associated with a list of individuals.

        Parameters
        ----------
        individual_ids : Sequence[str]
            List of individual IDs.

        Returns
        -------
        pd.Series
            Index: individual IDs
            Values: list of PMIDs as strings. Empty list if no PMIDs found.
        """
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