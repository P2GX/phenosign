import logging
from os import path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.stats
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, triu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ..core import PhenotypeDataset
from .correlation_type import CorrelationType

logger = logging.getLogger(__name__)

class HPOCorrelationAnalyzer:

    """
    Analyze pairwise correlations between HPO terms.

    This class computes and visualizes pairwise correlations between HPO
    features using Spearman, Kendall, or Phi coefficients.

    Notes
    -----
    - Input HPO matrices are expected to use ``1`` for observed terms,
      ``0`` for excluded terms, and ``NaN`` for unknown terms.
    - Correlations are evaluated only for pairs with sufficient numbers of
      valid individuals.
    """
    def __init__(
        self,  
        dataset: PhenotypeDataset, 
        min_individuals_for_correlation_test: int = 20,
        min_cooccurrence_count = 1
    ):
        """
        Parameters
        ----------
        dataset : PhenotypeDataset
            Dataset containing HPO feature data and metadata.
        min_individuals_for_correlation_test : int, default=20
            Minimum number of valid individuals required to evaluate a
            pairwise correlation.
        min_cooccurrence_count : int, default=1
            Minimum number of ``0/0`` observations required for a feature
            pair to be considered valid for correlation testing.
        """
        if not isinstance(dataset, PhenotypeDataset):
            raise TypeError("`dataset` must be a `PhenotypeDataset` instance.")
        self.dataset= dataset
        self.hpo_matrix = self.dataset.hpo_data.matrix
        self.hpo_terms = self.hpo_matrix.columns
        self.n_features = self.hpo_matrix.shape[1]
        self.label_mapping = self.dataset.hpo_data.label_mapping 

        relationship_mask_df = self.dataset.hpo_data.relationship_mask

        if relationship_mask_df is not None:
            self.relationship_mask = relationship_mask_df.to_numpy(copy=True)
        else:
            logger.warning("No relationship_mask provided. All feature pairs will be evaluated for synergy.")
            self.relationship_mask = np.zeros((self.n_features, self.n_features))
            np.fill_diagonal(self.relationship_mask, np.nan)    
        
        self.min_individuals_for_correlation_test = min_individuals_for_correlation_test
        self.min_coccurrence_count = min_cooccurrence_count

        self._correlation_computed: bool = False

    def _calculate_stats( 
        self,
        observed_status_A: np.ndarray, 
        observed_status_B: np.ndarray,
        correlation_type: CorrelationType = CorrelationType.SPEARMAN,
    ) -> tuple[float, float]:
        """
        Compute a correlation coefficient and p-value for two binary vectors.

        Parameters
        ----------
        observed_status_a : np.ndarray
            Binary values for the first variable.
        observed_status_b : np.ndarray
            Binary values for the second variable.
        correlation_type : CorrelationType, default=CorrelationType.SPEARMAN
            Correlation metric to compute.

        Returns
        -------
        tuple[float, float]
            Correlation coefficient and p-value.

        Raises
        ------
        ValueError
            If the correlation type is not supported.
        """
        if correlation_type == CorrelationType.SPEARMAN:
            coef, pval = scipy.stats.spearmanr(observed_status_A, observed_status_B)
            return coef, pval

        elif correlation_type == CorrelationType.KENDALL:
            coef, pval = scipy.stats.kendalltau(observed_status_A, observed_status_B)
            return coef, pval

        elif correlation_type == CorrelationType.PHI:
            confusion_matrix = pd.crosstab(observed_status_A, observed_status_B, dropna=False)
            try:
                chi2, p, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
                n = confusion_matrix.sum().sum()
                phi = np.sqrt(chi2 / n)
                return phi, p
            except ValueError:
                return np.nan, np.nan

        else:
            raise ValueError(f"Unsupported CorrelationType '{correlation_type!r}'.")

    def _calculate_pairwise_correlation(
        self,
        col_a: int,
        col_b: int, 
        correlation_type: CorrelationType = CorrelationType.SPEARMAN,
        include_pmids: bool = True
    ) -> tuple[int, int, float, float, dict]:
        """
        Compute the correlation between two HPO term columns.

        Parameters
        ----------
        col_a : int
            Index of the first HPO term column.
        col_b : int
            Index of the second HPO term column.
        correlation_type : CorrelationType, default=CorrelationType.SPEARMAN
            Correlation metric to compute.
        include_pmids : bool, default=True
            If ``True``, aggregate PMIDs from contributing individuals and
            include them in the result table.


        Returns
        -------
        tuple[int, int, float, float, dict]
            Column indices, coefficient, p-value, and contingency counts.
        """
        
        matrix = self.hpo_matrix.values
        mask = (~np.isnan(matrix[:, col_a])) & (~np.isnan(matrix[:, col_b]))
        col_a_values = matrix[mask, col_a]
        col_b_values = matrix[mask, col_b]

        count_11 = np.sum((col_a_values == 1) & (col_b_values == 1))
        count_10 = np.sum((col_a_values == 1) & (col_b_values == 0))
        count_01 = np.sum((col_a_values == 0) & (col_b_values == 1))
        count_00 = np.sum((col_a_values == 0) & (col_b_values == 0))
        total = len(col_a_values)

        empty_counts = {
            "00": 0,
            "01": 0,
            "10": 0,
            "11": 0,
            "N": 0,
            "n_pmid": np.nan,
            "pmids": [],
        }
        
        if np.all(col_a_values == col_a_values[0]) or np.all(col_b_values == col_b_values[0]):
            return (col_a, col_b, np.nan, np.nan, empty_counts)
        
        excluded_excluded = np.sum((col_a_values == 0) & (col_b_values == 0))
        if excluded_excluded <= self.min_coccurrence_count:
            return (col_a, col_b, np.nan, np.nan, empty_counts)

        try:
            coef, p_val = self._calculate_stats(col_a_values, col_b_values, correlation_type=correlation_type)
            individual_ids = self.hpo_matrix.index[mask]
            if include_pmids:
                pmids_list = self.dataset.get_pmids(individual_ids).to_numpy()
                all_pmids = sorted(
                    {
                        str(pmid)
                        for pmids in pmids_list
                        if pmids is not None
                        for pmid in pmids
                        if pd.notna(pmid)
                    }
                )
                n_pmids = len(all_pmids)
            else:
                all_pmids = []
                n_pmids = np.nan

            return (col_a, col_b, coef, p_val, {
                "00": count_00,
                "01": count_01,
                "10": count_10,
                "11": count_11,
                "N": total,
                "n_pmid": n_pmids,
                "pmids": all_pmids,
            })
        except Exception as e:
            logger.error(
                "Error calculating correlation for columns %d and %d: %s",
                col_a,
                col_b,
                e,
            )
            return col_a, col_b, np.nan, np.nan, empty_counts

    def compute_correlation_matrix(
        self, 
        correlation_type: CorrelationType = CorrelationType.SPEARMAN, 
        n_jobs: int = -1,
        include_pmids: bool = True
    ) -> pd.DataFrame:
        """
        Compute pairwise correlations between HPO terms.

        Parameters
        ----------
        correlation_type : CorrelationType, default=CorrelationType.SPEARMAN
            Correlation metric to compute.
        n_jobs : int, default=-1
            Number of parallel jobs. ``-1`` uses all available CPUs.
        include_pmids : bool, default=True
            If ``True``, aggregate PMIDs from contributing individuals and
            include them in the result table.


        Returns
        -------
        pd.DataFrame
            Table of pairwise correlation results.
        """
        if not isinstance(correlation_type, CorrelationType):
            raise ValueError(
                f"`correlation_type` must be a `CorrelationType`, got {type(correlation_type).__name__}."
            )
        self.correlation_type = correlation_type
        x = self.hpo_matrix.to_numpy()

        has_one = np.any(x == 1)
        has_zero = np.any(x == 0)

        if not has_one or not has_zero:
            raise ValueError(
                "HPO matrix lacks sufficient variation for correlation analysis.\n"
                f"Detected values: "
                f"{'1 present, ' if has_one else 'no 1, '}"
                f"{'0 present' if has_zero else 'no 0'}.\n"
                "At least one observed (1) and one excluded (0) value are required.\n"
                "Please check your preprocessing (e.g., missing exclusion annotations)."
            )

        mask = ~np.isnan(x)  
        valid_counts = mask.T.astype(int) @ mask.astype(int)
        valid_counts_sparse = triu(coo_matrix(valid_counts), k=1)
        rows, cols, counts = (
            valid_counts_sparse.row, 
            valid_counts_sparse.col, 
            valid_counts_sparse.data
        )

        ontology_values = self.relationship_mask[rows, cols]
        ontology_candidate = ~np.isnan(ontology_values)
        candidate_idx = np.where(ontology_candidate & (counts >= self.min_individuals_for_correlation_test))[0]

        rows_cand, cols_cand = rows[candidate_idx], cols[candidate_idx]
        pairs = list(zip(rows_cand, cols_cand))

        if len(pairs) == 0:
            logger.warning(
                "No valid HPO term pairs remain after pre-filtering for correlation analysis.\n"
                "This occurs before pairwise computation and usually indicates that no feature pairs\n"
                "passed the initial candidate selection.\n\n"
                "Possible reasons:\n"
                f"- too few valid individuals per pair (min required = {self.min_individuals_for_correlation_test})\n"
                "- ontology relationship masking removed most pairs (ancestor/descendant/self)\n"
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._calculate_pairwise_correlation)(i, j, correlation_type=correlation_type, include_pmids=include_pmids)
            for i, j in tqdm(pairs, desc="Calculating pairwise correlation")
        )
        
        coef_matrix = np.full((self.n_features, self.n_features), np.nan)
        pvalue_matrix = np.full((self.n_features, self.n_features), np.nan)

        rows = []
        for r in results:
            i, j, coef, pval, counts = r
            coef_matrix[i, j] = coef
            coef_matrix[j, i] = coef
            pvalue_matrix[i, j] = pval
            pvalue_matrix[j, i] = pval

            hpo_a, hpo_b = self.hpo_terms[i], self.hpo_terms[j]
            if j > i:  
                if not np.isnan(coef):
                    row_data = {
                    "HPO_A": hpo_a,
                    **({"HPO_A_label": self.label_mapping.get(hpo_a)} if self.label_mapping.get(hpo_a) else {}),
                    "HPO_B": hpo_b,
                    **({"HPO_B_label": self.label_mapping.get(hpo_b)} if self.label_mapping.get(hpo_b) else {}),
                    "correlation": coef,
                    "p_value": pval,
                    "Count_00": counts["00"],
                    "Count_01": counts["01"],
                    "Count_10": counts["10"],
                    "Count_11": counts["11"],
                    "n_individuals": counts["N"],
                    }
                    if include_pmids:
                        row_data["n_pmids"] = counts["n_pmid"]
                        row_data["pmids"] = ";".join(counts.get("pmids", []))
                    rows.append(row_data)

        valid_mask = ~(np.isnan(coef_matrix).all(axis=0)) 
        if len(valid_mask) == 0:
            logger.warning("No valid correlations were found between HPO terms. "
                            "The resulting correlation matrix will be empty.\n"
                            "Possible reasons include:\n"
                            "- Filtering thresholds are too strict (co-occurrence constraints)\n"
                            "- HPO terms are constant (no variability across individuals)\n")
                            
        filtered_columns = self.hpo_terms[valid_mask]
        self.coef_df = pd.DataFrame(coef_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)
        self.pval_df = pd.DataFrame(pvalue_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)

        self.correlation_results = pd.DataFrame(rows)
        if not self.correlation_results.empty:
            pvals = self.correlation_results["p_value"].values
            _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
            loc = int(self.correlation_results.columns.get_loc("p_value"))
            self.correlation_results.insert(
                loc + 1,
                "p_value_corrected",
                pvals_corrected
            )
            self.correlation_results.sort_values(by="p_value", ascending=True, inplace=True)

        self._correlation_computed = True

        return self.correlation_results
    
    def save_correlation_results(
        self, 
        corr_threshold: float = 0.1,
        adj_pval_threshold: float = 0.3,
        output_file: str="correlation_results.csv"
    ) -> None:
        """
        Save correlation results to a CSV or Excel file.

        Parameters
        ----------
        corr_threshold : float, default=0.0
            Minimum correlation coefficient to retain.
        adj_pval_threshold : float, default=0.3
            Maximum adjusted p-value to retain.
        output_file : str, default="correlation_results.csv"
            Output file path. Supported formats are ``.csv`` and ``.xlsx``.

        Raises
        ------
        ValueError
            If correlation results have not been computed or if thresholds
            are invalid.
        """
        if not self._correlation_computed:
            raise ValueError("Correlation results not computed. Run compute_correlation_matrix() first.")

        if corr_threshold < 0.0 or corr_threshold > 1.0:
            raise ValueError("abs_threshold must be between 0.0 and 1.0")
        df = df[df["correlation"].abs() >= corr_threshold]

        if adj_pval_threshold < 0.0 or adj_pval_threshold > 1.0:
            raise ValueError("adj_pval_threshold must be between 0.0 and 1.0")
        df = df[df["p_value_corrected"] < adj_pval_threshold]

        ext = path.splitext(output_file)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise ValueError(f"Unsupported file format: {ext}. Use '.csv' or '.xlsx'.")
  
        if ext == ".csv":
            df.to_csv(output_file, index=False)
        else:
            df.to_excel(output_file, index=False)    

    
    def filter_weak_correlations(
        self, 
        corr_threshold: float = 0.1,
        adj_pval_threshold: float = 0.3
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter the correlation and p-value matrices by effect size and significance.

        Parameters
        ----------
        corr_threshold : float, default=0.0
            Minimum correlation coefficient to retain.
        adj_pval_threshold : float, default=0.1
            Maximum adjusted p-value to retain.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Filtered correlation matrix and filtered p-value matrix.
        """
        if not self._correlation_computed:
            raise RuntimeError("Correlation matrix not found. Please run `compute_correlation_matrix()` first.")

        coef_matrix = self.coef_df.copy()
        p_value = self.pval_df.copy()

        if corr_threshold < 0.0 or corr_threshold > 1.0:
            raise ValueError("corr_threshold must be between 0.0 and 1.0")
        mask = coef_matrix.abs() < corr_threshold
        coef_matrix[mask] = np.nan
        p_value[mask] = np.nan

        if adj_pval_threshold < 0.0 or adj_pval_threshold > 1.0:
            raise ValueError("adj_pval_threshold must be between 0.0 and 1.0")
        
        non_signif = self.correlation_results.loc[
            (self.correlation_results["p_value_corrected"] >= adj_pval_threshold),
            ["HPO_A", "HPO_B"]
        ]
        for _, row in non_signif.iterrows():
            hpo_a, hpo_b = row["HPO_A"], row["HPO_B"]
            if hpo_a in coef_matrix.index and hpo_b in coef_matrix.columns:
                coef_matrix.loc[hpo_a, hpo_b] = np.nan
                coef_matrix.loc[hpo_b, hpo_a] = np.nan
                p_value.loc[hpo_a, hpo_b] = np.nan
                p_value.loc[hpo_b, hpo_a] = np.nan
    
        mask_rows = coef_matrix.isna().all(axis=1)
        mask_cols = coef_matrix.isna().all(axis=0)
        coef_matrix_cleaned = coef_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return coef_matrix_cleaned, p_value_cleaned
    
    def _format_hpo_pair(
        self, 
        hpo_id: str, 
        label: str | None
    ) -> str:
        """Format an HPO term for display."""
        if label:
            return f"{label} ({hpo_id})"
        return hpo_id
    
    def _format_pmids_for_tooltip(
        self,
        pmids: str | list[str] | None,
        max_pmids: int = 5,
    ) -> str:
        """Format PMID values for hover text."""
        if pmids is None or pmids == "":
            return "None"

        if isinstance(pmids, str):
            pmid_list = [p.strip() for p in pmids.split(";") if p.strip()]
        else:
            pmid_list = [str(p).strip() for p in pmids if str(p).strip()]

        if not pmid_list:
            return "None"

        if len(pmid_list) <= max_pmids:
            return ", ".join(pmid_list)

        shown = ", ".join(pmid_list[:max_pmids])
        remaining = len(pmid_list) - max_pmids
        return f"{shown} ... (+{remaining} more)"

    def plot_correlation_heatmap_with_significance(
        self,
        corr_threshold: float = 0.1,
        adj_pval_threshold: float = 0.3,
        title_name: str | None = None,
    ) -> go.Figure:
        """
        Plot a correlation heatmap with statistical filtering.

        Parameters
        ----------
        corr_threshold : float, default=0.0
            Minimum correlation coefficient to display.
        adj_pval_threshold : float, default=0.1
            Maximum adjusted p-value to display.
        title_name : str, optional
            Optional subtitle.

        Returns
        -------
        go.Figure
            Plotly heatmap figure.

        Example:
            >>> # Compute correlations first
            >>> analyzer.compute_correlation_matrix()
            >>> # Generate heatmap (returns a Plotly Figure)
            >>> fig = analyzer.plot_correlation_heatmap_with_significance(
            ...     stats_name="spearman",
            ...     corr_threshold=0.55,
            ...     adj_pval_threshold=0.1,
            ...     title_name="Cohort A"
            ... )
            >>> # Show in Jupyter or browser
            >>> fig.show()
        """
        coef_matrix, pval_matrix = self.filter_weak_correlations(
            corr_threshold=corr_threshold,
            adj_pval_threshold=adj_pval_threshold
        )

        if coef_matrix.empty or np.isnan(coef_matrix.values).all():
            raise ValueError(
                "The coefficient matrix is empty after filtering. "
                "Try adjusting `corr_threshold` or `adj_pval_threshold`."
            )

        raw_coef_df = self.coef_df.loc[coef_matrix.index, coef_matrix.columns]

        relationship_mask_df = pd.DataFrame(
            self.relationship_mask,
            index=self.hpo_terms,
            columns=self.hpo_terms,
        )

        relationship_mask_df = relationship_mask_df.loc[coef_matrix.index, coef_matrix.columns]

        status_matrix = pd.DataFrame(
            "hidden_upper_triangle",
            index=coef_matrix.index,
            columns=coef_matrix.columns
        )

        relationship_isna = relationship_mask_df.isna()
        relationship_notna = ~relationship_isna

        status_matrix[relationship_isna] = "ontology_related"
        status_matrix[raw_coef_df.isna() & relationship_notna] = "invalid_computation"
        status_matrix[raw_coef_df.notna() & coef_matrix.isna()] = "filtered_by_statistics"

        n_rows, n_cols = coef_matrix.shape
        cell_size = 60  # Base pixel size per cell

        max_dim = max(n_rows, n_cols)
        fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

        title_fontsize = max(14 + max_dim // 2, 28)
        label_fontsize = max(8, 12 - max_dim // 8)
        annot_fontsize = max(6, 12 - max_dim // 8)

        triangle_mask = pd.DataFrame(
            np.tril(np.ones(coef_matrix.shape, dtype=bool), k=0),
            index=coef_matrix.index,
            columns=coef_matrix.columns
        )
        coef_matrix = coef_matrix.where(triangle_mask)
        pval_matrix = pval_matrix.where(triangle_mask)
        display_matrix = coef_matrix.where(triangle_mask)

        nan_bg = pd.DataFrame(np.nan, index=coef_matrix.index, columns=coef_matrix.columns)
        nan_bg[triangle_mask & coef_matrix.isna()] = 2

        text_matrix = np.where(
            np.isnan(coef_matrix.values),
            "",
            coef_matrix.round(2).astype(str)
        )

        counts_lookup = {}
        for row in self.correlation_results.itertuples():
            has_pmids = hasattr(row, "n_pmids")

            forward = {
                "Coefficient": row.correlation,
                "P_value": row.p_value,
                "P_value_corrected": row.p_value_corrected,
                "Count_00": row.Count_00,
                "Count_01": row.Count_01,
                "Count_10": row.Count_10,
                "Count_11": row.Count_11,
                "n_individuals": row.n_individuals,
            }

            backward = {
                "Coefficient": row.correlation,
                "P_value": row.p_value,
                "P_value_corrected": row.p_value_corrected,
                "Count_00": row.Count_00,
                "Count_01": row.Count_10,  # swapped
                "Count_10": row.Count_01,  # swapped
                "Count_11": row.Count_11,
                "n_individuals": row.n_individuals,
            }

            if has_pmids:
                forward["n_pmids"] = row.n_pmids
                forward["pmids"] = getattr(row, "pmids", "")
                backward["n_pmids"] = row.n_pmids
                backward["pmids"] = getattr(row, "pmids", "")

            counts_lookup[(row.HPO_A, row.HPO_B)] = forward
            counts_lookup[(row.HPO_B, row.HPO_A)] = backward

        hover_text = []
        for i, row in enumerate(coef_matrix.index):
            hover_row = []
            for j, col in enumerate(coef_matrix.columns):
                coef = coef_matrix.iloc[i, j]
                pval = pval_matrix.iloc[i, j]

                display_row = self._format_hpo_pair(row, self.label_mapping.get(row))
                display_col = self._format_hpo_pair(col, self.label_mapping.get(col))

                if not triangle_mask.iloc[i, j]:
                    hover_row.append("")
                elif np.isnan(coef):
                    status = status_matrix.iloc[i, j]
                    reason_map = {
                        "ontology_related": "these two HPO terms are ontologically related (ancestor/descendant/self).",
                        "invalid_computation": "the correlation could not be computed for this HPO pair.",
                        "filtered_by_statistics": (
                            f"the correlation did not pass the statistical filters "
                            f"(|corr| >= {corr_threshold} and adjusted p-value < {adj_pval_threshold})."
                        ),
                        "hidden_upper_triangle": "only the lower-left triangle is displayed.",
                    }

                    reason_text = reason_map.get(status, f"{status}")

                    hover_row.append(
                        f"<b>HPO_A</b>: {display_col}<br>"
                        f"<b>HPO_B</b>: {display_row}<br>"
                        f"<b>Not shown due to</b>: {reason_text}"
                    )
                else:
                    counts = counts_lookup.get((row, col), {})
                    has_pmids = "n_pmids" in counts
                    if not has_pmids:
                        pmid_block = ""
                    else:
                        pmid_text = self._format_pmids_for_tooltip(
                            counts.get("pmids", ""),
                            max_pmids=4,
                        )
                        pmid_block = (
                            f"<b>N_PMIDs</b>: {int(counts.get('n_pmids', 0))}<br>"
                            f"<b>PMIDs</b>: {pmid_text}"
                        )
                    hover_row.append(
                        f"<b>HPO_A</b>: {display_col}<br><b>HPO_B</b>: {display_row}<br>"
                        f"<b>Corr</b>: {coef:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>p-val_corrected</b>: {counts.get('P_value_corrected', np.nan):.6f}<br>"
                        f"<b>Counts(A/B): E/E</b>: {counts.get('Count_00', 0)}, "
                        f"<b>E/O</b>: {counts.get('Count_01', 0)}, "
                        f"<b>O/E</b>: {counts.get('Count_10', 0)}, "
                        f"<b>O/O</b>: {counts.get('Count_11', 0)}<br>"
                        f"<b>Total_individuals</b>: {counts.get('n_individuals', 0)}<br>"
                        f"{pmid_block}"
                    )
            hover_text.append(hover_row)
          
        coef_matrix.rename(index=self.label_mapping, columns=self.label_mapping, inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=nan_bg.values,
            x=coef_matrix.columns,
            y=coef_matrix.index,
            colorscale=[[0, "#dbe7f3"], [1, "#dbe7f3"]],
            showscale=False,
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        ))
        fig.add_trace(go.Heatmap(
                z=display_matrix.values,
                x=coef_matrix.columns,
                y=coef_matrix.index,
                colorscale="Tealgrn",
                zmin=-1,
                zmax=1,
                zmid=0,
                text=text_matrix,
                texttemplate=f"<span style='font-size:{annot_fontsize}px'>%{{text}}</span>",
                hovertext=hover_text,
                hoverinfo="text",
                colorbar=dict(title="Corr.", len=0.8, thickness=title_fontsize),
                xgap=1,
                ygap=1,
            ))
        
        max_ylabel_len = max(len(str(lbl)) for lbl in coef_matrix.index)
        left_margin = 60 + max_ylabel_len * label_fontsize

        clean_subtitle = title_name.strip() if title_name and title_name.strip() else ""

        main_title = f"<b>{self.correlation_type.name} Correlation</b>"
        
        full_title = f"{main_title}<br><span style='font-size:0.8em'>{clean_subtitle}</span>" if clean_subtitle else main_title

        fig.update_layout(
            title=dict(
                text=full_title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=min(title_fontsize, 24),
                    family="Arial"
                )
            ),
            xaxis=dict(
                tickangle=90,
                tickfont=dict(size=label_fontsize),
            ),
            yaxis=dict(
                tickfont=dict(size=label_fontsize),
                scaleanchor="x",
                scaleratio=1
            ),
            width=fig_size + left_margin,
            height=fig_size + left_margin,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        fig.update_yaxes(autorange="reversed")

        return fig
    

    def save_correlation_heatmap(
        self, 
        fig: go.Figure, 
        output_file: str
    ) -> None:
        """
        Save a correlation heatmap as an HTML file.

        Parameters
        ----------
        fig : go.Figure
            Heatmap figure.
        output_file : str
            Output HTML file path.
        """
        if not output_file.endswith(".html"):
            raise ValueError("output_file must have a '.html' extension")
        fig.write_html(output_file)