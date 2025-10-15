import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple,Optional
from joblib import Parallel, delayed
import scipy.stats
from .correlation_type import CorrelationType
import plotly.graph_objs as go
import logging
from os import path
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.sparse import coo_matrix, triu
from itertools import chain


logger = logging.getLogger(__name__)

class HPOStatisticsAnalyzer:

    """
    Analyze pairwise statistical relationships between HPO terms and disease-related targets in a cohort.

    This class provides tools to compute and visualize correlations between HPO terms and disease status 
    (or variant effect matrices), using a variety of statistical tests such as Spearman, Kendall, or Phi coefficient.
    
    It supports filtering weak correlations and highlighting statistically significant relationships in a heatmap.

    Example:
        from ppkt2synergy import CohortDataLoader, HPOStatisticsAnalyzer
        >>> phenopackets = CohortDataLoader.from_ppkt_store('FBN1')
        >>> hpo_matrix, target_matrix = PhenopacketMatrixProcessor.prepare_hpo_data(
                phenopackets, threshold=0.5, mode='leaf', use_label=True)
        >>> analyzer =HPOStatisticsAnalyzer(hpo_matrix, min_individuals_for_correlation_test=40)
        >>> coef_matrix, pval_matrix = analyzer.compute_correlation_matrices("Spearman")
        >>> analyzer.plot_correlation_heatmap_with_significance("Spearman")
    
    Notes:
        - Requires at least 30 valid data points per pairwise comparison.
        - Assumes binary input matrices (0/1 presence/absence format).
    """
    def __init__(
            self,  
            hpo_data: Tuple[pd.DataFrame,pd.DataFrame,Optional[pd.DataFrame]], 
            min_individuals_for_correlation_test: int = 30,
            min_cooccurrence_count = 1
        ):
            """
            Initialize the HPOStatisticsAnalyzer.

            Args:
            hpo_data(Tuple[pd.DataFrame,Optional[pd.DataFrame]]):
            - Feature matrix of shape (n_samples, n_features): 
                Non-NaN values must be 0 or 1. DataFrame inputs will be converted to a NumPy array.
            - relationship_mask (n_features, n_features):
                Optional 2D array (n_features x n_features) indicating valid feature pairs to evaluate.
                Can be used to skip predefined pairs (e.g. based on HPO hierarchy or previous results).
                If provided, it will be converted to a NumPy array and used to initialize the synergy matrix.
            min_individuals_for_correlation_test(int): (default: 30)
                Minimum number of valid individuals required to perform correlation tests.
            min_cooccurrence_count (int, default=1):
                Minimum number of co-occurrences (both features present, 1/1) 
                **and** co-exclusions (both features absent, 0/0) required 
                for a feature pair to be considered valid for correlation testing.
                This ensures that both positive and negative concordance are observed 
                more than once, avoiding spurious correlations.

            Raises:
                ValueError: If min_individuals_for_correlation_test is less than 30.
            """
            if isinstance(hpo_data, tuple):
                hpo_matrix, relationship_mask, pmids_matrix = hpo_data
            else:
                raise TypeError("hpo_data must be a tuple of (hpo_matrix, relationship_mask)")
            if isinstance(hpo_matrix, pd.DataFrame):
                self.hpo_matrix = hpo_matrix
                self.hpo_terms = hpo_matrix.columns
                self.n_features = hpo_matrix.shape[1]
            else:
                raise TypeError("hpo_matrix must be a pandas DataFrame")
            
            if isinstance(pmids_matrix, pd.DataFrame):
                self.patient_pmids = pmids_matrix
            
            if not np.all(np.isin(hpo_matrix.to_numpy()[~np.isnan(hpo_matrix.to_numpy())], [0, 1])):
                raise ValueError("Non-NaN values in HPO Matrix must be either 0 or 1")
            
            self.relationship_mask = None
            if relationship_mask is not None:
                if isinstance(relationship_mask, pd.DataFrame):
                    self.relationship_mask = relationship_mask.to_numpy() 
                else:
                    raise ValueError("relationship_mask must be a pd.DataFrame")
                    
                if relationship_mask.shape[0] != relationship_mask.shape[1] or \
                    relationship_mask.shape[0] != hpo_matrix.shape[1]:
                    raise ValueError("relationship_mask must have the same number of rows and columns as hpo_matrix has features")
                
                if not np.all(np.isin(self.relationship_mask[~np.isnan(self.relationship_mask)], [0])):
                    raise ValueError("relationship_mask must contain only 0 or NaN")
            
            #if min_individuals_for_correlation_test < 30:
            #    raise ValueError("min_individuals_for_correlation_test must not be less than 30.")
            self.min_individuals_for_correlation_test = min_individuals_for_correlation_test
            self.min_coccurrence_count = min_cooccurrence_count

    def _calculate_pairwise_stats( 
            self,
            observed_status_A: np.ndarray, 
            observed_status_B: np.ndarray,
            correlation_type: CorrelationType = CorrelationType.SPEARMAN,
        ) -> Dict[str, Union[float, str]]:
            """
            Calculate selected statistical metric (spearman, kendall, or phi) and its p-value
            for two binary (0/1) observed status vectors.

            Args:
                observed_status_A(np.ndarray): 
                    Binary values (0/1) for the first variable.
                observed_status_B(np.ndarray): 
                    Binary values (0/1) for the second variable.
                correlation_type (CorrelationType): (default: CorrelationType.SPEARMAN)
                    Correlation metric to compute. One of:
                    - CorrelationType.SPEARMAN
                    - CorrelationType.KENDALL
                    - CorrelationType.PHI

            Returns:
                Dict[str, Union[float, str]]: 
                    A dictionary with the selected statistic and its p-value.

            Raises:
                ValueError: If the provided correlation_name is not supported.
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
                raise ValueError(f"Unsupported CorrelationType '{stats_type}'.")

    def _calculate_pairwise_correlation(
            self,
            col_A: int,
            col_B: int, 
            correlation_type: CorrelationType = CorrelationType.SPEARMAN,
        ) -> Dict[str, Union[float, str]]:
            """
            Perform correlation tests between two columns (HPO terms, diseases).

            Args:
                col_A(int): 
                    The first column to correlate.
                col_B(int): 
                    The second column to correlate.
                correlation_type (CorrelationType): (default: CorrelationType.SPEARMAN)
                    Correlation metric to compute. One of:
                    - CorrelationType.SPEARMAN
                    - CorrelationType.KENDALL
                    - CorrelationType.PHI

            Returns:
                Optional[Dict[str, Union[float, str]]]:
                    Dictionary with correlation results, or None if invalid or insufficient data.

            Raises:
                ValueError: If insufficient data for correlation test or invalid columns (all 0 or 1).
            """
            
            matrix = self.hpo_matrix.values
            mask = (~np.isnan(matrix[:, col_A])) & (~np.isnan(matrix[:, col_B]))
            col_A_values = matrix[mask, col_A]
            col_B_values = matrix[mask, col_B]

            count_11 = np.sum((col_A_values == 1) & (col_B_values == 1))
            count_10 = np.sum((col_A_values == 1) & (col_B_values == 0))
            count_01 = np.sum((col_A_values == 0) & (col_B_values == 1))
            count_00 = np.sum((col_A_values == 0) & (col_B_values == 0))
            total = len(col_A_values)
            
            if np.all(col_A_values == col_A_values[0]) or np.all(col_B_values == col_B_values[0]):
                return (col_A, col_B, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0,"n_pmid": np.nan})
            
            # --- Count co-occurrence ---
            observed_observed = np.sum((col_A_values == 1) & (col_B_values == 1))
            excluded_excluded = np.sum((col_A_values == 0) & (col_B_values == 0))

            if observed_observed <= self.min_coccurrence_count or excluded_excluded <= self.min_coccurrence_count:
                return (col_A, col_B, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0,"n_pmid": np.nan})

            try:
                coef, p_val = self._calculate_pairwise_stats(col_A_values, col_B_values, correlation_type=correlation_type)
                patient_ids = self.hpo_matrix.index[mask]
                pmids_list = self.patient_pmids.loc[patient_ids, 'pmids'].to_numpy()
                all_pmids = set(chain.from_iterable(pmids_list))
                n_pmids = len(all_pmids)
                return (col_A, col_B, coef, p_val, {"00":count_00,"01":count_01,"10":count_10,"11":count_11,"N":total,"n_pmid": n_pmids})
            except Exception as e:
                return (col_A, col_B, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0,"n_pmid": np.nan})

    def compute_correlation_matrix(
            self, 
            correlation_type: CorrelationType = CorrelationType.SPEARMAN, 
            n_jobs: int = -1,
        ) -> None:
        """
        Compute pairwise correlation coefficients and p-values between HPO terms.

        This function first identifies valid feature pairs (columns) that have 
        sufficient non-missing individuals using a sparse matrix pre-filtering 
        approach, then computes correlation statistics in parallel.

        Args:
            correlation_type (CorrelationType, optional):
                Correlation metric to compute. One of:
                - CorrelationType.SPEARMAN
                - CorrelationType.KENDALL
                - CorrelationType.PHI
                Default: CorrelationType.SPEARMAN.
            n_jobs (int, optional):
                Number of parallel jobs to use for pairwise correlation.
                -1 uses all available CPU cores. Default: -1.

        Returns:
            pd.DataFrame:
                A DataFrame containing pairwise correlation results.
                Each row corresponds to one valid pair of HPO terms and includes:
                    - HPO_A (str): Name of the first HPO term.
                    - HPO_B (str): Name of the second HPO term.
                    - coefficient (float): Correlation coefficient.
                    - p_value (float): Corresponding p-value (NaN if not applicable).
                    - p_value_corrected (float): P-value adjusted for multiple testing using the Benjamini–Hochberg FDR method.
                    - Count_00 (int): Number of individuals with (0,0).
                    - Count_01 (int): Number of individuals with (0,1).
                    - Count_10 (int): Number of individuals with (1,0).
                    - Count_11 (int): Number of individuals with (1,1).
                    - n_patients (int): Total number of valid individuals.
                    - n_pmids (int): Number of PubMed references associated with the pair (if available).

        Side Effects:
            - Stores the full correlation coefficient matrix in `self.coef_df`.
            - Stores the full p-value matrix in `self.pval_df`.
            - Stores the pairwise results table in `self.correlation_results`.
        """
        if not isinstance(correlation_type, CorrelationType):
            raise ValueError(f"stats_type must be a CorrelationType, got {type(correlation_type)}")
        
        columns = self.hpo_terms
        n_cols = len(columns)
        X = self.hpo_matrix.to_numpy()
        mask = ~np.isnan(X)  # True = non-NaN
        # Compute valid counts for each pair
        valid_counts = mask.T.astype(int) @ mask.astype(int)
        valid_counts_sparse = triu(coo_matrix(valid_counts), k=1)
        rows, cols, counts = valid_counts_sparse.row, valid_counts_sparse.col, valid_counts_sparse.data

        # Apply relationship mask if present
        if self.relationship_mask is not None:
            ontology_values = self.relationship_mask[rows, cols]
            ontology_candidate = ~np.isnan(ontology_values)
            candidate_idx = np.where(ontology_candidate & (counts >= self.min_individuals_for_correlation_test))[0]
        else:
            candidate_idx = np.where(counts >= self.min_individuals_for_correlation_test)[0]

        rows_cand, cols_cand = rows[candidate_idx], cols[candidate_idx]
        pairs = list(zip(rows_cand, cols_cand))

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._calculate_pairwise_correlation)(i, j, correlation_type=correlation_type)
            for i, j in tqdm(pairs, desc="Calculating pairwise correlation")
        )
        
        matrix = np.full((n_cols, n_cols), np.nan)
        pvalue_matrix = np.full((n_cols, n_cols), np.nan)

        rows = []
        for r in results:
            i, j, coef, pval, counts = r
            matrix[i, j] = coef
            matrix[j, i] = coef
            pvalue_matrix[i, j] = pval
            pvalue_matrix[j, i] = pval
            hpo1, hpo2 = columns[i], columns[j]
            if j > i:  # only upper triangle
                if not np.isnan(coef):
                    rows.append({
                        "HPO_A": hpo1,
                        "HPO_B": hpo2,
                        "correlation": coef,
                        "p_value": pval,
                        "Count_00": counts["00"],
                        "Count_01": counts["01"],
                        "Count_10": counts["10"],
                        "Count_11": counts["11"],
                        "n_patients": counts["N"],
                        "n_pmids": counts["n_pmid"]
                    }) 
   
        self.correlation_results = pd.DataFrame(rows)
        if not self.correlation_results.empty:
            pvals = self.correlation_results["p_value"].values
            _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
            self.correlation_results.insert(
                self.correlation_results.columns.get_loc("p_value") + 1, 
                "p_value_corrected",                                       
                pvals_corrected
            )
        
        valid_mask = ~(np.isnan(matrix).all(axis=0)) | (np.nan_to_num(matrix, nan=0).sum(axis=0) == 0)
        if len(valid_mask) == 0:
            logger.warning("Warning: No valid correlation between HPO terms. Correlation matrix will be empty.")
        
        filtered_columns = self.hpo_terms[valid_mask]
        self.coef_df = pd.DataFrame(matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)
        self.pval_df = pd.DataFrame(pvalue_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)

        return self.correlation_results
    
    def save_correlation_results(
            self, 
            lower_bound: float=-0.0, 
            upper_bound: float=0.0,
            alpha: float = 0.0,
            output_file: str="correlation_results.csv"
        ) -> None:
        """
        Export the computed correlation results to a file.

        The correlation matrices (`self.coef_df` and `self.pval_df`) must have 
        been computed previously by calling `compute_correlation_matrix`.

        Args:
            lower_bound (float):
                Minimum correlation value to include. 
            upper_bound (float):
                Maximum correlation value to include. 
            alpha (float):
                Significance threshold for p-values. Only correlations with p < alpha are kept.
            output_file (str):
                Path to the output file. Supported formats:
                - ".csv": saves as a CSV file.
                - ".xlsx" or other extensions: saves as an Excel file.

        Raises:
            ValueError:
                If `self.correlation_results` has not been initialized
                (i.e., `compute_correlation_matrix` has not been run).

        Example:
            >>> analyzer.compute_correlation_matrix()
            >>> analyzer.save_correlation_results("correlations.csv")
            >>> analyzer.save_correlation_results("correlations.xlsx")
        """
        
        if not hasattr(self, "correlation_results"):
            raise ValueError("Correlation results not computed. Run compute_correlation_matrix() first.")
        
        df = self.correlation_results.copy()

        if alpha > 0.0 and "p_value" in df.columns:
            df = df[df["p_value"] < alpha]

        if lower_bound < 0.0 and upper_bound >0.0:
            df = df[(df["correlation"] <= lower_bound) | (df["correlation"] >= upper_bound)]

        ext = path.splitext(output_file)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise ValueError(f"Unsupported file format: {ext}. Use '.csv' or '.xlsx'.")
  
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
        else:
            df.to_excel(output_file, index=False)    

    
    def filter_weak_correlations(
            self, 
            lower_bound: float=-0.55, 
            upper_bound: float=0.55,
            alpha: float = 0.05,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove weak correlations from the correlation matrix based on the given threshold.

        Args:
            stats_name(str): (default: "spearman") 
                The name of the statistic to calculate (e.g., "spearman", "kendall", "phi").
            lower_bound(float): (default: -0.55)
                The lower bound for filtering weak correlations.
            upper_bound(float): (default: 0.55)
                The upper bound for filtering weak correlations.
            alpha: float = 0.05,

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - A DataFrame of cleaned correlation coefficients.
                - A DataFrame of cleaned p-values.
        """
        if not hasattr(self, 'coef_df') or not hasattr(self, 'pval_df'):
            raise RuntimeError("Correlation matrix not found. Please run `compute_correlation_matrix()` first.")

        coef_matrix,p_value = self.coef_df.copy(), self.pval_df.copy()

        mask = (coef_matrix > lower_bound) & (coef_matrix < upper_bound)
        coef_matrix[mask] = np.nan
        p_value[mask] = np.nan

        non_signif = self.correlation_results.loc[
            self.correlation_results["p_value_corrected"] >= alpha, ["HPO_A", "HPO_B"]
        ]
        for _, row in non_signif.iterrows():
            hpo1, hpo2 = row["HPO_A"], row["HPO_B"]
            if hpo1 in coef_matrix.index and hpo2 in coef_matrix.columns:
                coef_matrix.loc[hpo1, hpo2] = np.nan
                coef_matrix.loc[hpo2, hpo1] = np.nan
                p_value.loc[hpo1, hpo2] = np.nan
                p_value.loc[hpo2, hpo1] = np.nan
    
        mask_rows = coef_matrix.isna().all(axis=1)
        mask_cols = coef_matrix.isna().all(axis=0)
        coef_matrix_cleaned = coef_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return coef_matrix_cleaned, p_value_cleaned
    

    def plot_correlation_heatmap_with_significance(
            self,
            stats_name: str = "spearman",
            lower_bound: float = -0.55,
            upper_bound: float = 0.55,
            alpha: float = 0.05,
            title_name: str = "",
        ) -> go.Figure:
        """
        Create an interactive Plotly heatmap showing correlation coefficients between features,
        with hover information for p-values.

        Parameters:
            stats_name (str): 
                Type of correlation coefficient to use ("spearman", "kendall", "phi").
            lower_bound (float): 
                Lower threshold to filter out weak correlations.
            upper_bound (float): 
                Upper threshold to filter out weak correlations.
            alpha (float): (default: 0.05)
                Significance threshold for P-value.
            title_name (str): 
                Optional subtitle to display under the main title.

        Returns:
            plotly.graph_objects.Figure:
                A Plotly Figure object for the correlation heatmap.

        Example:
            >>> # Compute correlations first
            >>> analyzer.compute_correlation_matrix()
            >>> # Generate heatmap (returns a Plotly Figure)
            >>> fig = analyzer.plot_correlation_heatmap_with_significance(
            ...     stats_name="spearman",
            ...     lower_bound=-0.5,
            ...     upper_bound=0.5,
            ...     title_name="Cohort A"
            ... )
            >>> # Show in Jupyter or browser
            >>> fig.show()
        """
        # --- Compute correlation and filter weak correlations ---
        coef_matrix, pval_matrix = self.filter_weak_correlations(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            alpha=alpha,
        )

        if coef_matrix.empty or np.isnan(coef_matrix.values).all():
            raise ValueError("Coefficient matrix is empty. Try adjusting the lower_bound parameter.")

        # --- Dynamic layout scaling based on matrix size ---
        n_rows, n_cols = coef_matrix.shape
        cell_size = 60  # Base pixel size per cell

        max_dim = max(n_rows, n_cols)
        fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

        title_fontsize = max(14 + max_dim // 2, 28)
        label_fontsize = max(8, 12 - max_dim // 8)
        annot_fontsize = max(6, 12 - max_dim // 8)

        # --- Prepare matrix and annotations ---
        display_matrix = coef_matrix.fillna(0)
        text_matrix = np.where(
            np.isnan(coef_matrix.values),
            "",
            coef_matrix.round(2).astype(str)
        )

        # --- Generate custom hover text per cell ---
        hover_text = np.empty_like(coef_matrix, dtype=object)
        counts_lookup = {}
        for row in self.correlation_results.itertuples():
            # forward (original counts)
            counts_lookup[(row.HPO_A, row.HPO_B)] = {
                "Coefficient": row.correlation,
                "P_value": row.p_value,
                "P_value_corrected": row.p_value_corrected,
                "Count_00": row.Count_00,
                "Count_01": row.Count_01,
                "Count_10": row.Count_10,
                "Count_11": row.Count_11,
                "n_patients": row.n_patients,
                "n_pmids": row.n_pmids
            }
            # backward (exchange Count_01 和 Count_10)
            counts_lookup[(row.HPO_B, row.HPO_A)] = {
                "Coefficient": row.correlation,
                "P_value": row.p_value,
                "P_value_corrected": row.p_value_corrected,
                "Count_00": row.Count_00,
                "Count_01": row.Count_10,  # swapped
                "Count_10": row.Count_01,  # swapped
                "Count_11": row.Count_11,
                "n_patients": row.n_patients,
                "n_pmids": row.n_pmids
            }


        hover_text = []
        for i, row in enumerate(coef_matrix.index):
            hover_row = []
            for j, col in enumerate(coef_matrix.columns):
                coef = coef_matrix.iloc[i, j]
                pval = pval_matrix.iloc[i, j]
                if np.isnan(coef):
                    hover_row.append("")
                else:
                    counts = counts_lookup.get((row, col), {})
                    hover_row.append(
                        f"<b>HPO_A</b>: {col}<br><b>HPO_B</b>: {row}<br>"
                        f"<b>Corr</b>: {coef:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>p-val_corrected</b>: {counts.get('P_value_corrected', np.nan):.6f}<br>"
                        f"<b>Counts_ab</b>: {counts.get('Count_00', 0)}, "
                        f"<b>Counts_aB</b>: {counts.get('Count_01', 0)}, "
                        f"<b>Counts_Ab</b>: {counts.get('Count_10', 0)}, "
                        f"<b>Counts_AB</b>: {counts.get('Count_11', 0)}<br>"
                        f"<b>Total patients</b>: {counts.get('n_patients', 0)}<br>"
                        f"<b>PMIDs</b>: {counts.get('n_pmids', 0)}"
                    )
            hover_text.append(hover_row)
          

        # --- Create heatmap figure ---
        fig = go.Figure(
            go.Heatmap(
                z=display_matrix.values,
                x=coef_matrix.columns,
                y=coef_matrix.index,
                colorscale="RdBu",
                zmid=0,
                text=text_matrix,
                texttemplate=f"<span style='font-size:{annot_fontsize}px'>%{{text}}</span>",
                hovertext=hover_text,
                hoverinfo="text",
                colorbar=dict(title="Corr.", len=0.8, thickness=title_fontsize),
                zmin=-1,
                zmax=1,
                xgap=1,
                ygap=1,
            )
        )

        # --- Adjust layout ---
        max_ylabel_len = max(len(str(lbl)) for lbl in coef_matrix.index)
        left_margin = 60 + max_ylabel_len * label_fontsize

        fig.update_layout(
            title=dict(
                text=f"<b>{stats_name.capitalize()} Correlation</b><br>"
                    f"<span style='font-size:0.8em'>{title_name}</span>",
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
            plot_bgcolor="rgba(240,240,240,0.1)"
        )
        return fig
    

    def save_correlation_heatmap(
            self, 
            fig: go.Figure, 
            output_file: str
        ) -> None:
        """
        Save a correlation heatmap figure to an HTML file.

        Args:
            fig (plotly.graph_objects.Figure): 
                The heatmap figure generated by `plot_correlation_heatmap_with_significance`.
            output_file (str): 
                Path to the HTML file where the figure should be saved. Must end with '.html'.

        Raises:
            ValueError:
                If the output_file extension is not '.html'.

        Example:
            >>> fig = analyzer.plot_correlation_heatmap_with_significance()
            >>> analyzer.save_correlation_heatmap(fig, "correlation_heatmap.html")
        """
        if not output_file.endswith(".html"):
            raise ValueError("output_file must have a '.html' extension")
        fig.write_html(output_file)