from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import logging
import os
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.sparse import coo_matrix, triu

from ..core import PhenotypeDataset

logger = logging.getLogger(__name__)

class SynergyAnalyzer:
    """
    Analyze pairwise synergy between HPO terms with respect to a target.

    This class computes pairwise feature synergy using mutual information
    and permutation testing. Targets can be retrieved from pre-built target
    matrices or generated from metadata.
    """
    def __init__(
        self, 
        dataset: PhenotypeDataset, 
        n_permutations: int = 100,
        min_individuals_for_synergy_calculation: int = 40, 
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        dataset : PhenotypeDataset
            Analysis-ready dataset containing HPO features, targets, and metadata.
        n_permutations : int, default=100
            Number of permutations used to estimate p-values.
        min_individuals_for_synergy_calculation : int, default=40
            Minimum number of valid individuals required to evaluate a feature pair.
        random_state : int, default=42
            Random seed for reproducible permutation testing.

        Examples:
            >>> analyzer = PairwiseSynergyAnalyzer(
            ...     dataset=dataset,
            ...     n_permutations=100,
            ...     min_individuals_for_synergy_caculation=40,
            ...     random_state
            ... )
        """
        if not isinstance(dataset, PhenotypeDataset):
            raise TypeError("dataset must be an instance of PhenotypeDataset.")
        
        self.dataset = dataset
        hpo_df =self.dataset.hpo_data.matrix
        self.X = hpo_df.to_numpy(copy=True)
        self.hpo_terms = hpo_df.columns
        self.n_features = hpo_df.shape[1]
        self.individual_ids = hpo_df.index
    
        self.label_mapping = self.dataset.hpo_data.label_mapping
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_state)

        relationship_mask_df = self.dataset.hpo_data.relationship_mask
        if relationship_mask_df is not None:
            self.relationship_mask = relationship_mask_df.to_numpy(copy=True)
        else:
            logger.warning("No relationship_mask provided. All feature pairs will be evaluated for synergy.")
            self.relationship_mask = np.zeros((self.n_features, self.n_features))
            np.fill_diagonal(self.relationship_mask, np.nan)

        self.min_individuals_for_synergy_calculation = min_individuals_for_synergy_calculation

    @staticmethod
    def _encode_joint_binary_index( 
        xi:np.ndarray, 
        xj:np.ndarray
    ) -> np.ndarray:
        """
        Encode two binary features into a joint integer representation.

        Parameters
        ----------
        xi : np.ndarray
            Binary values for the first feature.
        xj : np.ndarray
            Binary values for the second feature.

        Returns
        -------
        np.ndarray
            Joint encoding with values in ``{0, 1, 2, 3}``.
        """
        return (xi.astype(int) << 1) | xj.astype(int)

    def evaluate_pair_synergy(
        self, 
        i:int,
        j:int,
        include_pmids: bool = True
    ) -> tuple[int, int, float, float, dict]: 
        """
        Compute synergy and a permutation-based p-value for one feature pair.

        Parameters
        ----------
        i : int
            Index of the first feature.
        j : int
            Index of the second feature.
        include_pmids : bool, default=True
            If ``True``, aggregate PMIDs from contributing individuals.

        Returns
        -------
        tuple[int, int, float, float, dict]
            Feature indices, corrected synergy, p-value, and count summary.
        """
        mask = (~np.isnan(self.X[:, i]) & ~np.isnan(self.X[:, j]) & ~np.isnan(self.y))
        xi = self.X[mask, i]
        xj = self.X[mask, j]
        y = self.y[mask]
        total = len(xi)

        empty_counts = {
            "00|y=0": 0,
            "01|y=0": 0,
            "10|y=0": 0,
            "11|y=0": 0,
            "N_y0": 0,
            "00|y=1": 0,
            "01|y=1": 0,
            "10|y=1": 0,
            "11|y=1": 0,
            "N_y1": 0,
            "N": 0,
            "n_pmid": np.nan,
            "pmids": [],
        }

        if np.all(xi == xi[0]) or np.all(xj == xj[0]) or np.all(y == y[0]):
            return i, j, np.nan, np.nan, empty_counts
        
        if np.array_equal(xi, xj):
            return i, j, np.nan, np.nan, empty_counts

        if include_pmids:
            individual_ids = self.individual_ids[mask]
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
        
        counts = {
        # y=0 
        "00|y=0": np.sum((xi == 0) & (xj == 0) & (y == 0)),
        "01|y=0": np.sum((xi == 0) & (xj == 1) & (y == 0)),
        "10|y=0": np.sum((xi == 1) & (xj == 0) & (y == 0)),
        "11|y=0": np.sum((xi == 1) & (xj == 1) & (y == 0)),
        "N_y0": np.sum(y == 0),

        # y=1 
        "00|y=1": np.sum((xi == 0) & (xj == 0) & (y == 1)),
        "01|y=1": np.sum((xi == 0) & (xj == 1) & (y == 1)),
        "10|y=1": np.sum((xi == 1) & (xj == 0) & (y == 1)),
        "11|y=1": np.sum((xi == 1) & (xj == 1) & (y == 1)),
        "N_y1": np.sum(y == 1),
        
        "N": total,
        "n_pmid": n_pmids,
        "pmids": all_pmids,
        }

        mi_i = mutual_info_score(xi, y) / np.log(2)
        mi_j = mutual_info_score(xj, y) / np.log(2)

        joint_index = self._encode_joint_binary_index(xi, xj)
        mi_ij = mutual_info_score(joint_index, y) / np.log(2)
        observed_synergy = mi_ij - (mi_i + mi_j)

        perm_synergies = np.zeros(self.n_permutations)
        for k in range(self.n_permutations):
            y_perm = self.rng.permutation(y)  # Shuffle the target values
            mi_i_perm = mutual_info_score(xi, y_perm) / np.log(2)
            mi_j_perm = mutual_info_score(xj, y_perm) / np.log(2)
            mi_ij_perm = mutual_info_score(joint_index, y_perm) / np.log(2)
            perm_synergies[k] = mi_ij_perm - (mi_i_perm + mi_j_perm)

        p_value = (np.sum(np.abs(perm_synergies) >= np.abs(observed_synergy)) + 1) / (self.n_permutations + 1)
        corrected_synergy = observed_synergy - perm_synergies.mean()

        return i, j, corrected_synergy, p_value, counts


    def compute_synergy_matrix(
        self, 
        target_type: str,
        target_name: str | None = None,
        positive_class=None,
        n_jobs=-1,
        include_pmids: bool = True
    ) -> pd.DataFrame:
        """
        Compute pairwise synergy scores for all valid HPO term pairs.

        Parameters
        ----------
        target_type : str
            Target type to analyze.
        target_name : str | None, optional
            Target column name for built target matrices.
        positive_class : str | None, optional
            Positive class for metadata-derived targets.
        n_jobs : int, default=-1
            Number of parallel jobs. ``-1`` uses all available CPUs.
        include_pmids : bool, default=True
            If ``True``, aggregate PMIDs from contributing individuals and
            include them in the result table.

        Returns
        -------
        pd.DataFrame
            Long-format table of pairwise synergy results.

        Examples:
            >>> analyzer = PairwiseSynergyAnalyzer(
            ...     dataset=dataset,
            ...     n_permutations=100,
            ...     min_individuals_for_synergy_caculation=40,
            ...     random_state
            ... )
            >>> # Example 1: Disease target (pre-built matrix)
            >>> analyzer.compute_synergy(
            ...     target_type="disease",
            ...     target_name="OMIM:123456",  # disease column in target matrix
            ...)
            >>> # Example 2: Metadata target (binary from sample metadata)
            >>> analyzer.compute_synergy(
            ...     target_type="sex",
            ...     positive_class="female",)
        """
        target = self.dataset.get_target(
            target_type=target_type,
            target_name=target_name,
            positive_class=positive_class,
        )
        self.y = target.to_numpy(copy=True)
        
        has_y_one = np.any(self.y == 1)
        has_y_zero = np.any(self.y == 0)

        if not has_y_one or not has_y_zero:
            raise ValueError(
                "Target lacks sufficient variation for synergy analysis.\n"
                f"Detected values: "
                f"{'1 present, ' if has_y_one else 'no 1, '}"
                f"{'0 present' if has_y_zero else 'no 0'}.\n"
                "At least one positive (1) and one negative (0) target value are required."
            )

        has_x_one = np.any(self.X == 1)
        has_x_zero = np.any(self.X == 0)

        if not has_x_one or not has_x_zero:
            raise ValueError(
                "HPO matrix lacks sufficient variation for synergy analysis.\n"
                f"Detected values: "
                f"{'1 present, ' if has_x_one else 'no 1, '}"
                f"{'0 present' if has_x_zero else 'no 0'}.\n"
                "At least one observed (1) and one excluded (0) value are required.\n"
                "Please check your preprocessing (e.g., missing exclusion annotations)."
            )

        mask_X = ~np.isnan(self.X)  # X valid
        mask_y = ~np.isnan(self.y)  # y valid
        mask_combined = mask_X & mask_y[:, None]  # shape (n_samples, n_features)

        valid_counts = mask_combined.T.astype(int) @ mask_combined.astype(int)
        valid_counts_sparse = triu(coo_matrix(valid_counts), k=1)
        rows, cols, counts = valid_counts_sparse.row, valid_counts_sparse.col, valid_counts_sparse.data

        ontology_values = self.relationship_mask[rows, cols]
        ontology_candidate = ~np.isnan(ontology_values)

        candidate_idx = np.where(ontology_candidate & (counts >= self.min_individuals_for_synergy_calculation))[0]
        rows_cand, cols_cand = rows[candidate_idx], cols[candidate_idx]
        pairs = list(zip(rows_cand, cols_cand))

        if len(pairs) == 0:
            logger.warning(
                "No valid HPO term pairs remain after pre-filtering for synergy analysis.\n"
                "This occurs before pairwise computation and usually indicates that no feature pairs\n"
                "passed the initial candidate selection.\n\n"
                "Possible reasons:\n"
                f"- too few valid individuals per pair (min required = {self.min_individuals_for_synergy_calculation})\n"
                "- ontology relationship masking removed most pairs (ancestor/descendant/self)\n"
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_pair_synergy)(i, j, include_pmids=include_pmids) for i, j in tqdm(pairs, desc="Calculating pairwise synergy")
        )
        synergy_matrix = np.full((self.n_features, self.n_features), np.nan)
        pvalue_matrix = np.full((self.n_features, self.n_features), np.nan)

        rows = []
        for i, j, synergy, pval, counts in results:
            synergy_matrix[i, j] = synergy_matrix[j, i] = synergy
            pvalue_matrix[i, j] = pvalue_matrix[j, i] = pval
            hpo_a, hpo_b = self.hpo_terms[i], self.hpo_terms[j]
            if j > i:  # only upper triangle
                if not np.isnan(synergy):
                    row_data = {
                        "HPO_A": hpo_a,
                        **({"HPO_A_label": self.label_mapping.get(hpo_a)} if self.label_mapping.get(hpo_a) else {}),
                        "HPO_B": hpo_b,
                        **({"HPO_B_label": self.label_mapping.get(hpo_b)} if self.label_mapping.get(hpo_b) else {}),
                        "synergy": synergy,
                        "p_value": pval,
                        "Count_00_y0": counts["00|y=0"],
                        "Count_01_y0": counts["01|y=0"],
                        "Count_10_y0": counts["10|y=0"],
                        "Count_11_y0": counts["11|y=0"],
                        "N_y0": counts["N_y0"],
                        "Count_00_y1": counts["00|y=1"],
                        "Count_01_y1": counts["01|y=1"],
                        "Count_10_y1": counts["10|y=1"],
                        "Count_11_y1": counts["11|y=1"],
                        "N_y1": counts["N_y1"],
                        "n_individuals": counts["N"],
                    }
                    if include_pmids:
                        row_data["n_pmids"] = counts["n_pmid"]
                        row_data["pmids"] = ";".join(counts.get("pmids", []))

                    rows.append(row_data)

        valid_mask = ~((np.isnan(synergy_matrix).all(axis=0)) | (np.nan_to_num(synergy_matrix, nan=0).sum(axis=0) == 0))
        valid_hpo_terms = self.hpo_terms[valid_mask]
        if len(valid_hpo_terms) == 0:
            logger.warning(
                "No valid synergy values were found between HPO terms. "
                "The synergy matrix will be empty.\n"
                "Possible reasons include:\n"
                "- some HPO term pairs have no variation after masking (only one observed state)\n"
                "- the target has no variation within valid samples for many pairs\n"
                "- paired HPO terms are identical after masking\n"
            )

        self.synergy_matrix = pd.DataFrame(
            synergy_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )

        self.pvalue_matrix = pd.DataFrame(
            pvalue_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )
        self.synergy_results = pd.DataFrame(rows)
        if not self.synergy_results.empty:
            pvals = self.synergy_results["p_value"].values
            _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
            loc = int(self.synergy_results.columns.get_loc("p_value"))
            self.synergy_results.insert(
                loc + 1,
                "p_value_corrected",
                pvals_corrected
            )
            self.synergy_results.sort_values(by="p_value", ascending=True, inplace=True)
        
        return self.synergy_results
    

    def save_synergy_results(
        self, 
        synergy_threshold: float = 0.0,
        adj_pval_threshold: float = 1.0,
        output_file: str = "synergy_results.csv"
    ) -> None:
        """
        Save synergy results to a CSV or Excel file.

        Parameters
        ----------
        synergy_threshold : float, default=0.0
            Minimum synergy value to retain.
        adj_pval_threshold : float, default=1.0
            Maximum adjusted p-value to retain.
        output_file : str, default="synergy_results.csv"
            Output file path. Supported formats are ``.csv`` and ``.xlsx``.
        """
        if not hasattr(self, "synergy_results"):
            raise ValueError("Synergy results not computed. Run compute_synergy_matrix() first.")
        
        if self.synergy_results.empty:
            logger.warning("Warning: Synergy results are empty. No file will be saved.")
            return
        
        df = self.synergy_results.copy()
        if synergy_threshold < 0.0:
            raise ValueError("synergy_threshold must be non-negative.")
        df = df[df["synergy"] >= synergy_threshold]

        if adj_pval_threshold < 0.0 or adj_pval_threshold > 1.0:
            raise ValueError("adj_pval_threshold must be between 0.0 and 1.0.")
        df = df[df["p_value_corrected"] < adj_pval_threshold]

        ext = os.path.splitext(output_file)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise ValueError(f"Unsupported file format: {ext}. Use '.csv' or '.xlsx'.")
        
        if ext == ".csv":
            df.to_csv(output_file, index=False)
        else:
            df.to_excel(output_file, index=False)
   
    
    def filter_weak_synergy(
        self, 
        synergy_threshold: float = 0.08, 
        adj_pval_threshold: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter the synergy and p-value matrices by effect size and significance.

        Parameters
        ----------
        synergy_threshold : float, default=0.08
            Minimum synergy value to retain.
        adj_pval_threshold : float, default=0.1
            Maximum adjusted p-value to retain.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Filtered synergy matrix and filtered p-value matrix.
        """
        if not hasattr(self, 'pvalue_matrix'):
            raise RuntimeError("Synergy matrix not found. Please run `compute_synergy_matrix()` first.")
        
        synergy_matrix, p_value = self.synergy_matrix.copy(), self.pvalue_matrix.copy()
        if self.synergy_results.empty:
            logger.warning("Warning: Synergy results are empty.")
            return 

        if synergy_threshold < 0.0:
            raise ValueError("synergy_threshold must be non-negative.")
        mask = synergy_matrix < synergy_threshold
        synergy_matrix[mask] = np.nan
        p_value[mask] = np.nan
        
        if adj_pval_threshold < 0.0 or adj_pval_threshold > 1.0:
            raise ValueError("corrected_alpha must be between 0.0 and 1.0.")

        non_signif = self.synergy_results.loc[
        (self.synergy_results["p_value_corrected"] >= adj_pval_threshold), ["HPO_A", "HPO_B"]
        ]
        for _, row in non_signif.iterrows():
            hpo_a, hpo_b = row["HPO_A"], row["HPO_B"]
            if hpo_a in synergy_matrix.index and hpo_b in synergy_matrix.columns:
                synergy_matrix.loc[hpo_a, hpo_b] = np.nan
                synergy_matrix.loc[hpo_b, hpo_a] = np.nan
                p_value.loc[hpo_a, hpo_b] = np.nan
                p_value.loc[hpo_b, hpo_a] = np.nan

        # Remove rows/columns that are completely NaN
        mask_rows = synergy_matrix.isna().all(axis=1)
        mask_cols = synergy_matrix.isna().all(axis=0)
        synergy_matrix_cleaned = synergy_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return synergy_matrix_cleaned, p_value_cleaned    

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

    def plot_synergy_heatmap(
            self, 
            synergy_threshold: float = 0.08,
            adj_pval_threshold: float = 0.1,
            target_name: str = "",
        ) -> go.Figure:
        """
        Plot a heatmap of pairwise synergy values.

        Parameters
        ----------
        synergy_threshold : float, default=0.08
            Minimum synergy value to display.
        adj_pval_threshold : float, default=0.1
            Maximum adjusted p-value to display.
        target_name : str, optional
            Target name shown in the plot title.

        Returns
        -------
        go.Figure
            Plotly heatmap figure.
        """
        synergy_matrix, pvalue_matrix = self.filter_weak_synergy(synergy_threshold=synergy_threshold, adj_pval_threshold=adj_pval_threshold)

        if synergy_matrix.empty or np.isnan(synergy_matrix.values).all():
            raise ValueError(
                "No sufficient synergy pairs remain after filtering. "
                "Try adjusting `synergy_threshold` or `adj_pval_threshold`."
            )
        
        raw_synergy_matrix_df = self.synergy_matrix.loc[synergy_matrix.index, synergy_matrix.columns]
        relationship_mask_df = pd.DataFrame(
            self.relationship_mask,
            index=self.hpo_terms,
            columns=self.hpo_terms,
        )

        relationship_mask_df = relationship_mask_df.loc[synergy_matrix.index, synergy_matrix.columns]

        status_matrix = pd.DataFrame(
            "hidden_upper_triangle",
            index=synergy_matrix.index,
            columns=synergy_matrix.columns
        )

        relationship_isna = relationship_mask_df.isna()
        relationship_notna = ~relationship_isna

        status_matrix[relationship_isna] = "ontology_related"
        status_matrix[raw_synergy_matrix_df.isna() & relationship_notna] = "invalid_computation"
        status_matrix[raw_synergy_matrix_df.notna() & synergy_matrix.isna()] = "filtered_by_statistics"

        n_rows, n_cols = synergy_matrix.shape
        cell_size = 60  # Base pixel size per cell

        max_dim = max(n_rows, n_cols)
        fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

        title_fontsize = max(14 + max_dim // 2, 28)
        label_fontsize = max(8, 12 - max_dim // 8)
        annot_fontsize = max(6, 12 - max_dim // 8)

        triangle_mask = pd.DataFrame(
            np.tril(np.ones(synergy_matrix.shape, dtype=bool), k=0),
            index=synergy_matrix.index,
            columns=synergy_matrix.columns
        )
        synergy_matrix = synergy_matrix.where(triangle_mask)
        pvalue_matrix = pvalue_matrix.where(triangle_mask)
        display_matrix = synergy_matrix.where(triangle_mask)
        nan_bg = pd.DataFrame(np.nan, index=synergy_matrix.index, columns=synergy_matrix.columns)
        nan_bg[triangle_mask & synergy_matrix.isna()] = 2

        text_matrix = np.where(
            np.isnan(synergy_matrix.values),
            "",
            synergy_matrix.round(2).astype(str)
        )

        counts_lookup = {}
        for row in self.synergy_results.itertuples():
            has_pmids = hasattr(row, "n_pmids")
            forward = {
                "Synergy": row.synergy,
                "P_value": row.p_value,
                "P_value_corrected": row.p_value_corrected,
                "Count_00|y=0": row.Count_00_y0,
                "Count_01|y=0": row.Count_01_y0,
                "Count_10|y=0": row.Count_10_y0,
                "Count_11|y=0": row.Count_11_y0,
                "N_y0": row.N_y0,
                "Count_00|y=1": row.Count_00_y1,
                "Count_01|y=1": row.Count_01_y1,
                "Count_10|y=1": row.Count_10_y1,
                "Count_11|y=1": row.Count_11_y1,
                "N_y1": row.N_y1,
                "n_individuals": row.n_individuals,
            }

            backward = {
                "Synergy": row.synergy,
                "P_value": row.p_value,
                "P_value_corrected": row.p_value_corrected,
                "Count_00|y=0": row.Count_00_y0,
                "Count_01|y=0": row.Count_10_y0,
                "Count_10|y=0": row.Count_01_y0,
                "Count_11|y=0": row.Count_11_y0,
                "N_y0": row.N_y0,
                "Count_00|y=1": row.Count_00_y1,
                "Count_01|y=1": row.Count_10_y1,
                "Count_10|y=1": row.Count_01_y1,
                "Count_11|y=1": row.Count_11_y1,
                "N_y1": row.N_y1,
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
        for i, row in enumerate(pvalue_matrix.index):
            hover_row = []
            for j, col in enumerate(pvalue_matrix.columns):
                synergy = synergy_matrix.iloc[i, j]
                pval = pvalue_matrix.iloc[i, j]

                display_row = self._format_hpo_pair(row, self.label_mapping.get(row))
                display_col = self._format_hpo_pair(col, self.label_mapping.get(col))

                if not triangle_mask.iloc[i, j]:
                    hover_row.append("")
                elif np.isnan(synergy):
                    status = status_matrix.iloc[i, j]
                    reason_map = {
                        "ontology_related": "these two HPO terms are ontologically related (ancestor/descendant/self).",
                        "invalid_computation": "the synergy could not be computed for this HPO pair.",
                        "filtered_by_statistics": (
                            f"the synergy value did not pass the statistical filters "
                            f"(|corr| >= {synergy_threshold} and adjusted p-value < {adj_pval_threshold})."
                        ),
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
                        pmids_raw = counts.get("pmids", "")
                        pmid_text = self._format_pmids_for_tooltip(pmids_raw, max_pmids=4)
                        pmid_block = (
                            f"<b>N_PMIDs</b>: {int(counts.get('n_pmids', 0))}<br>"
                            f"<b>PMIDs</b>: {pmid_text}"
                        )
                    hover_row.append(
                        f"<b>HPO_A</b>: {display_col}<br><b>HPO_B</b>: {display_row}<br>"
                        f"<b>Synergy</b>: {synergy:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>p-val_corrected</b>: {counts.get('P_value_corrected', np.nan):.6f}<br>"
                        f"<b>Counts (y=0)</b><br>"
                        f"&nbsp;&nbsp;E/E: {counts.get('Count_00|y=0', 0)}, "
                        f"E/O: {counts.get('Count_01|y=0', 0)}, "
                        f"O/E: {counts.get('Count_10|y=0', 0)}, "
                        f"O/O: {counts.get('Count_11|y=0', 0)} "
                        f" (<i>N={counts.get('N_y0', 0)}</i>)<br>"
                        f"<b>Counts (y=1)</b><br>"
                        f"&nbsp;&nbsp;E/E: {counts.get('Count_00|y=1', 0)}, "
                        f"E/O: {counts.get('Count_01|y=1', 0)}, "
                        f"O/E: {counts.get('Count_10|y=1', 0)}, "
                        f"O/O: {counts.get('Count_11|y=1', 0)} "
                        f" (<i>N={counts.get('N_y1', 0)}</i>)<br>"
                        f"<b>Total_individuals</b>: {counts.get('n_individuals', 0)}<br>"
                        f"{pmid_block}"
                    )
            hover_text.append(hover_row)
       
        synergy_matrix.rename(columns=self.label_mapping, index=self.label_mapping, inplace=True)
        # --- Create heatmap figure ---
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=nan_bg.values,
            x=synergy_matrix.columns,
            y=synergy_matrix.index,
            colorscale=[[0, "#dbe7f3"], [1, "#dbe7f3"]],
            showscale=False,
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        ))
        colorscale = [
            [0.0, "#eef6f5"],   
            [0.25, "#c6e2df"],
            [0.5, "#8fbfba"],
            [0.75, "#4f8f8a"],
            [1.0, "#1f4f4b"],   # deep muted teal
        ]
        fig.add_trace(go.Heatmap(
                z=display_matrix.values,
                x=synergy_matrix.columns,
                y=synergy_matrix.index,
                colorscale=colorscale,
                zmid=0,
                text=text_matrix,
                texttemplate=f"<span style='font-size:{annot_fontsize}px'>%{{text}}</span>",
                hovertext=hover_text,
                hoverinfo="text",
                colorbar=dict(title="Synergy", len=0.8, thickness=title_fontsize),
                zmin=0,
                zmax=np.nanmax(display_matrix.values),
                xgap=1,
                ygap=1,
                )
            )

        max_ylabel_len = max(len(str(lbl)) for lbl in synergy_matrix.index)
        left_margin = 60 + max_ylabel_len * label_fontsize

        fig.update_layout(
            title=dict(
                text=f"<b>Pairwise Synergy Heatmap of HPO Features</b><br>"
                    f"<span style='font-size:0.8em'>With respect to {target_name}</span>",
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
    
    def save_synergy_heatmap(
        self, 
        fig: go.Figure, 
        output_file: str
    ) -> None:
        """
        Save a synergy heatmap as an HTML file.

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
