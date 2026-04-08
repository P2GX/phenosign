from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from typing import Tuple, Union
from ..preprocessing import HpoFeatureMatrix
import plotly.graph_objs as go
import logging
import os
from tqdm import tqdm
from itertools import chain
from statsmodels.stats.multitest import multipletests
from scipy.sparse import coo_matrix, triu

logger = logging.getLogger(__name__)

class SynergyAnalyzer:
    """
    Analyzes the pairwise synergy between features using mutual information and permutation testing.
    
    Example:
        from ppkt2synergy import load_phenopackets, PhenopacketAssembler,,SynergyAnalyzer
        >>> phenopackets = load_phenopackets("FBN1")
        >>> assembler = PhenopacketAssembler(phenopackets)
        >>> hpo_matrix, target_matrix = assembler.build()
        >>> target = target_matrix.disease_matrix["Some_Disease"]  

        >>> analyzer = SynergyAnalyzer(hpo_matrix, target, n_permutations=100)
        >>> synergy, pvalues = analyzer.compute_pairwise_synergy_matrix()
        >>> analyzer.plot_synergy_heatmap(significance_threshold=0.05)
    """
    def __init__(
            self, 
            hpo_data: HpoFeatureMatrix, 
            target: Union[pd.Series, pd.DataFrame],
            n_permutations: int = 100,
            min_individuals_for_synergy_calculation: int = 40, 
            random_state: int = 42,
        ):
        """
        Initialize the analyzer with feature data and target variable.

        Agrs:
        hpo_data(HpoFeatureMatrix):
            - Feature matrix of shape (n_samples, n_features): 
                Non-NaN values must be 0 or 1. DataFrame inputs will be converted to a NumPy array.
            - relationship_mask (n_features, n_features):
                Optional 2D dataframe (n_features x n_features) indicating valid feature pairs to evaluate.
                Can be used to skip predefined pairs (e.g. based on HPO hierarchy or previous results).
                If provided, it will be converted to a NumPy array and used to initialize the synergy matrix.
        target(Union[pd.Series, pd.DataFrame]):
            Target vector of shape (n_samples,). Series/DataFrame inputs will be converted to a 1D NumPy array.
        n_permutations(int): (default: 100)
            Number of permutations for calculating p-values.
        min_individuals_for_synergy_caculation(int): (default: 40)
            Minimum number of samples required to calculate synergy.
        random_state(int): (default: 42)
            Seed for reproducible results.

        Raises:
        ValueError:
            - If hpo_matrix is not a 2D array.
            - If target's length does not match hpo_matrix's row count.
            - If hpo_matrix contains values other than 0, 1, or NaN.
            - If mask has an incompatible shape.
            - If min_individuals_for_synergy_calculation is less than 40.
        """
        if not isinstance(hpo_data, HpoFeatureMatrix):
            raise TypeError("hpo_data must be an instance of HpoFeatureMatrix.")
        
        # --- Validate HPO matrix ---
        hpo_matrix = hpo_data.hpo_matrix.to_numpy(copy=True)
        self.hpo_terms = hpo_data.hpo_matrix.columns
        valid_vals = hpo_matrix[~np.isnan(hpo_matrix)]
        if not np.all((valid_vals == 0) | (valid_vals == 1)):
            raise ValueError("Non-NaN values in HPO Matrix must be either 0 or 1")
       
        # --- Validate target ---
        if isinstance(target, pd.DataFrame):
            if target.shape[1] != 1:
                raise ValueError("Target DataFrame must have exactly one column")
            target = target.iloc[:, 0].to_numpy(copy=True)
        elif isinstance(target, pd.Series):
            target = target.to_numpy(copy=True)
        else:
            raise TypeError("target must be a pandas Series or single-column DataFrame")
        if len(target) != hpo_matrix.shape[0]:
            raise ValueError("The number of samples in Target must match the number of samples in HPO Matrix")
        valid_target = target[~np.isnan(target)]
        if not np.all((valid_target == 0) | (valid_target == 1)):
            raise ValueError("Target must contain only 0, 1, or NaN")

        self.X = hpo_matrix
        self.y = target.copy()
        self.patient_ids = hpo_data.hpo_matrix.index
        self.patient_pmids = hpo_data.patient_info_df
        self.label_mapping = hpo_data.label_mapping
        self.n_features = hpo_matrix.shape[1]
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_state)

        # Relationship mask (ontology-aware filtering)
        relationship_mask = hpo_data.hpo_relationship_mask
        if relationship_mask is not None:
            if relationship_mask.shape != (self.n_features, self.n_features):
                raise ValueError("relationship_mask shape mismatch with HPO features")
            self.relationship_mask = relationship_mask.to_numpy(copy=True)
        else:
            logger.warning("No relationship_mask provided. All feature pairs will be evaluated for synergy.")
            self.relationship_mask = np.zeros((self.n_features, self.n_features))

        self.min_individuals_for_synergy_calculation = min_individuals_for_synergy_calculation
        if self.min_individuals_for_synergy_calculation < 30:
                logger.warning(
                    f"min_individuals_for_synergy_calculation is set to {self.min_individuals_for_synergy_calculation}, "
                    f"which is below the recommended threshold of 30. "
                    f"This may lead to unstable or less reliable synergy estimates."
                )

    @staticmethod
    def _encode_joint_binary_index( 
            xi:np.ndarray, 
            xj:np.ndarray
        ) -> np.ndarray:
        """
        Encodes two binary features into a unique integer index via bitwise operations.

        Args:
        xi(ndarray): 
            Feature i's values (int type, 0/1, no NaN).
        xj(ndarray): 
            Feature j's values (int type, 0/1, no NaN).

        Returns:
            ndarray:
            Combined index calculated as: 
            \( \text{joint\_index} = 2 \times \text{xi} + \text{xj} \)
            Possible values: 0 (0b00), 1 (0b01), 2 (0b10), 3 (0b11).

        Example:
        >>> xi = np.array([0, 1, 0, 1], dtype=int)
        >>> xj = np.array([0, 0, 1, 1], dtype=int)
        >>> _encode_joint_binary_index(xi, xj)
        array([0, 2, 1, 3])  # Corresponding to 0b00, 0b10, 0b01, 0b11
        """
        return (xi.astype(int) << 1) | xj.astype(int)

    def evaluate_pair_synergy(
            self, 
            i:int,
            j:int
        ) -> Tuple[int, int, float, float, dict]: 
        """
        Compute synergy and permutation-based p-value for feature pair (i, j).

        Synergy is calculated as:
            synergy = I(X_i, X_j; Y) - [I(X_i; Y) + I(X_j; Y)]

        where I(a; b) denotes the mutual information (in bits) between a and b.

        Args:
        i (int): 
            Index of the first feature.
        j (int): 
            Index of the second feature.

        Returns:
            Tuple[int, int, float, float, dict]:
                - i (int): Index of the first feature.
                - j (int): Index of the second feature.
                - corrected_synergy (float): Corrected synergy score.
                - p_value (float): Empirical p-value from permutation test.
                - counts (dict): Contingency table counts.
        """
        mask = (~np.isnan(self.X[:, i]) & ~np.isnan(self.X[:, j]) & ~np.isnan(self.y))
        xi = self.X[mask, i]
        xj = self.X[mask, j]
        y = self.y[mask]
        total = len(xi)

        if np.all(xi == xi[0]) or np.all(xj == xj[0]) or np.all(y == y[0]):
            return i, j, np.nan, np.nan, {}
        if np.array_equal(xi, xj):
            return i, j, np.nan, np.nan, {}

        patient_ids = self.patient_ids[mask]
        pmids_list = self.patient_pmids.loc[patient_ids, 'pmids'].to_numpy()
        all_pmids = set(chain.from_iterable(pmids_list))
        n_pmids = len(all_pmids)
        
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
        "n_pmid": n_pmids
        }

        mi_i = mutual_info_score(xi, y) / np.log(2)
        mi_j = mutual_info_score(xj, y) / np.log(2)

        joint_index = self._encode_joint_binary_index(xi, xj)
        mi_ij = mutual_info_score(joint_index, y) / np.log(2)

        observed_synergy = mi_ij - (mi_i + mi_j)

        # Permutation testing for p-value calculation
        perm_synergies = np.zeros(self.n_permutations)
        for k in range(self.n_permutations):
            y_perm = self.rng.permutation(y)  # Shuffle the target values
            mi_i_perm = mutual_info_score(xi, y_perm) / np.log(2)
            mi_j_perm = mutual_info_score(xj, y_perm) / np.log(2)
            mi_ij_perm = mutual_info_score(joint_index, y_perm) / np.log(2)
            perm_synergies[k] = mi_ij_perm - (mi_i_perm + mi_j_perm)

        # Calculate p-value as the proportion of permuted synergies greater than or equal to the observed synergy
        p_value = (np.abs(perm_synergies) >= np.abs(observed_synergy)).mean()
        
        # Correct the observed synergy by subtracting the mean of the permuted synergies
        corrected_synergy = observed_synergy - perm_synergies.mean()

        return i, j, corrected_synergy, p_value, counts


    def compute_synergy_matrix(
            self, 
            n_jobs=-1,
        ) -> pd.DataFrame:
        """
        Compute the pairwise synergy scores and permutation-based p-values for all valid feature pairs.

        The synergy score is evaluated for each pair of HPO terms/features that:
        - Have at least `self.min_individuals_for_synergy_caculation` valid samples.
        - Are not masked out in the existing `self.synergy_matrix` (i.e., not NaN).

        Results are stored in two symmetric matrices (`self.synergy_matrix`, `self.pvalue_matrix`)
        and an exported long-format table (`self.synergy_results`).

        Args:
            n_jobs (int, optional): (default: -1)
                Number of parallel jobs to run. 
                Set to -1 to use all available CPU cores.

        Returns:
            pd.DataFrame:
                DataFrame with one row per valid feature pair.
                Each row contains:
                    * HPO_A (str): Name of the first HPO term or feature.
                    * HPO_B (str): Name of the second HPO term or feature.
                    * synergy (float): Synergy score between the two features.
                    * p_value (float): Permutation-based p-value for the synergy.
                    - p_value_corrected (float): P-value adjusted for multiple testing using the Benjamini–Hochberg FDR method.
                    * Count_00_y0 (int): Number of samples with (0,0) for this pair under label y=0.
                    * Count_01_y0 (int): Number of samples with (0,1) for this pair under label y=0.
                    * Count_10_y0 (int): Number of samples with (1,0) for this pair under label y=0.
                    * Count_11_y0 (int): Number of samples with (1,1) for this pair under label y=0.
                    * N_y0 (int): Total number of valid samples under label y=0.
                    * Count_00_y1 (int): Number of samples with (0,0) for this pair under label y=1.
                    * Count_01_y1 (int): Number of samples with (0,1) for this pair under label y=1.
                    * Count_10_y1 (int): Number of samples with (1,0) for this pair under label y=1.
                    * Count_11_y1 (int): Number of samples with (1,1) for this pair under label y=1.
                    * N_y1 (int): Total number of valid samples under label y=1.
                    * n_patients (int): Total number of patients contributing to this pair (N_y0 + N_y1).
                    * n_pmids (int): Number of associated PubMed IDs for this feature pair.
        """
        # --- Step 1: Compute valid sample mask ---
        mask_X = ~np.isnan(self.X)  # X valid
        mask_y = ~np.isnan(self.y)  # y valid
        mask_combined = mask_X & mask_y[:, None]  # shape (n_samples, n_features)

        # Compute valid counts for each pair (like correlation)
        valid_counts = mask_combined.T.astype(int) @ mask_combined.astype(int)
        valid_counts_sparse = triu(coo_matrix(valid_counts), k=1)
        rows, cols, counts = valid_counts_sparse.row, valid_counts_sparse.col, valid_counts_sparse.data

        # --- Step 2: Apply relationship mask ---
        ontology_values = self.relationship_mask[rows, cols]
        ontology_candidate = ~np.isnan(ontology_values)

        # --- Step 3: Filter pairs ---
        candidate_idx = np.where(ontology_candidate & (counts >= self.min_individuals_for_synergy_calculation))[0]
        rows_cand, cols_cand = rows[candidate_idx], cols[candidate_idx]
        pairs = list(zip(rows_cand, cols_cand))

        if len(pairs) == 0:
            logger.warning("Warning: No valid pairs to calculate synergy after filtering.")
            return pd.DataFrame()

        # --- Step 4: Parallel computation ---
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_pair_synergy)(i, j) for i, j in tqdm(pairs, desc="Calculating pairwise synergy")
        )
        synergy_matrix = np.full((self.n_features, self.n_features), np.nan)
        pvalue_matrix = np.full((self.n_features, self.n_features), np.nan)

        rows = []
        for i, j, synergy, pval, counts in results:
            synergy_matrix[i, j] = synergy_matrix[j, i] = synergy
            pvalue_matrix[i, j] = pvalue_matrix[j, i] = pval
            f1, f2 = self.hpo_terms[i], self.hpo_terms[j]
            if j > i:  # only upper triangle
                if not np.isnan(synergy):
                    rows.append({
                        "HPO_A": f1,
                        **({"HPO_A_label": self.label_mapping.get(f1)} if self.label_mapping.get(f1) else {}),
                        "HPO_B": f2,
                        **({"HPO_B_label": self.label_mapping.get(f2)} if self.label_mapping.get(f2) else {}),
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
                        "n_patients": counts["N"],
                        "n_pmids": counts["n_pmid"]
                    })

        valid_mask = ~((np.isnan(synergy_matrix).all(axis=0)) | (np.nan_to_num(synergy_matrix, nan=0).sum(axis=0) == 0))
        valid_hpo_terms = self.hpo_terms[valid_mask]
        if len(valid_hpo_terms) == 0:
            logger.warning("Warning: No valid synergy between HPO terms. Synergy matrix will be empty.")

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
            alpha: float = 1.0,
            corrected_alpha: float = 1.0,
            output_file: str = "synergy_results.csv"
        ) -> None:
        """
        Export synergy scores and p-values to a file (CSV or Excel).

        The synergy results (`self.synergy_matrix` and `self.pvalue_matrix`) 
        must be computed first by calling `compute_synergy_matrix`. 
        Only the upper triangle of the symmetric matrices is exported to avoid duplication.

        Args:
            output_file (str):
                Path to the output file. Supported formats:
                - ".csv": saves as a CSV file.
                - ".xlsx": saves as an Excel file.
            synergy_threshold (float): (default: 0.0)
                Synergy threshold must be positive. Only pairs with synergy >= synergy_threshold will be included.
            alpha (float): (default: 0.0)
                Significance threshold for p-values. alpha must be between 0.0 and 1.0. Only pairs with p-value < alpha will be included.
            corrected_alpha (float): (default: 0.0)
                Significance threshold for corrected p-values. corrected_alpha must be between 0.0 and 1.0. Only pairs with corrected p-value < corrected_alpha will be included.

        Raises:
            ValueError:
                - If synergy results are not computed yet.
                - If file extension is not supported.
            OSError:
                If the output path is invalid or not writable. 

        Example:
            >>> analyzer.compute_synergy_matrix()
            >>> analyzer.save_synergy_results("synergy_results.csv")
            >>> analyzer.save_synergy_results("synergy_results.xlsx")
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
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        if corrected_alpha < 0.0 or corrected_alpha > 1.0:
            raise ValueError("corrected_alpha must be between 0.0 and 1.0.")
        
        df = df[(df["p_value"] < alpha) & (df["p_value_corrected"] < corrected_alpha)]

        ext = os.path.splitext(output_file)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise ValueError(f"Unsupported file format: {ext}. Use '.csv' or '.xlsx'.")

        
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
        else:
            df.to_excel(output_file, index=False)
   
    
    def filter_weak_synergy(
            self, 
            synergy_threshold: float = 0.08, 
            alpha: float = 0.05,
            corrected_alpha: float = 0.1
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter out feature pairs with weak synergy based on given threshold.

        Args:
            synergy_threshold (float): (default: 0.08)
                The minimum synergy value to be saves. Synergy threshold must be non-negative. Pairs with synergy < synergy_threshold will be set to NaN.
            alpha (float): (default: 0.05)
                Significance threshold for p-value. Only pairs with p-value < alpha will be retained. Must be between 0.0 and 1.0.
            corrected_alpha (float): (default: 0.1)
                Significance threshold for corrected p-value. Must be between 0.0 and 1.0. Only pairs with corrected p-value < corrected_alpha will be retained.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - Synergy score matrix with weak synergy pairs removed (set as NaN).
                - Corresponding p-value matrix with the same filtering applied.
        """
        if not hasattr(self, 'pvalue_matrix'):
            raise RuntimeError("Synergy matrix not found. Please run `compute_synergy_matrix()` first.")
        
        synergy_matrix, p_value = self.synergy_matrix.copy(), self.pvalue_matrix.copy()
        if self.synergy_results.empty:
            logger.warning("Warning: Synergy results are empty.")
            return 

        # Mask weak synergy values
        if synergy_threshold < 0.0:
            raise ValueError("synergy_threshold must be non-negative.")
        mask = synergy_matrix < synergy_threshold
        synergy_matrix[mask] = np.nan
        p_value[mask] = np.nan

        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        
        if corrected_alpha < 0.0 or corrected_alpha > 1.0:
            raise ValueError("corrected_alpha must be between 0.0 and 1.0.")

        non_signif = self.synergy_results.loc[
            (self.synergy_results["p_value"] >= alpha) & (self.synergy_results["p_value_corrected"] >= corrected_alpha), ["HPO_A", "HPO_B"]
        ]
        for _, row in non_signif.iterrows():
            hpo1, hpo2 = row["HPO_A"], row["HPO_B"]
            if hpo1 in synergy_matrix.index and hpo2 in synergy_matrix.columns:
                synergy_matrix.loc[hpo1, hpo2] = np.nan
                synergy_matrix.loc[hpo2, hpo1] = np.nan
                p_value.loc[hpo1, hpo2] = np.nan
                p_value.loc[hpo2, hpo1] = np.nan

        # Remove rows/columns that are completely NaN
        mask_rows = synergy_matrix.isna().all(axis=1)
        mask_cols = synergy_matrix.isna().all(axis=0)
        synergy_matrix_cleaned = synergy_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return synergy_matrix_cleaned, p_value_cleaned    

    def _format_hpo_pair(self, hpo_id: str, label: str | None) -> str:
        """Format HPO for display."""
        if label:
            return f"{label} ({hpo_id})"
        return hpo_id

    def plot_synergy_heatmap(
            self, 
            synergy_threshold: float = 0.08,
            alpha: float = 0.05,
            corrected_alpha: float = 0.1,
            target_name: str = "",
        ) -> go.Figure:
        """
        Generate a Plotly heatmap figure of pairwise synergy scores for HPO features.

        This function computes a heatmap of synergy scores, annotates significant pairs,
        and prepares hover text with correlation and p-values.

        Args:
            synergy_threshold (float): (default: 0.08)
                Minimum synergy value to include in the heatmap.synergy_threshold must be non-negative. Pairs with synergy <= synergy_threshold will be set to NaN and not displayed.
            alpha (float): (default: 0.05)
                Significance threshold for p-value. p-value must be between 0.0 and 1.0. Only pairs with p-value < alpha will be annotated as significant. 
            corrected_alpha (float): (default: 0.1)
                Significance threshold for corrected p-value. corrected_alpha must be between 0.0 and 1.0. Only pairs with corrected p-value < corrected_alpha will be annotated as significant.
            target_name (str, optional): 
                Name of the target variable for the plot title.                                 

        Returns:
            go.Figure: A Plotly heatmap figure object ready for display or saving.

        Raises:
            ValueError: If no sufficient synergy pairs exist after filtering.

        Example:
            >>> fig = analyzer.create_synergy_heatmap(lower_bound=0.2, target_name="Disease Status")
            >>> fig.show()
        """

        synergy_matrix, pvalue_matrix = self.filter_weak_synergy(synergy_threshold=synergy_threshold, alpha=alpha, corrected_alpha=corrected_alpha)

        if synergy_matrix.empty or np.isnan(synergy_matrix.values).all():
            raise ValueError("No sufficient synergy pairs to plot. Try adjusting the synergy_threshold parameter.")

        n_rows, n_cols = synergy_matrix.shape
        cell_size = 60  # Base pixel size per cell

        max_dim = max(n_rows, n_cols)
        fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

        title_fontsize = max(14 + max_dim // 2, 28)
        label_fontsize = max(8, 12 - max_dim // 8)
        annot_fontsize = max(6, 12 - max_dim // 8)

        # --- Prepare matrix and annotations ---
        display_matrix = synergy_matrix.fillna(0)
        text_matrix = np.where(
            np.isnan(synergy_matrix.values),
            "",
            synergy_matrix.round(2).astype(str)
        )

        # --- Generate custom hover text per cell ---
        hover_text = np.empty_like(synergy_matrix, dtype=object)
        counts_lookup = {}
        for row in self.synergy_results.itertuples():
            # forward (original counts)
            counts_lookup[(row.HPO_A, row.HPO_B)] = {
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
            "n_patients": row.n_patients,
            "n_pmids": row.n_pmids
            }
        
            # backward (swap 01 和 10)
            counts_lookup[(row.HPO_B, row.HPO_A)] = {
            "Synergy": row.synergy,
            "P_value": row.p_value,
            "P_value_corrected": row.p_value_corrected, 
            "Count_00|y=0": row.Count_00_y0,
            "Count_01|y=0": row.Count_10_y0,  # swap 01 和 10
            "Count_10|y=0": row.Count_01_y0,  # swap 01 和 10
            "Count_11|y=0": row.Count_11_y0,
            "N_y0": row.N_y0,
            "Count_00|y=1": row.Count_00_y1,
            "Count_01|y=1": row.Count_10_y1,  # swap 01 和 10
            "Count_10|y=1": row.Count_01_y1,  # swap 01 和 10
            "Count_11|y=1": row.Count_11_y1,
            "N_y1": row.N_y1,
            "n_patients": row.n_patients,
            "n_pmids": row.n_pmids
            }
        
        hover_text = []
        for i, row in enumerate(pvalue_matrix.index):
            hover_row = []
            for j, col in enumerate(pvalue_matrix.columns):
                synergy = synergy_matrix.iloc[i, j]
                pval = pvalue_matrix.iloc[i, j]
                if np.isnan(synergy):

                    hover_row.append("")
                else:
                    display_row = self._format_hpo_pair(row, self.label_mapping.get(row))
                    display_col = self._format_hpo_pair(col, self.label_mapping.get(col))
                    counts = counts_lookup.get((row, col), {})
                    hover_row.append(
                        f"<b>HPO_A</b>: {display_col}<br><b>HPO_B</b>: {display_row}<br>"
                        f"<b>Synergy</b>: {synergy:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>p-val_corrected</b>: {counts.get('P_value_corrected', np.nan):.6f}<br>"
                        f"<b>Counts (y=0)</b><br>"
                        f"&nbsp;&nbsp;ab: {counts.get('Count_00|y=0', 0)}, "
                        f"aB: {counts.get('Count_01|y=0', 0)}, "
                        f"Ab: {counts.get('Count_10|y=0', 0)}, "
                        f"AB: {counts.get('Count_11|y=0', 0)} "
                        f"(<i>N={counts.get('N_y0', 0)}</i>)<br>"
                        f"<b>Counts (y=1)</b><br>"
                        f"&nbsp;&nbsp;ab: {counts.get('Count_00|y=1', 0)}, "
                        f"aB: {counts.get('Count_01|y=1', 0)}, "
                        f"Ab: {counts.get('Count_10|y=1', 0)}, "
                        f"AB: {counts.get('Count_11|y=1', 0)} "
                        f"(<i>N={counts.get('N_y1', 0)}</i>)<br>"
                        f"<b>Total patients</b>: {counts.get('n_patients', 0)}<br>"
                        f"<b>PMIDs</b>: {counts.get('n_pmids', 0)}"
                    )
            hover_text.append(hover_row)
       
        synergy_matrix.rename(columns=self.label_mapping, index=self.label_mapping, inplace=True)
        # --- Create heatmap figure ---
        fig = go.Figure(
            go.Heatmap(
                z=display_matrix.values,
                x=synergy_matrix.columns,
                y=synergy_matrix.index,
                colorscale='Blues',
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

        # --- Adjust layout ---
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
            plot_bgcolor="rgba(240,240,240,0.1)"
        )
        return fig
    
    def save_synergy_heatmap(
            self, 
            fig: go.Figure, 
            output_file: str
        ) -> None:
        """
        Save a pairwise synergy heatmap figure to an HTML file.

        Args:
            fig (plotly.graph_objects.Figure): 
                The heatmap figure generated by `create_synergy_heatmap` or `plot_synergy_heatmap`.
            output_file (str): 
                Path to the HTML file where the figure should be saved. Must end with '.html'.

        Raises:
            ValueError:
                If the output_file extension is not '.html'.

        Example:
            >>> fig = analyzer.create_synergy_heatmap(lower_bound=0.2, target_name="Disease Status")
            >>> analyzer.save_synergy_heatmap(fig, "synergy_heatmap.html")
        """
        if not output_file.endswith(".html"):
            raise ValueError("output_file must have a '.html' extension")
        fig.write_html(output_file)
