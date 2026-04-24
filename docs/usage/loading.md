# Load phenopackets

Loading phenopacket data is the first step in any **ppkt2synergy** workflow. The package provides convenient functions to retrieve phenopackets from the **Phenopacket Store** (see [here](https://github.com/monarch-initiative/phenopacket-store)), based on:

* **Cohort identifiers** (e.g., gene-based cohorts)  
* **Disease identifiers** (e.g., OMIM IDs)

Before loading data, ensure that the requested cohort or disease exists in the [Phenopacket Store](https://github.com/monarch-initiative/phenopacket-store).

---

## Load by cohort

### Single cohort

```python
from ppkt2synergy import load_phenopackets_by_cohort

phenopackets = load_phenopackets_by_cohort("TGFBR1")
print(f"Loaded {len(phenopackets)} phenopackets")
```

Loads phenopackets associated with a single cohort (e.g., a gene-based cohort).

---

### Multiple cohorts

```python
multi_cohort_names = ["TGFBR1", "TGFBR2", "SMAD3", "TGFB2", "TGFB3", "SMAD2"]

phenopackets_multi = load_phenopackets_by_cohort(multi_cohort_names)
print(f"Loaded {len(phenopackets_multi)} phenopackets from multiple cohorts")
```

Combines phenopackets from multiple cohorts into a single list for aggregated analysis.

---

### All cohorts

```python
phenopackets_all = load_phenopackets_by_cohort()
print(f"Loaded {len(phenopackets_all)} phenopackets from all available cohorts")
```

Loads all available phenopackets in the store.

---

## Load by disease

### Single disease

```python
from ppkt2synergy import load_phenopackets_by_disease

phenopackets_disease = load_phenopackets_by_disease("OMIM:614816")
print(f"Loaded {len(phenopackets_disease)} phenopackets for the disease OMIM:614816")
```

Retrieves phenopackets associated with a specific disease identifier.

---

### Multiple diseases

```python
phenopackets_diseases = load_phenopackets_by_disease([
    "OMIM:614816",
    "OMIM:610168",
    "OMIM:609192"
])
print(f"Loaded {len(phenopackets_diseases)} phenopackets across multiple diseases")
```

Aggregates phenopackets for several diseases into a single collection.

---

## Specify Phenopacket Store version

By default, **ppkt2synergy** uses the latest available release of the Phenopacket Store. To ensure reproducibility, you can specify a particular version with the ppkt_store_version argument:

```python
phenopackets = load_phenopackets_by_cohort(
    "TGFBR1",
    ppkt_store_version="0.1.23"
)
```

---

## Notes

* **Cohort-based** loading is ideal for gene-centered analyses.
* **Disease-based** loading allows aggregation at the condition level.
* The returned phenopacket objects can be used directly for downstream processing, such as building feature matrices and target variables.

---

## Next steps

Once phenopackets are loaded, proceed to **Build dataset** to transform raw phenotypic data into structured matrices suitable for correlation and synergy analysis.
