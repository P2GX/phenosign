# Load phenopackets

Loading phenopacket data is the first step of any analysis workflow in **ppkt2synergy**.

The package provides convenient functions to load phenopackets from a phenopacket store based on:

* cohort identifiers (e.g., gene-based cohorts)
* disease identifiers (e.g., OMIM IDs)

Before loading data, ensure that the requested cohort or disease exists in the phenopacket store.

---

## Load by cohort

### Load a single cohort

```python
from ppkt2synergy import load_phenopackets_by_cohort

phenopackets = load_phenopackets_by_cohort("TGFBR1")
print(len(phenopackets))
```

Loads phenopackets associated with a single cohort (e.g., a gene-based cohort).

---

### Load multiple cohorts

```python
multi_cohort_names = [
    "TGFBR1", "TGFBR2", "SMAD3", "TGFB2", "TGFB3", "SMAD2"
]

phenopackets_multi = load_phenopackets_by_cohort(multi_cohort_names)
print(len(phenopackets_multi))
```

Combines phenopackets from multiple cohorts into a single list.

---

### Load all cohorts

```python
phenopackets_all = load_phenopackets_by_cohort()
print(len(phenopackets_all))
```

Loads all available phenopackets in the store.

---

## Load by disease

### Load a single disease

```python
from ppkt2synergy import load_phenopackets_by_disease

phenopackets_disease = load_phenopackets_by_disease("OMIM:614816")
print(len(phenopackets_disease))
```

Retrieves phenopackets associated with a specific disease identifier.

---

### Load multiple diseases

```python
phenopackets_diseases = load_phenopackets_by_disease([
    "OMIM:614816",
    "OMIM:610168",
    "OMIM:609192"
])

print(len(phenopackets_diseases))
```

Aggregates phenopackets across multiple diseases.

---

## Specify Phenopacket Store version

By default, the latest available release of the Phenopacket Store is used.

You can specify a particular release version using the ppkt_store_version argument:

```python
phenopackets = load_phenopackets_by_cohort(
    "TGFBR1",
    ppkt_store_version="0.1.23"
)
```

This ensures reproducibility when working with a fixed dataset version.

---

## Notes

* Cohort-based loading is typically used for gene-centered analysis
* Disease-based loading supports condition-level aggregation
* The returned objects can be used directly for downstream analysis

---

## Next steps

After loading phenopackets, the next step is to construct a dataset suitable for analysis.

See **Build dataset** for how to transform phenopackets into feature matrices and target variables.
