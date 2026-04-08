import pytest
import pandas as pd
import numpy as np
from ppkt2synergy import HPOHierarchyEngine
import pathlib

TEST_DIR = pathlib.Path(__file__).parent.parent.resolve()
HP_JSON_FILE = TEST_DIR / "data/hp.json"

@pytest.fixture
def toy_terms():
    # HP:0001250 (Seizures)
    # └── HP:0020219 (Motor seizure)
    # HP:0012759 (Neurological abnormality)
    return {"HP:0001250", "HP:0020219", "HP:0012759"}

@pytest.fixture
def hpo_engine():
    return HPOHierarchyEngine(hpo_file=str(HP_JSON_FILE))  

def test_propagate_hierarchy(hpo_engine):
    # Simulate a toy matrix before propagation
    data = {
        "HP:0020219": [1, 1, np.nan],
        "HP:0001250": [np.nan, 1, 0],
        "HP:0012759": [1, np.nan, 1],
    }
    df = pd.DataFrame(data, index=["Patient_1", "Patient_2", "Patient_3"])

    propagated = hpo_engine.propagate_hpo_hierarchy(df)

    # After propagation
    # Patient 1: full positive path
    assert propagated.loc["Patient_1", "HP:0020219"] == 1
    assert propagated.loc["Patient_1", "HP:0001250"] == 1
    assert propagated.loc["Patient_1", "HP:0012759"] == 1

    # Patient 2: 0020219 and 0001250 are 1, 0012759 is NaN
    assert propagated.loc["Patient_2", "HP:0020219"] == 1
    assert propagated.loc["Patient_2", "HP:0001250"] == 1
    assert pd.isna(propagated.loc["Patient_2", "HP:0012759"])

    # Patient 3: 0001250 is 0, 0020219 must also be 0; 0012759 is 1
    assert propagated.loc["Patient_3", "HP:0020219"] == 0
    assert propagated.loc["Patient_3", "HP:0001250"] == 0
    assert propagated.loc["Patient_3", "HP:0012759"] == 1

def test_build_relationship_mask(hpo_engine, toy_terms):
    terms = list(toy_terms)
    hpo_engine.tm.prepare_terms(set(terms))
    mask = hpo_engine.build_relationship_mask(terms)
    assert isinstance(mask, pd.DataFrame)
    assert mask.shape == (len(terms), len(terms))

    # Diagonal must be NaN
    assert all(np.isnan(mask.loc[t, t]) for t in terms)

    # Ancestor-descendant pairs should be NaN
    # HP:0001250 is ancestor of HP:0020219
    assert np.isnan(mask.loc["HP:0001250", "HP:0020219"])
    assert np.isnan(mask.loc["HP:0020219", "HP:0001250"])

    # Make sure unrelated terms have numeric value
    assert mask.loc["HP:0001250", "HP:0012759"] == 0

