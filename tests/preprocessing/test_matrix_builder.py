import pytest
import phenopackets as ppkt
import pandas as pd
from ppkt2synergy import HPOHierarchyEngine, PhenopacketMatrixBuilder
import pathlib

TEST_DIR = pathlib.Path(__file__).parent.parent.resolve()
HP_JSON_FILE = TEST_DIR/"data/hp.json"

@pytest.fixture
def mock_phenopackets():
    return [
        ppkt.Phenopacket(
            id="Patient_1",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0020219", label="Motor seizure")),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0012759", label="Neurological abnormality"))
            ],
            diseases=[
                ppkt.Disease(term=ppkt.OntologyClass(id="OMIM:101600", label="Marfan Syndrome"))
            ]
        ),
        ppkt.Phenopacket(
            id="Patient_2",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0001250", label="Seizures")),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0020219", label="Motor seizure"))
            ],
            diseases=[
                ppkt.Disease(term=ppkt.OntologyClass(id="OMIM:603903", label="Ehlers-Danlos syndrome"))
            ]
        ),
        ppkt.Phenopacket(
            id="Patient_3",
            phenotypic_features=[
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0001250", label="Seizures"), excluded=True),
                ppkt.PhenotypicFeature(type=ppkt.OntologyClass(id="HP:0012759", label="Neurological abnormality"))
            ],
            diseases=[
                ppkt.Disease(term=ppkt.OntologyClass(id="OMIM:101600", label="Marfan Syndrome"))
            ]
        )
    ]

@pytest.fixture
def hpo_engine():
    return HPOHierarchyEngine(hpo_file=str(HP_JSON_FILE))

def test_hpo_and_disease_matrix( mock_phenopackets, hpo_engine):

    matrix_generator = PhenopacketMatrixBuilder(mock_phenopackets, hpo_hierarchy=hpo_engine)

    hpo_matrix_with_propagation = matrix_generator.generate_hpo_matrix()
    assert isinstance(hpo_matrix_with_propagation, pd.DataFrame)
    assert hpo_matrix_with_propagation.loc["Patient_1", "HP:0001250"] == 1

    disease_matrix = matrix_generator.generate_disease_matrix()
    assert isinstance(disease_matrix, pd.DataFrame)
    assert disease_matrix.loc["Patient_1", "Marfan Syndrome"] == 1
    assert disease_matrix.loc["Patient_3", "Marfan Syndrome"] == 1
    assert disease_matrix.loc["Patient_1", "Ehlers-Danlos syndrome"] != 1
    assert disease_matrix.loc["Patient_2", "Ehlers-Danlos syndrome"] == 1