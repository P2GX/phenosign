import pandas as pd
import pytest
from types import SimpleNamespace

from ppkt2synergy.core import PhenopacketRecord
from ppkt2synergy.io import EnrichedPhenopacket
from ppkt2synergy.preprocessing.dataset_builder import PhenotypeDatasetBuilder


class DummyPhenopacket:
    def __init__(self, ppkt_id):
        self.id = ppkt_id


class DummyHpoFeatureData:
    def __init__(self):
        self.matrix = pd.DataFrame(
            {"HP:1": [1.0, 0.0]},
            index=["P1", "P2"],
        )
        self.label_mapping = {"HP:1": "Label 1"}
        self.relationship_mask = pd.DataFrame(
            [[float("nan")]],
            index=["HP:1"],
            columns=["HP:1"],
        )


class DummyTargetData:
    def __init__(self):
        self.disease_matrix = pd.DataFrame(
            {"DiseaseA": [1.0, 0.0]},
            index=["P1", "P2"],
        )
        self.variant_condition_matrix = None


class DummyHpoFeatureBuilder:
    def __init__(self, records, hpo_hierarchy):
        self.records = records
        self.hpo_hierarchy = hpo_hierarchy
        self.called_with_missing_threshold = None

    def build(self, missing_threshold=1.0):
        self.called_with_missing_threshold = missing_threshold
        return DummyHpoFeatureData()


class DummyTargetDataBuilder:
    def __init__(self, records, raw_phenopackets, hpo_hierarchy):
        self.records = records
        self.raw_phenopackets = raw_phenopackets
        self.hpo_hierarchy = hpo_hierarchy
        self.called_with_variant_effect_type = None
        self.called_with_mane_tx_id = None

    def build(self, variant_effect_type=None, mane_tx_id=None):
        self.called_with_variant_effect_type = variant_effect_type
        self.called_with_mane_tx_id = mane_tx_id
        return DummyTargetData()


class DummyHierarchyEngine:
    def __init__(self, hpo_file=None, release=None):
        self.hpo_file = hpo_file
        self.release = release


@pytest.fixture
def records():
    return [
        PhenopacketRecord(
            individual_id="P1",
            cohort="C1",
            observed_hpo_terms={"HP:1"},
            excluded_hpo_terms=set(),
            observed_diseases={"DiseaseA"},
            excluded_diseases=set(),
            sex="male",
            age="P10Y",
            metadata={"pmids": ["12345"]},
        ),
        PhenopacketRecord(
            individual_id="P2",
            cohort="C2",
            observed_hpo_terms={"HP:2"},
            excluded_hpo_terms={"HP:3"},
            observed_diseases=set(),
            excluded_diseases={"DiseaseA"},
            sex="female",
            age=None,
            metadata={"pmids": []},
        ),
    ]


def test_init_empty_phenopackets_raises():
    with pytest.raises(ValueError) as excinfo:
        PhenotypeDatasetBuilder([])

    assert "phenopackets cannot be empty" in str(excinfo.value)


def test_init_with_raw_phenopackets(monkeypatch, records):
    from ppkt2synergy.preprocessing import dataset_builder as module

    monkeypatch.setattr(module.ppkt, "Phenopacket", DummyPhenopacket)
    monkeypatch.setattr(module, "phenopackets_to_records", lambda x: records)
    monkeypatch.setattr(module, "HPOHierarchyEngine", DummyHierarchyEngine)

    phenopackets = [DummyPhenopacket("P1"), DummyPhenopacket("P2")]
    builder = PhenotypeDatasetBuilder(phenopackets)

    assert builder.raw_phenopackets == phenopackets
    assert builder.records == records
    assert isinstance(builder._hpo_engine, DummyHierarchyEngine)


def test_init_with_enriched_phenopackets(monkeypatch, records):
    from ppkt2synergy.preprocessing import dataset_builder as module

    monkeypatch.setattr(module.ppkt, "Phenopacket", DummyPhenopacket)
    monkeypatch.setattr(module, "enriched_phenopackets_to_records", lambda x: records)
    monkeypatch.setattr(module, "HPOHierarchyEngine", DummyHierarchyEngine)

    phenopackets = [
        EnrichedPhenopacket(phenopacket=DummyPhenopacket("P1"), cohort="C1"),
        EnrichedPhenopacket(phenopacket=DummyPhenopacket("P2"), cohort="C2"),
    ]
    builder = PhenotypeDatasetBuilder(phenopackets)

    assert [p.id for p in builder.raw_phenopackets] == ["P1", "P2"]
    assert builder.records == records
    assert isinstance(builder._hpo_engine, DummyHierarchyEngine)


def test_init_with_mixed_input_raises(monkeypatch):
    from ppkt2synergy.preprocessing import dataset_builder as module

    monkeypatch.setattr(module.ppkt, "Phenopacket", DummyPhenopacket)
    monkeypatch.setattr(module, "HPOHierarchyEngine", DummyHierarchyEngine)

    mixed = [
        DummyPhenopacket("P1"),
        EnrichedPhenopacket(phenopacket=DummyPhenopacket("P2"), cohort="C2"),
    ]

    with pytest.raises(TypeError) as excinfo:
        PhenotypeDatasetBuilder(mixed)

    assert "`phenopackets` must be a list of `Phenopacket` or `EnrichedPhenopacket` objects." in str(excinfo.value)


def test_build_individual_metadata(monkeypatch, records):
    from ppkt2synergy.preprocessing import dataset_builder as module

    monkeypatch.setattr(module.ppkt, "Phenopacket", DummyPhenopacket)
    monkeypatch.setattr(module, "phenopackets_to_records", lambda x: records)
    monkeypatch.setattr(module, "HPOHierarchyEngine", DummyHierarchyEngine)

    builder = PhenotypeDatasetBuilder([DummyPhenopacket("P1"), DummyPhenopacket("P2")])
    metadata = builder.build_individual_metadata()

    assert isinstance(metadata, pd.DataFrame)
    assert list(metadata.index) == ["P1", "P2"]
    assert list(metadata.columns) == ["cohort", "sex", "age", "pmids"]

    assert metadata.loc["P1", "cohort"] == "C1"
    assert metadata.loc["P1", "sex"] == "male"
    assert metadata.loc["P1", "age"] == "P10Y"
    assert metadata.loc["P1", "pmids"] == ["12345"]

    assert metadata.loc["P2", "cohort"] == "C2"
    assert metadata.loc["P2", "sex"] == "female"
    assert pd.isna(metadata.loc["P2", "age"])
    assert metadata.loc["P2", "pmids"] == []


def test_build_calls_sub_builders_and_returns_dataset(monkeypatch, records):
    from ppkt2synergy.preprocessing import dataset_builder as module

    captured = {}

    monkeypatch.setattr(module.ppkt, "Phenopacket", DummyPhenopacket)
    monkeypatch.setattr(module, "phenopackets_to_records", lambda x: records)
    monkeypatch.setattr(module, "HPOHierarchyEngine", DummyHierarchyEngine)

    def fake_hpo_builder(records, hpo_hierarchy):
        obj = DummyHpoFeatureBuilder(records, hpo_hierarchy)
        captured["feature_builder"] = obj
        return obj

    def fake_target_builder(records, raw_phenopackets, hpo_hierarchy):
        obj = DummyTargetDataBuilder(records, raw_phenopackets, hpo_hierarchy)
        captured["target_builder"] = obj
        return obj

    monkeypatch.setattr(module, "HpoFeatureBuilder", fake_hpo_builder)
    monkeypatch.setattr(module, "TargetDataBuilder", fake_target_builder)

    phenopackets = [DummyPhenopacket("P1"), DummyPhenopacket("P2")]
    builder = PhenotypeDatasetBuilder(phenopackets)

    variant_effect = SimpleNamespace(name="MISSENSE")
    dataset = builder.build(
        variant_effect_type=variant_effect,
        mane_tx_id=["NM_000001.1", "NM_000002.1"],
        missing_threshold=0.5,
    )

    assert captured["feature_builder"].called_with_missing_threshold == 0.5
    assert captured["target_builder"].called_with_variant_effect_type == variant_effect
    assert captured["target_builder"].called_with_mane_tx_id == ["NM_000001.1", "NM_000002.1"]

    assert hasattr(dataset, "hpo_data")
    assert hasattr(dataset, "targets")
    assert hasattr(dataset, "individual_metadata")

    assert list(dataset.individual_metadata.index) == ["P1", "P2"]


def test_build_passes_raw_phenopackets_to_target_builder(monkeypatch, records):
    from ppkt2synergy.preprocessing import dataset_builder as module

    captured = {}

    monkeypatch.setattr(module.ppkt, "Phenopacket", DummyPhenopacket)
    monkeypatch.setattr(module, "phenopackets_to_records", lambda x: records)
    monkeypatch.setattr(module, "HPOHierarchyEngine", DummyHierarchyEngine)

    monkeypatch.setattr(module, "HpoFeatureBuilder", lambda records, hpo_hierarchy: DummyHpoFeatureBuilder(records, hpo_hierarchy))

    def fake_target_builder(records, raw_phenopackets, hpo_hierarchy):
        captured["raw_phenopackets"] = raw_phenopackets
        return DummyTargetDataBuilder(records, raw_phenopackets, hpo_hierarchy)

    monkeypatch.setattr(module, "TargetDataBuilder", fake_target_builder)

    phenopackets = [DummyPhenopacket("P1"), DummyPhenopacket("P2")]
    builder = PhenotypeDatasetBuilder(phenopackets)
    builder.build()

    assert captured["raw_phenopackets"] == phenopackets