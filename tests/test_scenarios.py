import yaml

from forest.honeypot.scenario import DIR, load_all, render
from forest.intel.extract import extract
from forest.intel.labels import KINDS, SCAM_TYPES

SCENARIOS = load_all()


def alias_handles() -> set[str]:
    return set(yaml.safe_load((DIR / "aliases.yaml").read_text())["handles"])


def test_five_scenarios_present():
    assert len(SCENARIOS) == 5


def test_metadata_is_valid():
    for sc in SCENARIOS.values():
        assert sc["reported"] is True
        assert sc["scam_type"] in SCAM_TYPES
        assert all(item["kind"] in KINDS for item in sc["collect"])


def test_upi_handles_come_from_aliases():
    known = alias_handles()
    for sc in SCENARIOS.values():
        for item in sc["collect"]:
            if item["kind"] == "upi":
                assert item["value"] in known, (sc["id"], item["value"])


def test_every_scenario_reveals_at_least_one_upi():
    for sc in SCENARIOS.values():
        assert any(item["kind"] == "upi" for item in sc["collect"])


def test_every_line_renders_and_the_extractor_finds_what_it_reveals():
    for sc in SCENARIOS.values():
        for step in sc["steps"]:
            item = sc["collect"][step["reveal"]] if "reveal" in step else None
            assert len(step["lines"]) >= 2
            for line in step["lines"]:
                text = render(line, item)
                if item is None:
                    assert "{" not in text
                    continue
                found = {e.normalized for e in extract(text)}
                want = item["value"].lower() if item["kind"] == "upi" else item["value"]
                assert want in found, (sc["id"], text, found)
