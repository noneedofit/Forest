"""Load the scripted scammer scenarios and fill in the identifiers they hand out."""

from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent / "scenarios"


def load_all() -> dict[str, dict]:
    """Scenario id -> scenario, for every YAML here except the alias list."""
    out = {}
    for path in sorted(DIR.glob("*.yaml")):
        if path.name == "aliases.yaml":
            continue
        data = yaml.safe_load(path.read_text())
        out[data["id"]] = data
    return out


def spoken(handle: str) -> str:
    """How a caller reads a UPI id aloud: 'name dot x at the rate okaxis'."""
    name, psp = handle.split("@")
    return f"{name.replace('.', ' dot ')} at the rate {psp}"


def fields(item: dict) -> dict[str, str]:
    """Placeholder values for one `collect` entry."""
    value = item["value"]
    if item["kind"] == "upi":
        return {"upi": value, "spoken": spoken(value)}
    spaced = " ".join(value[i : i + 4] for i in range(0, len(value), 4))
    return {"account": value, "account_spaced": spaced, "ifsc": item["ifsc"]}


def render(line: str, item: dict | None) -> str:
    """Fill a line's placeholders from `item`; lines without placeholders pass through."""
    return line.format(**fields(item)) if item else line
