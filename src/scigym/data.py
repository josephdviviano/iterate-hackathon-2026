"""SciGym systems (Duan et al., 2025): a BioModels network with its reactions removed.

Each system has the partial SBML the agent starts from (species, compartments,
units, the parameters that are not reaction-specific), the reference SBML
the environment simulates, and a SED-ML time course. Identifiers are
scrambled in the dataset. The parquet is fetched once into cache/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import libsbml

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "scigym"
URL = "https://huggingface.co/datasets/h4duan/scigym-sbml/resolve/main/data/{split}-00000-of-00001.parquet"


@dataclass(frozen=True)
class Reaction:
    id: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    modifiers: tuple[str, ...]

    @property
    def key(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Reactant and product multisets, the match unit of SciGym's reaction score."""
        return tuple(sorted(self.reactants)), tuple(sorted(self.products))


@dataclass(frozen=True)
class System:
    id: str
    partial: str
    truth: str
    sedml: str
    species: tuple[str, ...]
    initial: dict[str, float]
    compartments: tuple[str, ...]
    parameters: dict[str, float]
    t_end: float
    n_steps: int
    truth_reactions: tuple[Reaction, ...]


def _course(sedml: str) -> tuple[float, int]:
    m = re.search(r'<uniformTimeCourse[^>]*outputEndTime="([^"]+)"[^>]*numberOfSteps="(\d+)"', sedml)
    return (float(m.group(1)), int(m.group(2))) if m else (10.0, 100)


def read_model(xml: str) -> libsbml.Model:
    doc = libsbml.readSBMLFromString(xml)
    if doc.getModel() is None:
        raise ValueError("unreadable SBML")
    return doc.getModel()


def reactions_of(model: libsbml.Model) -> tuple[Reaction, ...]:
    out = []
    for r in model.getListOfReactions():
        out.append(Reaction(r.getId(),
                            tuple(s.getSpecies() for s in r.getListOfReactants() for _ in range(max(1, round(s.getStoichiometry() or 1)))),
                            tuple(s.getSpecies() for s in r.getListOfProducts() for _ in range(max(1, round(s.getStoichiometry() or 1)))),
                            tuple(m.getSpecies() for m in r.getListOfModifiers())))
    return tuple(out)


def load_systems(split: str = "small", limit: int = 0) -> list[System]:
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}.parquet"
    if not path.exists():
        import httpx

        path.write_bytes(httpx.get(URL.format(split=split), follow_redirects=True, timeout=120).content)
    df = pd.read_parquet(path).sort_values("folder_name")
    out = []
    for _, row in df.iterrows():
        pm = read_model(row["partial"])
        species = tuple(s.getId() for s in pm.getListOfSpecies())
        initial = {s.getId(): (s.getInitialConcentration() if s.isSetInitialConcentration() else s.getInitialAmount())
                   for s in pm.getListOfSpecies()}
        params = {p.getId(): p.getValue() for p in pm.getListOfParameters()}
        t_end, n_steps = _course(row["truth_sedml"])
        out.append(System(row["folder_name"], row["partial"], row["truth_xml"], row["truth_sedml"], species, initial,
                          tuple(c.getId() for c in pm.getListOfCompartments()), params, t_end, n_steps,
                          reactions_of(read_model(row["truth_xml"]))))
        if limit and len(out) >= limit:
            break
    return out
