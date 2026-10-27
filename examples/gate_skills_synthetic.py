"""Build tiny, entirely invented gate inputs outside the repository.

These are arithmetic fixtures, not fitted research results or licensed data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def module(name):
    path = REPO / f"skills/sc-{name.replace('_', '-')}/scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def loss_rows(new=5.):
    rows = []
    for year in range(2015, 2022):
        day = date(year, 1, 15)
        while day.isocalendar().week % 2 != 1:
            day += timedelta(days=1)
        for offset in (0, 14):
            current = day + timedelta(days=offset)
            rows.append(dict(row_id=f"toy-{year}-{offset}", date=current.isoformat(), fit_parity=0,
                             n=10, sse_incumbent=10., sse_candidate=new))
    return rows


def declaration():
    return dict(declared_at="2026-01-01T00:00:00Z", evaluated_at="2026-01-02T00:00:00Z",
                predicted_sign="+", estimate_split="odd", pooled=dict(estimate=.2),
                years={str(y): dict(estimate=.1 if y < 2019 else -.1, t_day=3. if y < 2019 else -3.,
                                    t_week=3. if y < 2019 else -3.) for y in range(2015, 2022)})


def peer_contract():
    return dict(operator="synthetic neighbour mean", same_operator=True, exclude_self_and_supply_links=True,
                shared_missingness_controls=True, point_in_time=True, size_factor=2, industries=["indpeer"])


def peer_rows():
    return [dict(row_id=r["row_id"], date=r["date"], fit_parity=0, n=r["n"], sse_graph=5., sse_indpeer=7.)
            for r in loss_rows()]


def graph(directed=True):
    return dict(directed=directed, nodes={n: "toy-industry" for n in "ABCDEFGH"},
                edges=[dict(source=a, target=b, period="2015-01", old=True)
                       for a, b in (("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"))])


def placebo(draws=100):
    rewire = module("gate_rewire_placebo")
    original = graph()
    items = []
    for i in range(draws):
        changed, _ = rewire.rewire(original, seed=20261026+i)
        items.append(dict(seed=20261026+i, graph=changed, rows=loss_rows(9.5)))
    return dict(graph=original, observed_rows=loss_rows(), draws=items,
                operator="synthetic neighbour mean", same_operator_refit=True)


def write(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        path = directory / name
        with path.open("x", encoding="utf-8", newline="") as stream:
            if path.suffix == ".csv":
                writer = csv.DictWriter(stream, fieldnames=list(value[0]))
                writer.writeheader(); writer.writerows(value)
            else:
                json.dump(value, stream, indent=2, allow_nan=False)
    save("validation.csv", loss_rows())
    save("peers.csv", peer_rows())
    save("peer_contract.json", peer_contract())
    save("sign.json", declaration())
    save("placebo.json", placebo())
    snapshot = dict(nodes=list("ABCD"), vintage="2020-01-01", monthly_universe={"2020-02": list("ABCD")},
        edges=[dict(source=a, target=b, vintage="2020-01-01", valid_from="2020-01-01")
               for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"))])
    save("snapshot.json", snapshot)
    save("router_case.json", dict(case_id="toy-router", firm_id="A", cutoff="2020-02-01",
                                  candidate_neighbours=["B", "C"], inspections=[]))
    save("acceptance.json", dict(attempt_id="toy-pass-1", profile="agent-v4", validation="validation.csv",
        sign_stability="sign.json", industry_peer=dict(rows="peers.csv", contract="peer_contract.json"),
        rewire_placebo="placebo.json", placebo_threshold=.05, draws=100, two_sided=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    write(parser.parse_args().out)
