"""The ten reference flowsheets: solve in DWSIM 10, pass the verifier, pass textbook checks and
match the DWSIM snapshot in packages/bench/expected_ref.yaml (regression band 0.1 %)."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.bench.cases_ref import CASES, BY_ID  # noqa: E402
from packages.bench.run_ref import EXPECTED, run_case  # noqa: E402

EXPECTED_ALL = yaml.safe_load(EXPECTED.read_text())


@pytest.mark.parametrize("cid", [c.id for c in CASES])
def test_reference_case(cid):
    case = BY_ID[cid]
    ok, rep, readouts, checks, regress = run_case(case, EXPECTED_ALL.get(cid))
    failed = [f"{n}: {d}" for n, k, d in checks + regress if not k]
    assert rep.solved, rep.solver_errors
    assert rep.ok, rep.markdown()
    assert ok, "\n".join(failed)


def test_every_case_has_snapshot():
    assert set(EXPECTED_ALL) == {c.id for c in CASES}
