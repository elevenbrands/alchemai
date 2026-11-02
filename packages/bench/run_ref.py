"""Run the reference cases: solve, verify balances, apply literature checks, snapshot readouts.

    uv run python packages/bench/run_ref.py            # run all, compare with expected_ref.yaml
    uv run python packages/bench/run_ref.py --update   # (re)write expected_ref.yaml from DWSIM
    uv run python packages/bench/run_ref.py bomba reciclo
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.bench.cases_ref import CASES, BY_ID  # noqa: E402
from packages.dwsim_client.verify import verify  # noqa: E402

EXPECTED = Path(__file__).with_name("expected_ref.yaml")
REL_TOL = 1e-3


def run_case(case, expected: dict | None):
    fs, h = case.build()
    rep = verify(fs)
    readouts = {k: float(v) for k, v in case.readouts(h).items()} if rep.solved else {}
    checks = case.checks(h, rep) if rep.solved else [("solved", False, "; ".join(rep.solver_errors))]
    regress = []
    if expected:
        for k, v in expected.items():
            got = readouts.get(k)
            ok = got is not None and abs(got - v) <= REL_TOL * max(abs(v), 1e-9) + 1e-9
            regress.append((k, ok, f"{got} vs {v}"))
    ok = rep.ok and all(c[1] for c in checks) and all(r[1] for r in regress)
    return ok, rep, readouts, checks, regress


def main(argv):
    update = "--update" in argv
    ids = [a for a in argv if not a.startswith("--")] or [c.id for c in CASES]
    expected_all = yaml.safe_load(EXPECTED.read_text()) if EXPECTED.exists() else {}
    new_expected = dict(expected_all)
    all_ok = True
    for cid in ids:
        case = BY_ID[cid]
        ok, rep, readouts, checks, regress = run_case(case, None if update else expected_all.get(cid))
        all_ok &= ok
        print(f"\n## {cid} — {case.title_es}  {'✅' if ok else '❌'}")
        print(rep.markdown())
        for name, c_ok, detail in checks:
            print(f"- {'✅' if c_ok else '❌'} {name}  {detail}")
        for name, r_ok, detail in regress:
            print(f"- {'✅' if r_ok else '❌'} regresión {name}: {detail}")
        print("readouts:", {k: round(v, 6) for k, v in readouts.items()})
        if update and rep.solved:
            new_expected[cid] = readouts
    if update:
        EXPECTED.write_text("# DWSIM 10.2.12 reference readouts (PR / NRTL / IAPWS-IF97). Regenerate with --update.\n"
                            + yaml.safe_dump(new_expected, sort_keys=True, allow_unicode=True))
        print(f"\nwrote {EXPECTED}")
    print("\nALL OK" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
