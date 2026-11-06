"""Bloque 29: bench completo (50 tarjetas) con Haiku, Sonnet y Opus vía claude-code. Resume notebooks/bench_runs/m2/*.jsonl."""
import json,yaml,collections,pathlib
idx={c["id"]:c for c in yaml.safe_load(open("packages/bench/cards/index.yaml"))["cards"]}
for m in ["haiku","sonnet","opus"]:
    rows=[json.loads(l) for l in open(f"notebooks/bench_runs/m2/all_cc_{m}.jsonl")]
    by=collections.defaultdict(list); t=0; cost=0; turns=0
    for r in rows:
        v=idx[r["id"]]["version"]; by[v].append(r["score"]); t+=r["seconds"]
        ev=(r.get("response") or {}).get("evidence") or {}
        cost+=ev.get("cost_usd_reported") or 0; turns+=ev.get("num_turns") or 0
    print(m, f"{100*sum(r['score'] for r in rows)/len(rows):.1f}", f"t={t:.0f}s", f"cost={cost:.2f}", f"turns={turns}",
          {v:f"{100*sum(s)/len(s):.0f}%/{len(s)}" for v,s in sorted(by.items())})
    for r in rows:
        if r["score"]<1:
            bad=[x for x in r.get("rows",[]) if x[-1]!="ok"]
            print("   ",r["id"],idx[r["id"]]["version"],r["score"],f"{r['seconds']:.0f}s",bad[:3], r.get("status",""), (r.get("missing_evidence") or ""))
