"""RANIntent v1 corpus builder CLI (file exchange; no LLM calls from here).

  python -m src.ranbench.corpus.cli [--exchange-dir D] [--data-dir D] <command>

  write-schemas         regenerate src/ranbench/schemas/* from spec
  tuples                write the sampled tuples of both splits to <data-dir>/_tuples/<split>.jsonl (preview)
  export --stage gen|ver [--stub]   export todo batches; --stub also answers them with the stub generator/verifier
  ingest --stage gen|ver [--no-diversity]
  status                per-split progress
  assemble [--allow-stub]           build and freeze the corpus under <data-dir> with manifest.json
  dry-run [--rounds N]  stub gen -> ingest -> stub ver -> ingest (no diversity caps) until done, then assemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.ranbench.corpus.assemble import assemble_corpus, write_schema_files
from src.ranbench.corpus.exchange import RanExchange
from src.ranbench.corpus.spec import load_config
from src.ranbench.corpus.stub import write_stub_gen_done, write_stub_ver_done
from src.ranbench.corpus.tuples import sample_tuples

DEFAULT_DATA_DIR = "data/ranbench/ranintent-v1"
DEFAULT_EXCHANGE_DIR = "runs/_ranintent_exchange"


def dry_run(exchange: RanExchange, rounds: int = 10) -> dict:
    """Drive the whole exchange with stubs until every tuple is accepted, then assemble (allow_stub)."""
    for _ in range(rounds):
        exchange.export_gen()
        write_stub_gen_done(exchange)
        exchange.ingest_gen()
        exchange.export_ver()
        write_stub_ver_done(exchange.ver_todo_dir, exchange.ver_done_dir)
        exchange.ingest_ver(check_diversity=False)
        if all(it.status in ("accepted", "rejected") for it in exchange.items.values()):
            break
    return assemble_corpus(exchange, allow_stub=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="RANIntent v1 corpus builder")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--exchange-dir", default=DEFAULT_EXCHANGE_DIR)
    p.add_argument("--config", default=None, help="Config JSON (default: configs/ranbench/ranintent_v1.json)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write-schemas")
    sub.add_parser("tuples")
    pe = sub.add_parser("export")
    pe.add_argument("--stage", required=True, choices=["gen", "ver"])
    pe.add_argument("--split", default="all", choices=["test", "dev", "all"])
    pe.add_argument("--batch", type=int, default=100)
    pe.add_argument("--stub", action="store_true", help="Answer the exported batches with the stub (dry run)")
    pi = sub.add_parser("ingest")
    pi.add_argument("--stage", required=True, choices=["gen", "ver"])
    pi.add_argument("--no-diversity", action="store_true", help="Skip the diversity caps (stub dry runs only)")
    sub.add_parser("status")
    pa = sub.add_parser("assemble")
    pa.add_argument("--allow-stub", action="store_true")
    pd = sub.add_parser("dry-run")
    pd.add_argument("--rounds", type=int, default=10)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.cmd == "write-schemas":
        for path in write_schema_files(cfg):
            print(f"wrote {path}")
        return
    if args.cmd == "tuples":
        out = Path(args.data_dir) / "_tuples"
        out.mkdir(parents=True, exist_ok=True)
        for sp in cfg["splits"]:
            items = sample_tuples(sp, cfg)
            (out / f"{sp}.jsonl").write_text("".join(json.dumps(t.to_dict()) + "\n" for t in items), encoding="utf-8")
            print(f"{sp}: {len(items)} tuples -> {out / f'{sp}.jsonl'}")
        return

    ex = RanExchange(exchange_dir=args.exchange_dir, data_dir=args.data_dir, cfg=cfg)
    if args.cmd == "export":
        splits = None if args.split == "all" else [args.split]
        if args.stage == "gen":
            files = ex.export_gen(splits=splits, gen_batch_size=args.batch)
            if args.stub:
                write_stub_gen_done(ex)
        else:
            files = ex.export_ver(splits=splits, ver_batch_size=args.batch)
            if args.stub:
                write_stub_ver_done(ex.ver_todo_dir, ex.ver_done_dir)
        print(f"exported {len(files)} file(s)" + (" and answered them with the stub" if args.stub else ""))
    elif args.cmd == "ingest":
        res = ex.ingest_gen() if args.stage == "gen" else ex.ingest_ver(check_diversity=not args.no_diversity)
        print(json.dumps(res, indent=2, default=dict))
    elif args.cmd == "status":
        print(ex.print_status())
        print(json.dumps(ex.yields(), indent=2))
    elif args.cmd == "assemble":
        m = assemble_corpus(ex, allow_stub=args.allow_stub)
        print(f"assembled {len(m['files'])} files into {args.data_dir} (stub={m['stub']})")
    elif args.cmd == "dry-run":
        m = dry_run(ex, rounds=args.rounds)
        print(f"dry run assembled {len(m['files'])} files into {args.data_dir} (stub={m['stub']})")


if __name__ == "__main__":
    sys.exit(main())
