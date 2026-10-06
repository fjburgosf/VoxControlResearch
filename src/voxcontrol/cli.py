"""Command-line interface (headless).

    voxcontrol predict "abre el navegador"
    voxcontrol audio command.wav
    voxcontrol benchmark configs/exp_main.yaml
    voxcontrol calibrate configs/exp_main.yaml --out model.pkl
    voxcontrol example example_03_ambiguity
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _model(args):
    from .api import VoxModel
    if getattr(args, "model", None) and Path(args.model).exists():
        return VoxModel.load(args.model)
    return VoxModel.train(seed=getattr(args, "seed", 0))


def _print_result(r) -> None:
    out = {k: v for k, v in r.as_dict().items() if k != "expected_costs"}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="voxcontrol", description="VoxControlResearch command-line interface")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("predict", help="classify a text command and decide")
    a.add_argument("text"); a.add_argument("--model"); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--user", default="default")
    a = sub.add_parser("audio", help="transcribe a WAV file and decide")
    a.add_argument("path"); a.add_argument("--model"); a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("benchmark", help="run an experiment configuration")
    a.add_argument("config"); a.add_argument("--results", default="results")
    a.add_argument("--seeds", type=int, nargs="*", help="override the seeds in the config")
    a = sub.add_parser("calibrate", help="train + calibrate a model from a configuration and save it")
    a.add_argument("config"); a.add_argument("--out", default="model.pkl"); a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("example", help="run a bundled example")
    a.add_argument("name", nargs="?")
    a = sub.add_parser("analyze", help="paired comparisons (UCIL vs each method) for an experiment folder")
    a.add_argument("experiment_dir"); a.add_argument("--reference", default="UCIL")
    sub.add_parser("gui", help="open the graphical interface")
    sub.add_parser("selftest", help="check the core, examples, experiments and every interface control")
    args = p.parse_args(argv)
    from .paths import configure_caches
    configure_caches()

    if args.cmd == "predict":
        _print_result(_model(args).process_text(args.text, user=args.user))
    elif args.cmd == "audio":
        _print_result(_model(args).process_audio(args.path))
    elif args.cmd == "benchmark":
        from .experiments.runner import load_config, run
        cfg = load_config(args.config)
        if args.seeds:
            cfg["experiment"]["seeds"] = args.seeds
        rec = run(cfg, args.results)
        print(f"results written to {rec.dir}")
    elif args.cmd == "calibrate":
        from .api import VoxModel
        from .experiments.runner import load_config
        m = VoxModel.train(seed=args.seed, config=load_config(args.config))
        m.save(args.out)
        u = m.ucil
        print(json.dumps({"saved": args.out, "temperature": u.temperature.temperature,
                          "calibrator": u.calibrator_name, "cv_brier": u.calibrator_selection,
                          "ood_detector": u.ood_name}, indent=2))
    elif args.cmd == "example":
        from .examples_catalog import EXAMPLES, run_example
        if not args.name:
            for k, e in EXAMPLES.items():
                print(f"{k:34s} {e.title['en']}")
            return 0
        print(run_example(args.name, lang="en"))
    elif args.cmd == "analyze":
        from .analysis import analyze
        for p in analyze(args.experiment_dir, args.reference):
            print(f"written {p}")
    elif args.cmd == "selftest":
        from .selftest import run as selftest
        code = selftest()
        print(f"self-test {'passed' if code == 0 else 'FAILED'}; see selftest_report.txt")
        return code
    elif args.cmd == "gui":
        from .gui.app import main as gui_main
        gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
