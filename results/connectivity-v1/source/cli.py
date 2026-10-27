import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Quantum generative learning experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--workers", type=int, default=1)
    audit = commands.add_parser("audit")
    audit.add_argument("--output", required=True)
    report = commands.add_parser("report")
    report.add_argument("--results", nargs="+", required=True)
    report.add_argument("--output", default="docs")
    args = parser.parse_args()
    if args.command == "run":
        from .experiment import run_experiment
        run_experiment(args.config, args.output, args.workers)
    elif args.command == "audit":
        from .audit import audit_experiment
        print(json.dumps(audit_experiment(args.output), indent=2))
    else:
        from .report import build_report
        build_report(args.results, args.output)


if __name__ == "__main__":
    main()

