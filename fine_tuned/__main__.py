"""CLI entry point for fine_tuned package.

Usage:
    python -m fine_tuned extract_features --config fine_tuned/configs/extract_halt_medvqa.yaml
    python -m fine_tuned train --config fine_tuned/configs/train_stage2_hallucination.yaml
    python -m fine_tuned eval --config fine_tuned/configs/eval_stage2_hallucination.yaml
"""

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m fine_tuned {extract_features|train|eval} --config <path>")
        sys.exit(1)

    command = sys.argv[1]
    argv = sys.argv[2:]

    if command == "extract_features":
        from fine_tuned.extract_hallucination_features import main as run
    elif command == "train":
        from fine_tuned.train_hallucination import main as run
    elif command == "eval":
        from fine_tuned.eval_hallucination import main as run
    else:
        print(f"Unknown command: {command}")
        print("Available: extract_features, train, eval")
        sys.exit(1)

    run(argv)


if __name__ == "__main__":
    main()
