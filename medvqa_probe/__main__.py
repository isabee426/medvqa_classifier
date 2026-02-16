"""Allow running sub-commands via `python -m medvqa_probe <command>`."""

from __future__ import annotations

import sys


def main() -> None:
    usage = (
        "Usage: python -m medvqa_probe <command> [args]\n"
        "\n"
        "Commands:\n"
        "  extract_features   Extract internal-state features from the VQA backbone\n"
        "  train_classifier   Train the MLP hallucination classifier\n"
        "  eval_classifier    Evaluate a trained classifier checkpoint\n"
    )
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift so argparse in sub-modules works normally

    if command == "extract_features":
        from medvqa_probe.extract_features import main as cmd_main
    elif command == "train_classifier":
        from medvqa_probe.train_classifier import main as cmd_main
    elif command == "eval_classifier":
        from medvqa_probe.eval_classifier import main as cmd_main
    else:
        print(f"Unknown command: {command}\n")
        print(usage)
        sys.exit(1)

    cmd_main()


if __name__ == "__main__":
    main()
