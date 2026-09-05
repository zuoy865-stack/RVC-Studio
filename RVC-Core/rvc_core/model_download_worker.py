"""Small worker used by the native UI to download one PyMSS model."""

from __future__ import annotations

import argparse

from pymss.model_download import download_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    paths = download_model(
        args.model,
        model_dir=args.model_dir,
        source="huggingface",
    )
    print("PyMSS model ready:")
    if isinstance(paths, dict):
        for name, path in paths.items():
            print(f"  {name}: {path}")
    else:
        print(f"  {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
