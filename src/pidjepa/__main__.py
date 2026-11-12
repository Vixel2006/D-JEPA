"""Allow ``python -m pidjepa`` to run the benchmark."""

from pidjepa.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
