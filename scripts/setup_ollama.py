#!/usr/bin/env python3
"""Pull required Ollama models for local inference."""

import subprocess
import sys

MODELS = {
    "generation": "mistral:7b",
    "classification": "llama3.2:3b",
}


def check_ollama_running() -> bool:
    """Check if Ollama server is accessible."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False


def pull_model(model: str) -> bool:
    """Pull a single model from Ollama registry."""
    print(f"Pulling {model}...")
    result = subprocess.run(
        ["ollama", "pull", model],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Failed to pull {model}: {result.stderr}", file=sys.stderr)
        return False
    print(f"✓ {model} ready")
    return True


def main() -> int:
    """Entry point."""
    if not check_ollama_running():
        print(
            "ERROR: Ollama is not running or not installed.\n"
            "Install: https://ollama.com/download\n"
            "Start: ollama serve",
            file=sys.stderr,
        )
        return 1

    print("Ollama is running. Checking models...\n")
    success = True
    for purpose, model in MODELS.items():
        if not pull_model(model):
            success = False
        else:
            print(f"  → {purpose}: {model}")

    if success:
        print("\nAll models ready.")
        return 0
    else:
        print("\nSome models failed to pull.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
