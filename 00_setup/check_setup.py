"""
check_setup.py — a friendly pre-flight check for Day -1.

Run:  python 00_setup/check_setup.py

It verifies, in order:
  1. Python is new enough
  2. The core packages are installed
  3. A .env file exists with a key for your chosen provider
  4. The model actually answers

Each check prints PASS/FAIL and, on failure, exactly what to do next.
"""

import sys
import os

# Make the repo root importable so "from providers import ..." works no matter
# what directory you launched this from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN, RED, DIM, RESET = "\033[92m", "\033[91m", "\033[2m", "\033[0m"


def ok(msg):   print(f"{GREEN}  PASS{RESET}  {msg}")
def bad(msg):  print(f"{RED}  FAIL{RESET}  {msg}")
def hint(msg): print(f"{DIM}        -> {msg}{RESET}")


def check_python():
    if sys.version_info >= (3, 10):
        ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")
        return True
    bad(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old")
    hint("This course needs Python 3.10+. Install a newer Python and rebuild your venv.")
    return False


def check_packages():
    import importlib.util
    all_ok = True
    for pkg in ("langchain", "langgraph", "dotenv", "pydantic"):
        if importlib.util.find_spec(pkg) is None:
            bad(f"package '{pkg}' not installed")
            hint("Run: pip install -r requirements.txt")
            all_ok = False
        else:
            ok(f"package '{pkg}'")
    return all_ok


def check_env():
    from dotenv import load_dotenv
    load_dotenv()

    provider = os.getenv("PROVIDER", "groq").lower()
    key_var = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY", "fm": "FM_API_KEY"}.get(provider)

    if key_var is None:
        bad(f"PROVIDER='{provider}' is not one of: groq, gemini, fm")
        return False

    value = os.getenv(key_var, "")
    if not value or value.startswith("your_"):
        bad(f"{key_var} is missing (provider is '{provider}')")
        hint("Copy .env.example to .env and paste your real key.")
        return False

    ok(f"PROVIDER='{provider}', {key_var} is set")
    return True


def check_model_call():
    from providers import get_chat_model
    try:
        model = get_chat_model()
        reply = model.invoke("Reply with the single word: ready")
        ok(f"model answered: {reply.content.strip()!r}")
        return True
    except Exception as e:
        bad("the model call failed")
        hint(f"{type(e).__name__}: {e}")
        return False


def main():
    print("\nDay -1 setup check\n" + "-" * 20)
    steps = [check_python, check_packages, check_env, check_model_call]
    results = []
    for step in steps:
        results.append(step())
        # Stop early if a prerequisite failed — later checks would just be noise.
        if not results[-1]:
            break

    print("-" * 20)
    if all(results) and len(results) == len(steps):
        print(f"{GREEN}All checks passed — you're ready for Day 1.{RESET}\n")
        return 0
    print(f"{RED}Fix the FAIL above, then run this again.{RESET}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
