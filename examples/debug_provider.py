"""
examples/debug_provider.py  —  test your LLM provider directly.

Run this to see the exact error if basic_sync.py fails:
  python examples/debug_provider.py
"""

import asyncio
import os
import sys

# Load saved keys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from helix.config_store import apply_saved_config, best_available_model

apply_saved_config()


async def test_gemini():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("✗ GOOGLE_API_KEY not set. Run: helix config set GOOGLE_API_KEY your-key")
        return False

    print(f"  Key: {key[:8]}...{key[-4:]}")

    # Test 1: import
    try:
        import google.generativeai as genai

        print(f"  SDK version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
    except ImportError:
        print("✗ google-generativeai not installed. Run: pip install google-generativeai")
        return False

    # Test 2: configure
    genai.configure(api_key=key)
    print("  Configured SDK ✓")

    # Test 3: list models (validates key)
    try:
        models = [
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        gemini_models = [m for m in models if "gemini" in m]
        print(f"  Models available: {len(gemini_models)} gemini models")
        if gemini_models:
            print(f"  First few: {gemini_models[:3]}")
    except Exception as e:
        print(f"✗ list_models() failed: {type(e).__name__}: {e}")
        print("  This usually means the API key is invalid or has no permissions.")
        return False

    # Test 4: actual generation — prefer models from the live list, fall back to known names
    # Build candidate list: live models first, then static fallbacks
    live_candidates = [m for m in gemini_models if "flash" in m and "2.0" in m]
    live_candidates += [m for m in gemini_models if "flash" in m and "2.5" in m]
    live_candidates += [m for m in gemini_models if "flash" in m]
    live_candidates += [m for m in gemini_models if "pro" in m]
    # Deduplicate while preserving order
    seen: set = set()
    ordered: list = []
    for m in live_candidates:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    # Append short-form fallbacks in case live list uses full paths
    for fallback in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-lite"]:
        if fallback not in seen:
            ordered.append(fallback)
    model_names_to_try = ordered[:6]  # cap at 6 attempts
    for model_name in model_names_to_try:
        try:
            print(f"\n  Trying model: {model_name}")
            client = genai.GenerativeModel(model_name=model_name)
            response = client.generate_content("Say 'Hello from Helix!' and nothing else.")
            text = response.text
            print(f"  Response: {text.strip()}")
            print(f"\n✓ Gemini working! Use model name: '{model_name}'")
            return model_name
        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}")

    print("\n✗ All model names failed.")
    return False


async def main():
    print("=" * 60)
    print("Helix Provider Debug")
    print("=" * 60)

    print(f"\nAuto-detected model: {best_available_model()}")
    print()

    print("Testing Gemini (Google):")
    result = await test_gemini()

    if result and result is not True:
        # Got a working model name
        working_model = result
        print(f"\n{'=' * 60}")
        print("Action needed:")
        print(f"  helix config set HELIX_DEFAULT_MODEL {working_model}")
        print(f"{'=' * 60}")


asyncio.run(main())
