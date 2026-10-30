import os
import sys

import groq
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

    if not api_key:
        print("[FAILED] GROQ_API_KEY is missing. Add it to your .env file.")
        return 1

    client = groq.Groq(api_key=api_key, timeout=60)
    prompt = "Explain a Python variable to a beginner in 3 bullet points."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except groq.APITimeoutError:
        print("[FAILED] The request timed out. Try again.")
        return 1
    except groq.APIConnectionError:
        print("[FAILED] Could not reach Groq. Check your internet.")
        return 1
    except groq.AuthenticationError:
        print("[FAILED] The API key was rejected. Check GROQ_API_KEY.")
        return 1
    except groq.RateLimitError:
        print("[FAILED] Rate limit reached. Wait a bit and retry.")
        return 1
    except groq.APIStatusError as exc:
        print(f"[FAILED] Groq returned an error (HTTP {exc.status_code}).")
        return 1

    text = (response.choices[0].message.content or "").strip()
    if not text:
        print("[FAILED] Groq answered with an empty reply.")
        return 1

    print(f"[OK] Model: {model}\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())