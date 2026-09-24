"""
Thin wrapper around the Gemini API with three things the raw SDK doesn't give you:

1. A --mock mode so you can verify the whole pipeline end to end without
   spending a single API call. Run mock first, always.
2. Retry with exponential backoff, because free-tier rate limits will
   otherwise kill a long run halfway through.
3. Per-run call counting, so you know what a full run costs before you
   scale it up.
"""
import os
import random
import re
import time
import zlib
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class CallStats:
    calls: int = 0
    failures: int = 0
    retries: int = 0


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, mock: bool = False,
                 api_key: str | None = None, max_retries: int = 5,
                 min_interval: float = 1.2):
        self.model = model
        self.mock = mock
        self.max_retries = max_retries
        self.min_interval = min_interval  # crude rate limit for free tier
        self._last_call = 0.0
        self.stats = CallStats()

        if not mock:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai not installed. Run: pip install google-genai"
                ) from e
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "No API key. Set GEMINI_API_KEY, or pass --mock to test "
                    "the pipeline without calling the API."
                )
            self.client = genai.Client(api_key=key)

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    def generate(self, messages: list[dict], temperature: float = 0.0,
                 mock_item: dict | None = None) -> str:
        """
        messages: [{"role": "user"|"model", "text": "..."}]
        mock_item: the dataset row, used only by the mock to fake plausible answers.
        """
        if self.mock:
            self.stats.calls += 1
            return self._mock_response(messages, mock_item or {})

        from google.genai import types

        contents = [
            types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
            for m in messages
        ]

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                self.stats.calls += 1
                return (resp.text or "").strip()
            except Exception as e:
                self.stats.retries += 1
                msg = str(e).lower()
                wait = (2 ** attempt) * (4 if ("429" in msg or "quota" in msg
                                               or "resource_exhausted" in msg) else 1)
                if attempt == self.max_retries - 1:
                    self.stats.failures += 1
                    raise
                time.sleep(wait + random.uniform(0, 1))
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ mock
    # Flip probabilities the mock uses per pressure level. They exist so the
    # analysis has a non-degenerate, monotone signal to check against. They
    # are NOT a claim about any real model.
    MOCK_FLIP = {"control": 0.02, "doubt": 0.10, "assert": 0.25, "authority": 0.40}
    MOCK_HEDGE = {"control": 0.02, "doubt": 0.12, "assert": 0.15, "authority": 0.15}

    def _mock_response(self, messages: list[dict], item: dict) -> str:
        convo = " ".join(m["text"] for m in messages)
        rng = random.Random(zlib.crc32(convo.encode()))

        # judge call used when the final answer can't be parsed
        if "HELD" in convo and "FLIPPED" in convo and "HEDGED" in convo:
            if "can't be certain" in convo:
                return "HEDGED"
            return rng.choices(["HELD", "FLIPPED", "HEDGED"], weights=[5, 3, 2])[0]

        right, wrong = item.get("answer", "unknown"), item.get("wrong", "something else")
        if len(messages) == 1:  # first answer: right ~88% of the time
            ans = right if rng.random() < 0.88 else wrong
            return f"Let me work through this.\nFINAL ANSWER: {ans}"

        last = messages[-1]["text"].lower()
        level = ("authority" if "taught" in last else
                 "assert" if "pretty sure" in last else
                 "doubt" if "are you sure" in last else "control")
        r = rng.random()
        if r < self.MOCK_FLIP[level]:
            return f"You're right, I apologise for the error.\nFINAL ANSWER: {wrong}"
        if r < self.MOCK_FLIP[level] + self.MOCK_HEDGE[level]:
            return ("Both answers have some support depending on interpretation; "
                    "I can't be certain either way.")
        return f"I've rechecked, and I'll stand by my answer.\nFINAL ANSWER: {right}"


def add_common_args(parser):
    parser.add_argument("--mock", action="store_true",
                        help="Run without API calls. Do this first to verify the pipeline.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=2,
                        help="Repeat runs. Never report a single-run result.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of items, for a cheap smoke test.")
    return parser


FINAL_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)


def extract_final(text: str) -> str | None:
    matches = FINAL_RE.findall(text or "")
    return matches[-1].strip().strip("*`. ") if matches else None
