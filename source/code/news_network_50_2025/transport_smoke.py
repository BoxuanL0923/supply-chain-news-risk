"""Single synthetic tool-transport test; no labels or financial forecasts."""
from .test_design import fixture
from .runtime import NativeClient
from .config import OUT, LIMITS
from news_agent_r2_2025.config import CLAUDE
from news_agent_batched_runtime_2025 import write


if __name__ == "__main__":
    out = OUT / "transport_smoke"
    path = out / "capability.json"
    write(path, fixture())
    client = NativeClient(out, LIMITS, CLAUDE)
    client.budget.start()
    result = client.native('This is a SYNTHETIC tool transport test, not real news. Call inspect_event on source_id a, page 0. '
                           'Then return JSON {"source_id":"a","quote":"the exact sentence about the backup facility"}. '
                           'Do not call market_state. No financial prediction.', path)
    print(result["answer"], flush=True)
    print([r["tool"] for r in result["trace"]["calls"]], flush=True)
