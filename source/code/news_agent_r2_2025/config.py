from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/news_agent_r2_2025_8w"
OLD = ROOT / "results/thematic_weekly_learning_2025_8w"
MODEL = "claude-sonnet-4-5-20250929"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE = Path(r"C:\Users\Boxuan\.vscode\extensions\anthropic.claude-code-2.1.289-win32-x64\resources\native-binary\claude.exe")
DEV_DATES = ["2025-08-01", "2025-08-08", "2025-08-15", "2025-08-22",
             "2025-08-29", "2025-09-05", "2025-09-12", "2025-09-19"]
TEST_DATES = ["2025-10-03", "2025-10-10", "2025-10-17", "2025-10-24",
              "2025-10-31", "2025-11-07", "2025-11-14", "2025-11-21"]
WORKERS = 24
# Operational safeguards, not financial budgets. All truncation is disclosed.
MAX_EVENTS, MAX_EVIDENCE_CHARS = 30, 24000
MAX_AGENT_ITERS = 12
CONTEXT_CHARS = 100000
# A fixed search extent prevents optional stopping at a favorable test result.
GEPA_FULL_EVALS = 12
SEED = 20261007
VERSION = "metadata-agent-panel-r2-v1"
SEED_POLICY = """EVIDENCE USE
Use own-company and macro/financial news available at the decision cutoff. Begin
with the supplied packet. Browse events to check coverage, search for a concrete
missing development, and read surrounding text to resolve qualifications.
Read multiple sources together when useful. Compare updates before treating
repeated reporting as multiple shocks. Company tags identify candidates, not
material exposure. Distinguish the target's facts from another firm's facts.
Keep a short evidence-based account of unresolved uncertainties. Stop when no
specific information gap remains; tool use is optional, not a quota.

RISK ANALYSIS
Estimate expected log(mean of the next five trading sessions' squared decimal
total returns), minus the provided log GARCH variance forecast b. This numerical
score is a log-variance revision, not sentiment, a percentile, or confidence.
Consider both uncertainty creation and resolution, event timing and materiality.
An already reported price move is not itself evidence of future volatility.
Macro events may affect the firm without naming it, but state the transmission
supported by available evidence. Separate imminent developments from long-term
themes. Use numerical state without simply repeating GARCH. Do not assume that
every realized forecast error was predictable from news. Cite only inspected
sources, distinguish missing evidence from evidence of no risk, and submit one
score with concise supporting reasoning."""
