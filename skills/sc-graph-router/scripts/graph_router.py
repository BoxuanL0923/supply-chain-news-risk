"""Repository-local standalone entry point; no model or active runner imports."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/code"))
from news_network_50_2025.graph_router import main

if __name__ == "__main__":
    raise SystemExit(main())
