"""Standalone repository CLI; standard-library numerical graph calculation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/code"))
from news_network_50_2025.graph_skills import cli

if __name__ == "__main__":
    cli("triangles")
