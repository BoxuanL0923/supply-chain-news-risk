# Supply chain news risk data and code package

This package documents the inputs, variable definitions, and tool-accessible search universe for the completed 50-firm news-risk experiment. It also includes the later source-grounded toolkit as a separately identified implementation. It is intended for collaborator review, not as evidence that the later toolkit produced the earlier results.

## Start here

- [Q1 graph skills and local prompt convergence report](docs/Q1_GRAPH_SKILLS_AND_PROMPT_GD.md): explicit degree/triangle tools, analytic GD, local-Qwen SPSA, and an August-only validation smoke. Engineering evidence, separate from the completed historical experiment.

- Explicit Skill invocation: Claude Code uses `/supply-chain-news-risk`; Codex
  uses `$supply-chain-news-risk`. The provider-neutral bundle is
  `skills/supply-chain-news-risk/`; the thin `.claude/skills/` wrapper disables
  implicit Claude invocation. OpenAI's machine-readable policy also disables
  implicit invocation. Neither entry point runs production forecasts.
- [Data dictionary](DATA_DICTIONARY.md): raw fields, identifiers, formulas, clocks, missingness, and known limitations.
- [Tool search universe](TOOL_SEARCH_UNIVERSE.md): what the analyst can search, which Python scripts implement it, and how versions differ.
- `dictionaries/variables.csv` and `dictionaries/variables.json`: machine-readable response and covariate definitions.
- `dictionaries/raw_fields.csv` and `dictionaries/observed_schemas.json`: field-level source notes and observed local headers/schemas. These are project dictionaries, not redistributed vendor manuals.
- `dictionaries/tool_registry.json`: statically extracted tool signatures, implementations, and arm availability.
- `metadata/run_schedule.csv`, `metadata/firm_universe.csv`: forecast origins/splits and selected firms, without returns, market-cap values, or news text.
- `metadata/verification.json`: checks against saved input artifacts, with no new forecasting or outcome reads.
- `metadata/source_manifest.csv`: source hashes and comparison with the historical run's recorded hashes where available.
- `metadata/file_manifest.json`: hashes of all package files other than the manifest itself.
- `source/code/`: Python source snapshot, preserving the original relative paths and bytes.
- `examples/`: synthetic numerical examples and a standard-library-only test, not actual Reuters or FactSet records.

## Experiment identified by this package

The completed experiment is `results/news_network_50_2025`: 50 firms and 12 forecast origins, with 200 training, 100 validation, and 300 evaluation firm-origins. Its exact controls are `b`, `rv1`, `rv5`, `rv22`, `earnings_confirmed`, `calendar_unknown`, `post_earnings`, `pending_cash_deal`, `log_news_count`, `log_neighbor_count`, and `log_degree`. Each origin forecasts the next five exchange sessions; this is not a panel of daily forecasts.

The original retrieval server is `news_network_50_2025/toolkit.py`. The later `network50_grounded_toolkit.py` and `network50_evidence_toolkit.py` are included for engineering review. In particular, the evidence toolkit was checked on cached failure cases and at the native tool interface; that is not a completed rerun of the 50-firm forecasting experiment. The paused December 30-firm experiment has a revised control specification and must not be merged silently with the 50-firm result.

## Access and redistribution

No Reuters article bodies/headlines, FactSet relationship rows, CRSP returns, realized-risk labels, model responses, capabilities, credentials, or original vendor PDFs are included. The source inventory identifies where the licensed inputs reside locally; collaborators need their own authorized access or an institution-approved private transfer. The original FactSet user guide was inspected locally and is identified by filename/hash, not copied.

Keep this package private until source-code ownership and data-license permissions have been reviewed. A private repository does not itself grant redistribution rights. No blanket open-source license is assigned by this package.

The Python snapshot contains original workstation paths, including source-directory and executable settings. These are configuration details, not embedded data or credentials. Some legacy helpers use dynamic imports and fixed paths. This is an auditable source snapshot, not a portable one-command replication environment. Required licensed artifacts and cached indexes are deliberately absent. `metadata/import_inventory.json` lists static dependencies and explicit dynamic-import supplements. `metadata/environment.json` distinguishes recorded historical versions from the environment used to build this package.

## Safe local checks

Audit the current forecasting dictionary without reading licensed rows, outcome
files, scores, or model responses:

```powershell
python skills/supply-chain-news-risk/scripts/validate_dictionary.py --repo . --mode forecasting
```

Use `--mode causal` before making any shock-propagation or mechanism claim. A
nonzero exit is intentional when the structured causal contract is incomplete.

From the package directory, run:

```powershell
python examples/test_formulas.py
```

From the original research workspace, rebuild the metadata and source snapshot with:

```powershell
.\.venv_dspy\Scripts\python.exe code/build_supply_chain_collaborator_package.py
```

The builder reads source code, table schemas/headers, existing provenance, saved numerical inputs/controls, and calendar features. It does not import forecasting runners, read outcome/score files, scan full Reuters months, fit models, or contact an LLM. Production runner scripts in the snapshot can incur model usage if deliberately executed; they are supplied for inspection, not launched by packaging.

## Important interpretation details

`rv1`, `rv5`, and `rv22` are natural logs of trailing mean squared daily decimal total returns, not intraday realized variance. The response has the same daily-return basis. The earnings calendar is reconstructed from supported Reuters diary rows, not a complete issuer calendar. Unknown coverage is not evidence of no event. The pending-cash-deal regex was ineffective in the completed sample; the package retains its exact definition rather than substituting the later repair.

The graph is reconstructed as of June 6, 2025 and held constant by research assumption through year-end. It is not a certified historical vendor-delivery snapshot. Source/return revisions and model training-cutoff limitations also prevent claiming that all possible look-ahead channels are eliminated.
