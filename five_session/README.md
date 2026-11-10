# Five-session firm volatility: firm-only benchmark, graph/news risk head and LLM characteristic selectors

This folder contains the code and handoff documents for the one-outcome experiment in the collaborator TODO ("One-outcome
experiment: five-session firm volatility"). The outcome is y = log(mean of the next five daily squared total returns)
for firm i at forecast cutoff t.

## What is in this folder, and what is not

It holds code and documentation only: no data file, provider response, article text or key. The handoffs record run
facts: row, withholding and test counts, costs, memory use, and the 2021-2022 selection settings. They contain no
test-period loss or gain.

This is a public copy. It differs from the private frozen code in a few documented places, listed in
`PUBLIC_COPY_NOTES.md`. For that reason the SHA256 values in the handoffs match the private copy, not every file
here.

Inputs, frozen models, provider ledgers, clearances and results live in private roots outside the repository. The
code reads these paths directly:
- `D:/OptResearch_data/five_session/` (with `td2/`, `td3/`, `td4/` and `clearance/`);
- `C:/OptResearch/data/macro/` (CRSP daily compact returns, the label source);
- `C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/`:
  - `processed/`: CRSP, Compustat, JKP, earnings dates and Reuters articles;
  - `FactSetRevere/`: raw FactSet links;
- `C:/OptResearch/results/sc_risk/` and `C:/OptResearch/results/hist_panel_2015_2024/`;
- `D:/OptResearch_data/sc_risk/`;
- `D:/OptResearch_data/external/factset_baidu/FSB1_pre2025_v1/`;
- `D:/OptResearch_data/wrds_20261031/`;
- the NLTK `words` corpus;
- `C:/Users/Boxuan/.claude/uploads/<session-id>/`, which holds the collaborator TODO. It is read only by the
  inspection and probe scripts and holds no experiment data;
- `D:/OptResearch_data/five_session_review/`, which holds the review audits. The TD4 v2 redaction also runs the
  reviewers' detector scripts from this folder (`td4/v2/td4v2_flags.py`), so it cannot be re-run from this repository
  alone.

## Layout and how it maps to the TODO

| Folder | Step | TODO item | Final version |
| --- | --- | --- | --- |
| `./` and `v2/` | TD1: point-in-time firm-week manifest, availability clocks, firm-only MLP (64, 32); fit 2015-2020, tune 2021-2022, refit through 2022, freeze | sections 1-2 | `v2/`, handoff `TD1_HANDOFF_v2.md`. `FIRST_HANDOFF.md` is the source inventory. |
| `td2/` | TD2: risk menu (size, cash, cash flow, leverage, inventory), exposures Z = C × E with exact-hop H1/H2/H3 coverage, one shared ridge head, synthetic fixture, selector harness | section 3 | `td2/`, handoff `td2/TD2_HANDOFF.md` |
| `td3/` | TD3: masking and the small DeepSeek / Jev Choice / Jev Noul selector pilot | section 3 (selectors, pilot) | `td3/v6/`, handoff `td3/TD3_HANDOFF_v6.md`. As that handoff directs, the v1-v5 code and handoffs are not published. |
| `td4/` | TD4: full selector calls, selection on 2021-2022, once-only pooled 2023-2025 evaluation, label diagnostics | section 4 | `td4/v2/` at code version v2.3. Handoff `td4/TD4_HANDOFF_v2.md`, sections 11 and 16-18. `td4/TD4_HANDOFF.md` and the `td4/*.py` files are v1, and v2 still imports parts of them. |

What the handoffs contain:
- Each step handoff (TD1-TD4) gives the exact commands and the gates: clearance files written by independent
  reviewers.
- TD2-TD4 also give the SHA256 of the bound files, plus their limitations or open risks. TD1 names its hash receipts
  instead.
- `FIRST_HANDOFF.md` is the source inventory.

## Conventions in the code

- **Availability.**
  - Every feature, news version, graph edge and model input is dated strictly before the cutoff.
  - Label windows end by 2024-12-31. TD4 is the exception: its labels may end as late as 2025-12-31. Nothing dated
    2026 is used.
  - FactSet relationship start dates are retrospective validity proxies from an October 2025 snapshot, not
    publication dates.
- **Gates.**
  - These refuse unless a reviewer-written clearance file matches the frozen code and inputs by SHA256:
    - provider calls (TD3, TD4);
    - every selection or evaluation outcome read: TD1 `evaluate`, the TD2 2023-2024 evaluation, TD4 `select` and
      `evaluate`.
  - The 2015-2022 development labels that fit the TD1 MLP and the TD2 head are read without a clearance.
  - The TD4 evaluation writes a once-only latch before it reads any test label.
- **Call paths after v2.3.** Under code version v2.3, `run_td4v2.py call` and `reask` refuse, and so does
  `td4v2_gate.admit`. The superseded TD3 v6 and TD4 v1 runners keep their call paths behind their own clearances.
- **Providers.**
  - DeepSeek (`api.deepseek.com`, off-peak hours only, with a balance floor).
  - Jev (`api.typesafe.ai/v1/systemone`, model pinned to jev-1.14.0).
  - Keys are read only from the environment variables `DEEPSEEK_API_KEY` and `TYPESAFE_API_KEY`. They are never
    written to any file.
- **LLM outputs.** They only gate which characteristic term is active. They are never used as regressors.
- **Runtime.**
  - Windows. The code uses job-object memory caps through `ctypes`, plus `msvcrt` locks and `winreg`.
  - Python 3.13 with numpy, pandas, pyarrow, scipy, scikit-learn, rapidfuzz, pypdf and arch.
  - The NLTK `words` corpus file.
  - For TD1, Git for Windows `date.exe`.
- **Running.**
  - Run with the full Python 3.13 interpreter path and `-B`, as each handoff shows.
  - TD2-TD4 run from the step's folder. TD1 runs by script path, and its handoffs give the old clone path.

## Not included

- The firm-week manifests, CRSP/Compustat/FactSet extracts and Reuters articles.
- Masked or redacted provider inputs, and provider responses and ledgers.
- Frozen models, selection and evaluation results, review audits and clearance files.

All of these stay in the private roots named above.
