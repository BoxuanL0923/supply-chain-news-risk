# Public copy: differences from the private frozen code

The private, hash-bound code is at `D:/OptResearch_data/external/supply-chain-news-risk/five_session/`. The
clearances and handoffs bind that copy. This repository copy differs from it in the places below and nowhere else.
A pre-commit review (2026-11-09) checked the folder for keys, licensed data, article text and record keys.

## Not published

- `td3/run_td3.py`, the TD3 v1 modules `td3/td3_*.py`, the handoffs `td3/TD3_HANDOFF.md` to `td3/TD3_HANDOFF_v5.md`,
  and the folders `td3/v2/` to `td3/v5/`.
  - These superseded TD3 versions contain firm-name tables, real-firm examples and fixtures, and sample-derived
    strings. `td3/TD3_HANDOFF_v6.md` says they must not be pushed.
  - No final module imports them. TD4 uses `td3/v6/`.
- `__pycache__` folders and `.pyc` files.

## Changed in this copy

| File | Change | Effect |
| --- | --- | --- |
| `td4/v2/td4v2_flags.py`, `td4/v2/td4v2_redact.py` | Two hard-coded CRSP PERMNOs, the firms whose names begin with the index words 'NASDAQ' and 'DOW', are replaced by the empty constant `INDEX_NAME_PERMNOS` | In this copy the detector class MA2:G3 never fires. The private copy keeps the two identifiers. |
| `td4/v2/td4v2_mask.py` (one comment) | A real firm name used as an example in a comment is replaced by "<Name> Airlines" | Behaviour unchanged, but the file's SHA256 changes |
| `td3/v6/td3_build.py` (one docstring), `td3/v6/td3_scan.py` (one comment) | A real firm's short name and a CRSP-form name fragment, used as examples, are replaced by neutral text | Behaviour unchanged, but the files' SHA256 change |
| `td4/TD4_HANDOFF_v2.md` (line 347) | The two lists of label-context IDs, which can be matched back to firm-weeks, are replaced by "(IDs in the private copy)". The lists are the 38 contexts changed by the redaction and the 7 withheld contexts. | Documentation only |
| `td4/TD4_HANDOFF_v2.md` | Two short strings quoted from article text, a price and a model number, are replaced by their kind | Documentation only |
| `td4/TD4_HANDOFF_v2.md` | A note stating the 2021-2022 selection outcome is reduced to a pointer to the private report | Documentation only |

## Effect on the gates

- The TD3 and TD4 gates check the absolute path and SHA256 of every bound `.py` file.
- This copy lives at a different path, and several of its files have different hashes, so it can never pass those
  gates. No provider call and no gated selection or evaluation outcome read can run from it.
- Only the private copy reproduces the runs.

## Added

- `README.md`.
- This file.

## Kept as they are (reviewed, not tables)

- **Generic word lists.** The masking and scan modules hold stop lists and generic vocabulary. These include
  exchange, index, agency and data-vendor names, and common acronyms that are also tickers.
- **A few brand or name fragments.** These are used as collocation guards in regular expressions:
  - `td4/v2/td4v2_scan.py`;
  - `td4/v2/td4v2_mask.py`;
  - a detector class name used in `td4/v2/td4v2_redact_final.py`, its tests and the TD4 handoff.

## Run facts in the code

These are run facts, not results:
- `td4/v2/td4v2_spec.py` (`V2_2_FIX`) holds per-period redaction counts.
- `td4/td4_cost.py` holds the DeepSeek balance read before the TD4 v1 projection.
- `td4/td4_spec.py` and `td4/TD4_HANDOFF.md` hold the TD2 ridge penalties chosen on 2021-2022.
