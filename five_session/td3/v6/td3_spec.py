"""TD3 provider-selector pilot, correction round v6 (TD3F): frozen constants, rules and provider questions.

Public file: no private observations, article text, keys, outputs, sampled firm names or tickers are stored here.
Nothing runs on import. The risk menu text is read from the hash-bound TD2 risk_menu.csv at run time; the
firm-specific masking tables are read from the hash-bound private table file MASK_TABLES.

v6 changes against v5 (review TD3C5_AUDIT.md, D1 only): a unit name made of the firm's ticker or initialism and a
corporate-unit word (td3_mask.UNIT_WORDS) becomes the firm's token, with residual, verify and scan checks; one common
name added to the private table; v6 clearance and file names; the manual-read rule requires the audit-listed input and
every changed input to be read again in final content.

v5 changes against v4 (review TD3C4_AUDIT.md, D1, D2, R1 and N-a to N-f): glued tokens take the whole alphanumeric
word and a house number before a street-word token is absorbed (D1); firm codes PREFIX[-]digits[letters] with a
prefix from the firm's tickers, initialisms, leading-letter abbreviations or the private table's code_prefixes
section, and near names from the CRSP and FactSet name histories, become the firm's token (D2); the limitation text
rewritten with invented examples that a test checks (R1); ordinal dates (N-a); press-release IDs after "Source text for
Eikon:" (N-b); joint-partner and prefix passes inside the product loop (N-c); the D: free-space guard lowered to
25 GiB (N-e); invented test strings only (N-f); every input that no reviewer read end to end and every input changed
from build_v5 read end to end by reader subagents that did not write the masking code (manual-read rule v5).

v4 changes against v3 (review TD3C3_AUDIT.md, E1-E3 and N1-N6): firm-owned product, unit and joint-product names
absorbed into the firm's token, with their acronyms, from a frozen source-noted private table, a verify check and an
adjacency sweep with the hidden word of every role token next to a capitalised word (E1); month typos and day-range
remnants next to [DATE]/[YEAR] (E2); the context-identification limitation rewritten to describe the masker (E3); no
person or place guard (N1, N2); fixed-phrase guards restricted to index, price and idiom phrases (N3); FactSet-only
single-word variants must be distinctive (N4); Jev Choice probability-sum tolerance 0.03 = 6 options x 0.005 (N5);
invented names only in the public code and tests, firm-specific tables private (N6).

v2 changes against v1 (review TD3C_AUDIT.md, corrections C1-C5 and R1-R5): broader alias generation and an
independent second residual scanner (C1), numeric dates, two-digit and fiscal years, Reuters story boilerplate and
bare-domain URLs stripped (C2), Jev Choice probability-sum tolerance 0.02 with the raw sum recorded and a halt on
repeated structural invalids (C3), Jev priced inside the same USD cap (C4), returned-model and reasoning checks (C5),
final HTTP 429/5xx units in the failure halt (R1), no redirects (R2), a run lock (R3), edge ages (R4).

v3 changes against v2 (review TD3BC_AUDIT.md, D1-D4 and m1-m5): issuer RICs and tickers, embedded names,
edit-distance-1 misspellings and a context guard so that no mask gives the hidden word away (D1); years glued to a
time and generic bare-domain URLs (D2); DeepSeek model deepseek-flash with thinking disabled, per the official API
docs read 2026-11-05 (D3); separate consecutive-failure counters per provider (D4); any reasoning field or a
non-numeric reasoning count is a model violation (m1); public test strings are invented (m2); context
identification is a recorded limitation (m3); units refused by the model rules are never re-asked under the same
clearance (m4).
"""
import csv
import hashlib
import json
from pathlib import Path

VERSION = 'TD3-selector-pilot-v6'
INPUT_VERSION = 'TD3-input-v7'
QUESTION_VERSION = 'TD3-questions-v1'     # question texts unchanged from v1 (the bundle adds the model and thinking)

CODE = Path(__file__).resolve().parent            # .../five_session/td3/v6
TD2_CODE = CODE.parent.parent / 'td2'             # .../five_session/td2 (imported read-only)
OUT = Path('D:/OptResearch_data/five_session/td3')
RUNS = OUT / 'runs'                               # shared with v1, so every real run of any version counts
FIXTURES = OUT / 'fixtures'
RUN_LOCK = RUNS / 'TD3_RUN.lock'
TD2 = Path('D:/OptResearch_data/five_session/td2')
CLEARANCE = Path('D:/OptResearch_data/five_session/clearance/TD3_CLEARANCE_v6.json')
CLEARANCE_SCHEMA = 'TD3_CLEARANCE_v6'
FREEZE_SCHEMA = 'TD3_FREEZE_v6'
FREEZE_NAME = 'FREEZE_TD3_v6.json'
CLEARANCE_REQUEST_NAME = 'CLEARANCE_REQUEST_TD3_v6.json'
QUALIFICATION_PREFIX = 'QUALIFICATION_TD3v6_'
SUPERSEDED_FREEZES = ('FREEZE_TD3_v1.json', 'FREEZE_TD3_v2.json', 'FREEZE_TD3_v3.json',
                      'FREEZE_TD3_v4.json', 'FREEZE_TD3_v5.json')       # never to be cleared
# v4 (TD3C3 E1, N6), v5 (TD3C4 D2), v6 (TD3C5 D1): the frozen, source-noted firm-specific masking tables (common
# names, firm-owned product, unit and joint-product names, fixed phrases keyed by a firm word, firm code prefixes).
# Private: it names sampled firms. Hash-bound in BUILD.json and the freeze; the build refuses without it.
MASK_TABLES = Path('D:/OptResearch_data/five_session/td3/tables_v6/MASK_TABLES_v4.json')
PROCESSED = Path('C:/Users/Boxuan/Documents/GitHub/ReutersNewstTraining/processed')
ARTICLE = PROCESSED / 'articles_classified.parquet'
CRSP_NAME_FILES = (PROCESSED / 'crsp_stocknames.csv', PROCESSED / 'crsp_names.csv')
# FactSet company names (FSB1 provisional test data, researcher-approved for tests on 2026-11-02), rows with start_
# on or before 2022-12-31, linked to CRSP PERMNOs by 8-digit CUSIP: a second masking name source and a name source of
# the independent scanner. Masking only removes text; no FactSet text enters an input.
FACTSET_COMPANY = Path('D:/OptResearch_data/external/factset_baidu/FSB1_pre2025_v1/company_pre2025_v1.parquet')
# Frozen English word list (NLTK 'words' corpus = Unix web2, 235,886 words): a ticker is also masked in title case
# ("Mufg") only when its lower-case form is not an English word.
ENGLISH_WORDS = Path('C:/Users/Boxuan/AppData/Roaming/nltk_data/corpora/words/en')
DATE_EXE = 'C:/Program Files/Git/usr/bin/date.exe'
# The TD3C reviewer's 79 confirmed v2 leak strings and the TD3BC reviewer's confirmed findings (private files);
# verify checks that none occurs in the same-rank input (TD3BC recoverable masks: the hidden word is no longer masked
# or the phrase no longer exists).
REVIEW_LEAKS = Path('D:/OptResearch_data/five_session_review/td3c/confirmed_leaks.json')
REVIEW_FINDINGS_BC = Path('D:/OptResearch_data/five_session_review/td3bc/confirmed_findings_td3bc.json')
# v4: the TD3C3 reviewer's E1 evidence (masked text of each unit-compound case) and E2 date-typo list (private files);
# verify checks that no E1 masked phrase and no E2 typo remnant occurs in the same-rank input.
REVIEW_FINDINGS_C3 = Path('D:/OptResearch_data/five_session_review/td3c3/evidence_unit_compounds.json')
REVIEW_DATE_TYPOS_C3 = Path('D:/OptResearch_data/five_session_review/td3c3/s4_date_typos.txt')
# v5: the TD3C4 D1, D2 and N-b items transcribed from TD3C4_AUDIT.md into a private evidence file (rank, string,
# class); verify checks that no listed string occurs in the same-rank input.
REVIEW_FINDINGS_C4 = Path('D:/OptResearch_data/five_session/td3/tables_v5/TD3C4_FINDINGS.json')
# v6: the TD3C5 D1 item (rank, string, class) transcribed from TD3C5_AUDIT.md into a private evidence file.
REVIEW_FINDINGS_C5 = Path('D:/OptResearch_data/five_session/td3/tables_v6/TD3C5_FINDINGS.json')

TD2_FREEZE = TD2 / 'FREEZE_v2.json'
TD2_FREEZE_SHA256 = '2d383ffb4894f4054728b30f15c9ce2197a5d03537aac95033e011b66aff03c5'
TD2_MENU = TD2 / 'risk_menu.csv'
TD2_COST = TD2 / 'TOKEN_COST_ESTIMATE_v2.json'
TD2_CODE_USED = ('fast_builder.py', 'engine.py', 'spec.py')  # imported read-only at build time

NAMES = ['size', 'cash', 'cash_flow', 'leverage', 'inventory']
HOPS = (1, 2, 3)

# Sample (selection period only). Unchanged from v1: the same 200 firm-weeks and 20 re-asks.
SEED = 73001
SAMPLE_N = 200
REASK_N = 20
YEARS = (2021, 2022)
FIRST_DAY, LAST_DAY = '2021-01-01', '2022-12-31'
LAST_MONTH = 202212
NEWS_YEARS = (2020, 2021, 2022)
ARTICLE_LO = '2020-12-20'      # covers the 7-day lookback of the first 2021 origins
LOOKBACK_DAYS = 7
HEADLINE_CHARS, BODY_CHARS = 200, 1000
CANDIDATE_BATCH = 400

# Providers.
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
DEEPSEEK_MODEL = 'deepseek-flash'
# D3: official create-chat-completion reference (read 2026-11-05): model is deepseek-flash or deepseek-v4-pro; thinking
# is enabled by default and is turned off with "thinking": {"type": "disabled"}; temperature has no effect in thinking
# mode; response_format {"type": "json_object"} enables JSON output.
DEEPSEEK_THINKING = {'type': 'disabled'}
DEEPSEEK_MAX_TOKENS = 200
DEEPSEEK_TEMPERATURE = 0
JEV_URL = 'https://api.typesafe.ai/v1/systemone'
JEV_MODEL = 'jev-1.14.0'
ALLOWED_HOSTS = ('api.deepseek.com', 'api.typesafe.ai')
KEY_VARS = {'deepseek': 'DEEPSEEK_API_KEY', 'jev': 'TYPESAFE_API_KEY'}
SELECTORS = ('deepseek_order', 'jev_choice', 'jev_noul')
PROVIDER = {'deepseek_order': 'deepseek', 'jev_choice': 'jev', 'jev_noul': 'jev'}
CHOICE_KEY = 'risk_choice'
NOUL_PREFIX = 'applies_'
NONE_OPTION = 'none'

# Budget. One USD cap covers DeepSeek and Jev; Jev is also capped by request count.
MAX_USD = 5.00
MAX_JEV_REQUESTS = 600
PRICE_SOURCE = ('DeepSeek: official pricing page api-docs.deepseek.com/quick_start/pricing, read by the TD3C2 builder '
                'on 2026-11-05: deepseek-flash USD per million tokens 0.003 input cache hit, 0.15 input cache miss, '
                '0.60 output off-peak, twice that at peak; one price for thinking and non-thinking modes; peak hours '
                '01:00-04:00 and 06:00-10:00 UTC Monday-Friday, excluding Chinese public holidays (the code treats '
                'every weekday as possibly peak, which is conservative); the response example reports model '
                '"deepseek-flash". Jev: official Jev models page as read by the TD3C reviewer on 2026-11-05: '
                'jev-1.14.0 USD 42 per billion input tokens (USD 0.042 per million); output tokens free.')
PRICES = {'off_peak': {'input_miss': 0.15, 'input_hit': 0.003, 'output': 0.60},
          'peak': {'input_miss': 0.30, 'input_hit': 0.006, 'output': 1.20}}
# C5: frozen, priced allowlist of returned DeepSeek model names. Each entry's price table must equal PRICES; any
# other returned model (or a missing model field) halts the run after its first HTTP-200 response.
DEEPSEEK_RETURNED_MODEL_PRICES = {'deepseek-flash': PRICES}
JEV_PRICE_INPUT_PER_M = 0.042
JEV_PRICE_OUTPUT_PER_M = 0.0
PEAK_WINDOWS_UTC = ((1, 4), (6, 10))   # [start hour, end hour), Monday-Friday
UPPER_TOKENS_PER_BYTE = 0.5            # pre-call bound on request-body tokens (English ~0.24)
UPPER_FRAMING_TOKENS = 64
ESTIMATE_TOKENS_PER_BYTE = 0.2386      # TD2 ledger calibration; dry-run estimates only
ESTIMATE_OUTPUT_TOKENS = 30            # dry-run estimate for a JSON ordering
HTTP_TIMEOUT_S = 120
MAX_RESPONSE_BYTES = 1_000_000
TRANSPORT_RETRIES = 2                  # extra attempts after a transport error or HTTP 429/5xx
BACKOFF_S = (5, 20)
RETRY_STATUS = (429, 500, 502, 503, 504)
HALT_STATUS = (401, 402, 403)
CLIENT_ERROR_LIMIT = 3                 # halt after this many other 4xx responses in one run
FAILURE_UNIT_LIMIT = 3                 # D4: halt after 3 consecutive units OF ONE PROVIDER ending in a transport error
                                       # or HTTP 429/5xx (one counter per provider)
# C3/N5: Jev docs say the probabilities sum "to approximately 1" (API reference) and "is 1" (Choice page), with
# two-decimal examples. Rounding each of the n = 6 options to two decimals moves the sum by less than n x 0.005 = 0.03,
# so the tolerance is 0.03 (inclusive): rounding to the documented precision never makes a valid answer invalid. The
# raw sum is always stored.
CHOICE_OPTIONS = 6                     # five menu items + none
PROBABILITY_DECIMALS = 2
PROBABILITY_SUM_TOLERANCE = 0.03
assert PROBABILITY_SUM_TOLERANCE >= CHOICE_OPTIONS * 0.5 * 10 ** -PROBABILITY_DECIMALS - 1e-12
STRUCTURAL_PROBE_N = 3                 # halt when 3 consecutive HTTP-200 responses of a selector share a defect
STRUCTURAL_ISSUES = ('probability_keys', 'missing_probabilities', 'probability_sum', 'probability_values',
                     'missing_answer', 'missing_answers', 'invalid_choice', 'invalid_scores', 'invalid_json',
                     'missing_ordering')

# Resources.
JOB_MEMORY_CAP = 2_000_000_000
# v5 (TD3C4 N-e): the project's standing D: floor (25 GiB), so that an unrelated long job does not stop the pilot
MIN_FREE_BYTES = 25 * 1024 ** 3

ELIGIBILITY_RULE = (
    'TD2 features_v2 primary variant; origin session and cutoff both within 2021-01-01..2022-12-31; has_news; mapped '
    'focal; at least one non-missing menu characteristic (the TD2 DeepSeek call unit); and at least one partner '
    'article that passes TD3 re-verification. Years 2023-2025 are never read from features, evidence or news.')
SAMPLING_RULE = (
    'Eligible firm-weeks are ordered by SHA256("TD3-pilot-v1|73001|<firm_id>|<cutoff ISO>") ascending and examined '
    'in batches of 400; the first 200 that pass article re-verification form the sample (sample_rank = order). The '
    're-ask set is the 20 sampled firm-weeks with the smallest SHA256("TD3-reask-v1|73001|<firm_id>|<cutoff ISO>"). '
    'Unchanged from v1.')
ARTICLE_RULE = (
    'Eligible article for a shown partner: an article ID in the TD2 primary evidence row of that partner whose TD2 '
    'news_scores record has date in [origin-7 days, origin) (origin day excluded), available_at (next New York '
    'midnight) strictly before the cutoff, and the partner among its tagged firms with mapping first month before the '
    'origin month; its archive text must be the record TD2 scored (first physical row with a mapped RIC on that day; '
    'tagged firms and all five lexicon scores must match). Unchanged from v1.')
AGGREGATION_RULE = (
    'Shown partners are the TD2 primary selected partners (up to five news-bearing partners per exact hop H1, H2, H3, '
    'SHA256(cutoff, partner) priority; the same partners that define E and Z). For each shown partner the single most '
    'recent eligible article is used (latest day; ties by smallest alt_id). An article chosen for several partners '
    'appears once and lists all of them. Articles are ordered by hop, then partner priority. Text = headline first 200 '
    'characters and body_short first 1000 characters (the TD2 scoring window), a trailing partial word trimmed, then '
    'masked. A shown partner whose articles all fail re-verification stays in the graph state, marked as having no '
    'eligible article. Unchanged from v1.')
MASKING_RULE = (
    'Firms masked: the focal firm, every shown partner and every intermediate path firm, each with its role token. '
    'Name sources (all rows dated on or before 2022-12-31; masking only removes text, so rows dated after the origin '
    'add no look-ahead information): every CRSP comnam and ticker row (crsp_stocknames.csv and crsp_names.csv) of the '
    'firm and of its issuer siblings (other PERMNOs sharing a CRSP issuer CUSIP6: share classes, preferred and debt '
    'securities, predecessors); every name and US-listed ticker of the FactSet FSB1 company ids linked to the firm\'s '
    'issuer CUSIP6; RICs of the firm and its siblings from the TD2 development RIC map; the common-name section of the '
    'frozen private table file (MASK_TABLES). CRSP names are normalized before variant generation: runs of single '
    'letters collapse to an acronym (Q R T SYSTEMS -> QRT SYSTEMS), single-letter ampersand names compact (Q T & Q -> '
    'QT&Q), trailing province/state codes drop, frozen abbreviations expand (td3_mask.ABBREV, e.g. ENT -> '
    'ENTERTAINMENT) and inverted person names gain the natural order (ZOLDEN STEVEN -> STEVEN ZOLDEN). Variants: full '
    'name, trailing legal/state suffixes stripped stage by stage, first word (>=4 letters, not generic, not a place '
    'word; v4: from a FactSet-only name only when it is distinctive, i.e. not an English word, generic word, place '
    'word or given name), first two words (>=7 characters, not a place phrase), collapsed acronyms (>=3 letters, not '
    'on td3_mask.ACRONYM_STOP), word initials, initials of two-word X & Y names and the natural order of inverted '
    'person names. Match forms: words joined by spaces, hyphens, periods, nothing, or an optional connector (of, the, '
    'and, &); acronym letters with optional periods and spaces; abbreviation or expansion; possessives inside words '
    '(Glorpino\'s); a trailing possessive. Multi-word variants of >=6 characters match case-insensitively; other '
    'variants only when not all lower case; table aliases of <=3 characters only in exact case. Owner rule: in an '
    'article chosen for a firm, its first name word is also masked if it is a distinctive generic word '
    '(td3_mask.OWNER_GENERIC). Tickers: CRSP tickers and RIC roots, masked in RIC form, after an exchange prefix, in '
    'parentheses, as a cashtag, and standalone when >=2 characters, upper case and not on td3_mask.ACRONYM_STOP; full '
    'RICs are masked; sibling and FactSet-only tickers are masked standalone only when >=3 letters and not English. '
    'Issuer RICs (v3): in an article chosen for a firm, a RIC of the archive rics field whose root starts with a '
    'masked firm\'s ticker (a preferred-share or class suffix) takes that firm\'s token, every <RIC> of an exchange '
    'order-imbalance or trading-halt headline takes the token of the firm the article was chosen for, and a RIC written '
    'right after a role token takes that token. Embedded names (v3): a ticker or acronym of >=3 letters (not English) '
    'or a distinctive name word of >=4 letters inside a longer token (camel-case compounds such as ZetaQRX, an '
    'upper-case token starting with the acronym plus at most two letters or ending with it after an English word, a '
    'name word plus at most two letters) masks the whole token. Misspellings (v3): a Title or UPPER case token at edit '
    'distance 1 from a distinctive name word of >=6 letters (same first letter; for a 6-letter word only an inserted '
    'or deleted letter) masks the whole token. Firm-owned names (v4, TD3C3 E1): the firm-products section of the '
    'private table lists, per CRSP comnam prefix and with a source note, the product, unit, subsidiary, asset and '
    'program phrases that follow the firm word in a name owned by the firm, the phrases that precede it, and '
    'joint-product partners; a role token immediately followed by one of its own firm\'s phrases (any case unless the '
    'entry is marked capitalized, optional registered or trademark signs, optional possessive, a space or hyphen '
    'between; for a phrase marked as a model line also a following model designation such as "14 Pro Max") absorbs '
    'the phrase ("[X] Zorbcloud Services" -> "[X]"; v5, TD3C4 N-d: the name rule takes a possessive with the name, '
    'so raw "Zorbex\'s Zorbcloud Services unit" becomes "[X] unit"); v5 (TD3C4 D1): when the absorbed phrase ends in a '
    'street word (td3_mask.STREET_WORDS), a 1-6 digit house number right before the token is absorbed as well ("455 '
    'Zorbex Way" -> "[X]"); a preceding phrase of the firm joins the token the same way ("Fulfilled by [X]" -> "[X]"); '
    'a joint-product '
    'partner joined to the token by and, &, a hyphen, slash or plus, "and (its) partner" or a space is replaced with the '
    'token as a whole ("the Quillco and [X] vaccine" -> "the [X] vaccine", "[X]-Quillco" -> "[X]"); and in an article '
    'where a phrase was absorbed, its acronyms (the table\'s acronym and the firm initial plus the initials of the '
    'phrase words, >= 3 letters) are masked as the token; v5 (TD3C4 N-c): the joint-partner and preceding-phrase '
    'passes run inside the product loop, so a partner that becomes adjacent only after a product is absorbed is '
    'replaced too. Firm codes (v5, TD3C4 D2), applied before the name rules: a product or drug code written '
    'PREFIX, an optional hyphen or dash, 1-6 digits, up to three letters (and optional further -<digits> parts), not '
    'inside a longer word, whose PREFIX is in upper or Title case and is one of the firm\'s code prefixes '
    '(td3_mask.code_prefix_terms: every ticker, sibling or FactSet ticker and RIC root of letters only; the initials of '
    'the first k >= 2 significant words of a CRSP or FactSet name, camel-case parts of a mixed-case FactSet name '
    'counting as words and a collapsed acronym contributing all its letters; the first 2..n letters of the first '
    'significant word of a CRSP name, or of a FactSet name when that word is distinctive, when it has >= 3 letters and '
    'is not a generic or place word; the prefixes of the private table\'s code_prefixes section; never a word on '
    'td3_mask.CODE_PREFIX_STOP) is replaced as a whole by the firm\'s token ("ZRBX-1181", "Zrbx1181", "ZOR-09" -> '
    '"[X]"), unless the code is on the frozen general-vocabulary stop list td3_mask.NON_FIRM_CODES (CO2, N95, '
    'COVID-19, ...); a deal code name "Project <prefix>" (with an optional Parent, Merger Sub, Holdings, Holdco, '
    'Bidco, Topco, Midco, Buyer, Purchaser, Intermediate or Acquisition tail) whose prefix, in upper or Title case, is '
    'one of the firm\'s code prefixes is masked as the firm\'s token as well ("Project ZB Parent" -> "[X]"). Unit names '
    '(v6, TD3C5 D1), applied after the name rules: a firm ticker or initialism (a code prefix whose rule is ticker or '
    'initialism, >= 2 letters, in upper or Title case, not on td3_mask.CODE_PREFIX_STOP or td3_mask.ACRONYM_STOP) '
    'followed by a word of the frozen corporate-unit list td3_mask.UNIT_WORDS (Ventures, Capital, Holdings, Partners, '
    'Group, Labs, Energy, Solutions, ...) not in lower case is masked as a whole as the firm\'s token ("ZB Ventures" -> '
    '"[X]"); because it runs after the name rules it acts where the prefix is still readable, and a ticker already '
    'masked by the ticker rules keeps the unit word after its token ("ZRBX Ventures" -> "[X] Ventures"). Near names '
    '(v5, TD3C4 D2): every two-word name in the CRSP comnam history (both files) or the '
    'FactSet company names (any company id) dated on or before 2022-12-31 whose first word is the first significant '
    'name word of the firm\'s CRSP name (>= 3 letters, not generic or place) minus its last one letter (>= 2 left) or '
    'two letters (>= 3 left), or that word plus one or two letters, the first word not being an English word, generic '
    'or place word or common acronym, is masked as the firm\'s token when not all lower case ("ZO Logistics" for a firm '
    'whose name starts with ZOR). Glued tokens (v5, TD3C4 D1): after every other pass, a token glued to letters or '
    'digits on either side (no separator) takes the whole alphanumeric word, digits included. Embedded names (v5, '
    'TD3C4 R1): a distinctive name word of >= 5 letters starting a longer token and followed, in the same case style, '
    'by up to four letters or by more letters that are not an English word ("Zorbexpro", "ZORBEXPRO") also masks the '
    'whole token, and a ticker or acronym starting an upper-case token followed by digits only ("ZRBX1181"). Context '
    'guard: a single-word alias is left as written inside '
    'a fixed phrase of the private table that names an index, a price or rate, an idiom, a material or a different '
    'entity (v4, TD3C3 N3: restricted to those phrases themselves; for example an index word only in an index header, '
    'index name or index movement, a price word only in a price or target phrase), in a hyphenated verb compound '
    '(dine-in), and an English-word ticker of >=3 letters is left in upper-case text; a single-word alias right before '
    '", <State>" or "<State>-based" is a headquarters town and is deleted. v4 (TD3C3 N1, N2): there is no person or '
    'place guard: a firm word is masked also after a given name (a person\'s surname: "Founder Michael [X]") and after '
    'a place word ("Fire at Ohio [X] refinery"). Place words (td3_mask.PLACE_WORDS) are never single-word variants. '
    'Precedence: focal, then partners in hop/priority order, then intermediate firms; longest alias first. The news '
    'agency name becomes [NEWS_AGENCY]. URLs (scheme, www., and any label.tld/path with a 2-6 letter lower-case TLD) '
    'become [URL], e-mail addresses [EMAIL]; bylines ("By <reporters> <DATELINE>, <date> (<agency>)"), datelines, '
    'nested contact blocks, RIC boilerplate, STORY fields and bracketed story IDs (v5, TD3C4 N-b: also bracketed IDs '
    'that start with P, such as the press-release IDs after "Source text for Eikon:"; and a run of three or more '
    'unbracketed alert numbers of three letters, three digits and two letters or digits) are removed. Calendar dates become '
    '[DATE] (v5, TD3C4 N-a: also an ordinal day before or after a month, "the 30th of September", "30th September", '
    '"September the 30th"); four-digit years (also glued to a clock time), two-digit fiscal years and apostrophe years '
    'become [YEAR] or [DATE] (v5: also a year cut to three digits by the source right after a masked date, at the end '
    'of a text or before punctuation, "[DATE] 202" -> "[DATE] [YEAR]"); v4 (TD3C3 E2): a source typo or unlisted '
    'spelling of a month before "<day>, [YEAR]" becomes [DATE] '
    '(pattern \\b[A-Za-z]{1,9}\\.?\\s+\\d{1,2}(?:st|nd|rd|th)?,\\s*\\[YEAR\\]; [YEAR] kept; words on the frozen '
    'td3_mask.NON_MONTH_STOP list excepted), and a day-range remnant [DATE] followed by -, en dash, to, and or through '
    'and a one- or two-digit day (not a decimal number) becomes [DATE]. Acceptance: the strict residual scan (every '
    'masking rule re-applied, including the guard, embedded, misspelling, firm-owned-name, firm-code, near-name, '
    'unit-name (v6) and glued-token checks) is empty, no role token is followed by a firm-products phrase of its own firm, every article '
    're-masked from its stored raw text reproduces the input, no token is followed by a hyphen and a digit and no house '
    'number precedes a token whose hidden text ends in a street word, the independent fuzzy scanner (td3_scan, v5 with '
    'firm-code, near-prefix, glued and house-number checks, v6 with an initialism-unit check) reports zero confirmed focal or shown-partner names in an '
    'adjudicated frozen file, the adjacency review of every role token finds no mask that gives the hidden word away, '
    'the adjacency sweep (every role token immediately followed or preceded by a Title-case, upper-case or number word, '
    'or by and/& plus such a word, or glued to a letter or digit, with the hidden raw word from the raw-versus-masked '
    'alignment) is adjudicated row by row with zero rows that give the hidden word back, and readers who did not write '
    'the masking code report no readable focal or shown-partner identifier and no give-back mask (MANUAL_READ_RULE).')
ADJACENCY_SWEEP_RULE = (
    'v4 (TD3C3 E1 item 4): for every article field of every input, the build aligns the raw (truncated) text with the '
    'masked text word by word (difflib on word and punctuation tokens) and lists every role token that is immediately '
    'followed or immediately preceded by a word starting with an upper-case letter or a digit (Title case or upper '
    'case; a number neighbour such as a house number counts) or a camel-case word with an upper-case letter inside, or '
    'by and/& plus such a word, and (v5, TD3C4 D1) every token glued to a letter or digit on either side, with up to '
    'three neighbour words on each side, the hidden raw text of the token from the alignment, the glued sides, the '
    'number-neighbour sides and the firm class (ADJACENCY_SWEEP.jsonl). Every row is adjudicated in '
    'ADJACENCY_SWEEP_REVIEW.json as safe (with a reason) or as giving the hidden word back; verify requires the review '
    'to cover every row of the current sweep file and to record zero rows that give the hidden word back, checks that '
    'the token positions of the sweep equal those found again in the masked text and (v5) that the rows with their '
    'hidden words equal those recomputed from the raw text stored in the metadata, and fails on any glued row and on '
    'any house number before a token whose hidden text ends in a street word.')
MANUAL_READ_MIN = 1
MANUAL_READ_RULE = (
    'v6 (TD3C5): every input whose content (beyond the version header) changed from build_v6 to build_v7, and every '
    'input listed as blocking in the TD3C5 audit (must_read), is read end to end in its final content by a reader '
    'subagent that did not write the masking code; every input outside the earlier end-to-end reads (reviewers TD3C, '
    'TD3BC, TD3C3, TD3C4, TD3C5 and the v5 reader subagents) would be read as well. v5 (TD3C4 task step 6): every input '
    'that no reviewer read end to end in an earlier round (TD3C 45, TD3BC 42, '
    'TD3C3 40 and TD3C4 40 end-to-end reads; flagged-article re-reads do not count) and every input whose content '
    '(beyond the version header) changed from build_v5 to build_v6 is read end to end by a reader subagent that did not '
    'write the masking code, under a written brief (report any readable focal or shown-partner name, ticker, RIC, '
    'abbreviation, code or unit name, and any mask whose fixed phrase gives back its word, with rank, token and exact '
    'text). The builder checks every reported item against the raw text, fixes the masker or table, and has every '
    'input whose content changed after its read re-read by a reader. The record lists the reads by reader, every '
    'reported item with its adjudication, and must report zero readable focal or shown-partner identifiers and zero '
    'give-back masks in the final inputs.')
INPUT_RULE = (
    'One frozen input per firm-week, byte-identical for every selector: menu with economic meanings and risk '
    'channels, focal characteristics (TD2 C values, availability age, same-cutoff percentile among primary firm-weeks), '
    'exact-hop graph counts and shown partners with role paths and the age of the newest relationship record on each '
    'path (days from the vendor record date, TD2 max_edge_available_at, to the forecast date), and the masked '
    'articles. Calendar dates and years are withheld; ages are calendar days before the origin (forecast) date. No '
    'firm identifier, PERMNO, label or return enters it.')
INTERPRETATION_DATED_STATE = (
    'Interpretation for the collaborator to confirm: the TODO\'s "dated graph/firm state" is rendered as the '
    'point-in-time state as of the cutoff (only information available before the cutoff) with its timing kept as '
    'ages in calendar days before the forecast date (characteristic ages, relationship-record ages, article ages); '
    'absolute calendar dates and years are withheld to limit recall. The input is byte-identical for every selector, '
    'so withholding dates does not bias the DeepSeek-versus-Jev comparison.')
LIMITATION_CONTEXT_DATES = (
    'Context-based date inference cannot be removed: events, product and variant names (Omicron BA.4/BA.5), fiscal '
    'wording ("quarter ended in September"), named executives and weekdays can date an article approximately even '
    'with calendar dates and years withheld. Masking likewise cannot stop identification from context (brand names, '
    'executives, products); the project found masked earnings calls identified 98.6% of the time.')
LIMITATION_CONTEXT_IDENTIFICATION = (
    'm3 (TD3BC; v5 text rewritten after TD3C4 R1; v6 adds the unit-name line after TD3C5 D1). Context identification '
    'remains: the masker removes names, tickers, '
    'RICs, firm codes and the firm-owned names listed for a firm, not knowledge. The examples use an invented firm '
    'ZORBEX INC (ticker ZRBX, FactSet name "ZorBex Inc.", a private-table entry with the product phrase "Zorbcloud '
    'Services", the street phrase "Way", the joint partner Quillco and the code prefix QX, and a name "ZORB LOGISTICS '
    'INC" in the name histories) and an invented firm Q R T BIO INC; each example is checked by a test '
    '(LIMITATION_EXAMPLES). Masked as a whole by the firm\'s token: (1) a single token that contains a firm name '
    'word, ticker or acronym: a distinctive name word of >= 4 letters followed by a digit, a camel-case boundary or '
    'at most two letters ("Zorbexon" -> [X]), a distinctive name word of >= 5 letters followed, in the same case '
    'style, by up to four letters or by more letters that are not an English word ("Zorbexpro" -> [X]; "Zorbexweapons" '
    'stays readable), a ticker or acronym of >= 3 letters that is not an English word starting '
    'an upper-case token with at most two more letters or with digits only ("ZRBXP" -> [X]) or after a lower-to-upper '
    'case boundary ("ZetaZRBX" -> [X]); (2) a product or drug code PREFIX[-]digits[letters] whose prefix, in upper or '
    'Title case, is a ticker or RIC root of the firm, an initialism of one of its names (camel-case parts of a '
    'mixed-case name count as words), its first name word or a leading-letter abbreviation of that word of two letters '
    'or more, or a prefix listed for the firm in the private table ("ZRBX1181", "Zrbx1181", "ZRBX-1181", '
    '"ZORBEX1181", "Zorbex-1181", "ZOR-09", "ZB-12", "QX-6137" -> [X]), and a deal code name "Project <prefix>" '
    'with such a prefix ("signed by Project ZB Parent" -> "signed by [X]"), and (v6) a unit name made of the firm\'s '
    'initialism (two letters or more) and a corporate-unit word ("ZB Ventures sold its stake" -> "[X] sold its stake"; '
    'a ticker before such a word is masked by the ticker rule, "ZRBX Ventures rose" -> "[X] Ventures rose"); a code on '
    'the frozen general-vocabulary '
    'stop list (CO2, N95, COVID-19, ...) and a code with any other prefix ("KQ-17") stay readable; (3) letters or '
    'digits glued to a token without a separator join the token ("Q.R.T.515" for Q R T BIO INC -> [Y]; no token is '
    'ever glued to a letter or digit); (4) a near name: a two-word name from the CRSP or FactSet name histories whose '
    'first word is the firm\'s first name word minus one or two trailing letters, or plus one or two letters, written '
    'not all in lower case ("Zorb Logistics said" -> "[X] said"); (5) a multi-word product, unit, subsidiary, asset, '
    'street, program or joint-product name written right next to the token when its phrase is listed for that firm '
    'in the private table, with its acronyms in the same article ("Zorbex Zorbcloud Services grew" -> "[X] grew"), '
    'and a house number before a listed street phrase ("455 Zorbex Way" -> "[X]"); the name rule takes a possessive '
    'with the name ("Zorbex\'s Zorbcloud Services unit" -> "[X] unit"); (6) a joint partner listed for the firm in the '
    'table and joined to the token by and, &, a hyphen, slash, plus, "and (its) partner" or a space, in any context '
    '("Zorbex and Quillco said on Monday" -> "[X] said on Monday"); (7) a person who shares the firm\'s name word, at '
    'the surname and at later mentions ("Founder Jane Zorbex said" -> "Founder Jane [X] said"). Stays readable: '
    'product, brand, subsidiary and asset names that contain no firm name word, written apart from the token '
    '("Quorbix, the Zorbex cloud, grew" -> "Quorbix, the [X] cloud, grew"); a multi-word firm-owned name next to the '
    'token whose phrase is not in the private table (built from the 200 frozen inputs; in a different sample such a '
    'name would be masked only at the firm word); rivals and partners not listed in the table ("Zorbex and Quillmark '
    'said" -> "[X] and Quillmark said"); executives; industry niches, events, nationality and headquarters-city '
    'phrases that are place words; fixed phrases left as written by the context guard because they name an index, a '
    'price or rate, an idiom, a material or a different entity; a role token\'s position in an alphabetical research '
    'list, which shows the range of the hidden name\'s first letters. The news agency is identifiable from mentions of '
    'its data and commentary products, so a story about the owner of the agency can point to that masked firm.')
# v5 (TD3C4 R1): the invented examples of LIMITATION_CONTEXT_IDENTIFICATION as (firm, raw text, masked text); a test
# builds the two invented firms (with the invented table entries and name-history row named in the text) and checks
# every pair.
LIMITATION_EXAMPLES = (
    ('ZORBEX INC', 'Zorbexon cut costs', '[X] cut costs'),
    ('ZORBEX INC', 'Zorbexpro launch', '[X] launch'),
    ('ZORBEX INC', 'Zorbexweapons were seized', 'Zorbexweapons were seized'),
    ('ZORBEX INC', 'ZRBXP shares', '[X] shares'),
    ('ZORBEX INC', 'ZetaZRBX deal', '[X] deal'),
    ('ZORBEX INC', 'PHASE 1 DATA FOR ZRBX1181 IN TUMORS', 'PHASE 1 DATA FOR [X] IN TUMORS'),
    ('ZORBEX INC', 'Phase 1 data for Zrbx1181', 'Phase 1 data for [X]'),
    ('ZORBEX INC', 'ZRBX-1181 trial', '[X] trial'),
    ('ZORBEX INC', 'ZORBEX1181 study', '[X] study'),
    ('ZORBEX INC', 'Zorbex-1181 study', '[X] study'),
    ('ZORBEX INC', 'Trial Of ZOR-09 In Adults', 'Trial Of [X] In Adults'),
    ('ZORBEX INC', 'ZB-12 data', '[X] data'),
    ('ZORBEX INC', 'QX-6137 study', '[X] study'),
    ('ZORBEX INC', 'the bid signed by Project ZB Parent passed', 'the bid signed by [X] passed'),
    ('ZORBEX INC', 'ZB Ventures sold its stake', '[X] sold its stake'),
    ('ZORBEX INC', 'ZRBX Ventures rose', '[X] Ventures rose'),
    ('ZORBEX INC', 'CO2 and N95 and COVID-19 rules', 'CO2 and N95 and COVID-19 rules'),
    ('ZORBEX INC', 'Trial Of KQ-17 In Adults', 'Trial Of KQ-17 In Adults'),
    ('Q R T BIO INC', 'Trial Of Q.R.T.515 In Adults', 'Trial Of [Y] In Adults'),
    ('ZORBEX INC', 'Zorb Logistics said', '[X] said'),
    ('ZORBEX INC', 'Zorbex Zorbcloud Services grew', '[X] grew'),
    ('ZORBEX INC', 'Site address: 455 Zorbex Way Source', 'Site address: [X] Source'),
    ('ZORBEX INC', "Zorbex's Zorbcloud Services unit", '[X] unit'),
    ('ZORBEX INC', 'Zorbex and Quillco said on Monday', '[X] said on Monday'),
    ('ZORBEX INC', 'Founder Jane Zorbex said', 'Founder Jane [X] said'),
    ('ZORBEX INC', 'Quorbix, the Zorbex cloud, grew', 'Quorbix, the [X] cloud, grew'),
    ('ZORBEX INC', 'Zorbex and Quillmark said', '[X] and Quillmark said'),
)


def menu_rows(path=TD2_MENU):
    """The frozen five-item menu from TD2 risk_menu.csv (order = NAMES, all eligible)."""
    with Path(path).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if [r['name'] for r in rows] != NAMES or any(r['eligible'] != 'True' for r in rows):
        raise ValueError('Risk menu differs from the frozen five eligible items')
    return [{'name': r['name'], 'economic_meaning': r['economic_meaning'], 'risk_channel': r['risk_channel']}
            for r in rows]


SYSTEM_DEEPSEEK = 'You are assisting a finance research experiment. Follow the answer format exactly.'


def deepseek_question(menu):
    names = ', '.join(m['name'] for m in menu)
    return ('Question: Using only the FROZEN INPUT below, rank all five firm characteristics in its RISK MENU by how '
            'relevant each one is to the stock return volatility of the focal firm [FOCAL] over the next five trading '
            'sessions after the forecast date. Put the most relevant characteristic first.\n'
            'Answer format: return only a JSON object of the form {"ordering": ["name_1", "name_2", "name_3", '
            '"name_4", "name_5"]}. The list must contain each of the five menu names exactly once, spelled exactly '
            f'as in the menu ({names}). Do not add any other text.')


def jev_choice_question(menu):
    criteria = {m['name']: f"{m['economic_meaning']}. Risk channel: {m['risk_channel']}." for m in menu}
    criteria[NONE_OPTION] = ('No characteristic in the risk menu is materially relevant to the focal firm\'s stock '
                             'return volatility over the next five trading sessions, given this state.')
    return {CHOICE_KEY: {'type': 'choice',
                         'instructions': ('Using only the state, choose the one firm characteristic from the risk '
                                          'menu that is most relevant to the stock return volatility of the focal '
                                          'firm [FOCAL] over the next five trading sessions after the forecast date. '
                                          'Choose none if no characteristic in the menu is materially relevant.'),
                         'criteria': criteria}}


def jev_noul_questions(menu):
    out = {}
    for m in menu:
        meaning = m['economic_meaning'][0].lower() + m['economic_meaning'][1:]
        channel = m['risk_channel'][0].lower() + m['risk_channel'][1:]
        out[NOUL_PREFIX + m['name']] = {
            'type': 'noul',
            'instructions': (f"Using only the state: is the focal firm [FOCAL]'s {m['name']} ({meaning}) relevant to "
                             'its stock return volatility over the next five trading sessions after the forecast '
                             f'date? Risk channel considered: {channel}. Answer yes if this characteristic is '
                             'materially relevant, otherwise no.')}
    return out


def question_bundle(menu):
    bundle = {'version': QUESTION_VERSION, 'deepseek_system': SYSTEM_DEEPSEEK,
              'deepseek_question': deepseek_question(menu), 'jev_choice': jev_choice_question(menu),
              'jev_noul': jev_noul_questions(menu), 'deepseek_model': DEEPSEEK_MODEL, 'jev_model': JEV_MODEL,
              'deepseek_temperature': DEEPSEEK_TEMPERATURE, 'deepseek_max_tokens': DEEPSEEK_MAX_TOKENS,
              'deepseek_thinking': DEEPSEEK_THINKING, 'deepseek_response_format': 'json_object'}
    return bundle, hashlib.sha256(json.dumps(bundle, sort_keys=True).encode('utf-8')).hexdigest()


def model_checks():
    """Frozen returned-model rules (C5), bound in the freeze and the clearance."""
    return {'deepseek_returned_model_allowlist': sorted(DEEPSEEK_RETURNED_MODEL_PRICES),
            'deepseek_allowlist_prices_equal_PRICES': all(v == PRICES for v in DEEPSEEK_RETURNED_MODEL_PRICES.values()),
            'deepseek_reasoning_allowed': False, 'deepseek_thinking_requested': DEEPSEEK_THINKING,
            'deepseek_reasoning_rule': ('violation when reasoning_content is present and not null (any type, empty '
                                        'included) or completion_tokens_details.reasoning_tokens is present and not '
                                        'null and is non-numeric or > 0'),
            'jev_returned_model_required': JEV_MODEL}


def rules():
    return {'eligibility': ELIGIBILITY_RULE, 'sampling': SAMPLING_RULE, 'article': ARTICLE_RULE,
            'aggregation': AGGREGATION_RULE, 'masking': MASKING_RULE, 'input': INPUT_RULE,
            'budget': (f'Before every attempt, reserve its pre-call upper bound: DeepSeek request bytes x '
                       f'{UPPER_TOKENS_PER_BYTE} tokens/byte + {UPPER_FRAMING_TOKENS} at peak cache-miss price plus '
                       f'{DEEPSEEK_MAX_TOKENS} output tokens at peak output price; Jev request bytes x '
                       f'{UPPER_TOKENS_PER_BYTE} tokens/byte + {UPPER_FRAMING_TOKENS} at USD {JEV_PRICE_INPUT_PER_M} '
                       f'per million input tokens (output free). Stop before any call whose bound would push cumulative '
                       f'estimated USD (DeepSeek + Jev, all real runs and sessions summed, unsettled calls at their '
                       f'bound) above {MAX_USD:.2f}, or any Jev request beyond {MAX_JEV_REQUESTS}; a cap stop halts '
                       f'the run. Settled USD uses reported tokens (DeepSeek: window at call start; Jev: '
                       f'usage.input_tokens x {JEV_PRICE_INPUT_PER_M} per million).'),
            'prices': PRICE_SOURCE,
            'models': (f'DeepSeek requests model {DEEPSEEK_MODEL} with thinking {DEEPSEEK_THINKING}, temperature '
                       f'{DEEPSEEK_TEMPERATURE} and JSON output. Halt after an HTTP-200 response whose returned model '
                       f'is not allowed: DeepSeek returned model must be one of {sorted(DEEPSEEK_RETURNED_MODEL_PRICES)} '
                       '(each priced at PRICES); any reasoning_content that is not null, or a reasoning_tokens count '
                       'that is not null and is non-numeric or positive, is a violation; Jev returned model must equal '
                       f'{JEV_MODEL}. A unit refused by these rules is never re-asked under the same clearance: a '
                       'resume after such a refusal needs a clearance file whose sha256 differs from that of the '
                       'session that recorded the refusal.'),
            'parsing': (f'Jev Choice probabilities must cover the {CHOICE_OPTIONS} options, lie in [0, 1] and sum to 1 '
                        f'within +/-{PROBABILITY_SUM_TOLERANCE} inclusive (v4, TD3C3 N5: {CHOICE_OPTIONS} options x '
                        f'0.005, the largest sum shift from rounding each value to {PROBABILITY_DECIMALS} decimals); '
                        'the raw sum is stored and a note probability_sum_rounded is '
                        'added when it is not exactly 1. Halt when 3 consecutive HTTP-200 responses of one selector '
                        'are all malformed or all invalid with a shared structural issue '
                        f'({", ".join(STRUCTURAL_ISSUES)}).'),
            'halts': ('HTTP 401/402/403; any HTTP 3xx (redirects are never followed); a third other 4xx from one '
                      f'provider; {FAILURE_UNIT_LIMIT} consecutive units of one provider (a separate counter per '
                      'provider: DeepSeek units, Jev units) whose final attempt ended in a transport error or HTTP '
                      '429/5xx; a cap stop; a structural probe; a disallowed returned model or reasoning output.'),
            'run_lock': ('A real run holds an OS byte-range lock on runs/TD3_RUN.lock for its whole session; a second '
                         'real run or resume refuses while it is held.'),
            'mask_tables': ('v4 (TD3C3 E1, N6), v5 (TD3C4 D2): the firm-specific masking tables (common names, '
                            'firm-owned product, unit, street and joint-product names with acronyms, fixed phrases '
                            'keyed by a firm word, firm code prefixes) are one frozen private JSON file, ' +
                            str(MASK_TABLES) + ', schema TD3_MASK_TABLES_v2, with a source note per entry; its '
                            'sha256 is bound in BUILD.json, VERIFY.json and the freeze.'),
            'limitation_examples': [list(x) for x in LIMITATION_EXAMPLES],
            'adjacency_sweep': ADJACENCY_SWEEP_RULE,
            'manual_read': MANUAL_READ_RULE,
            'interpretation_dated_state': INTERPRETATION_DATED_STATE,
            'limitation_context_dates': LIMITATION_CONTEXT_DATES,
            'limitation_context_identification': LIMITATION_CONTEXT_IDENTIFICATION,
            'no_labels': 'No five-session label, return, loss or TD1 manifest is read by any TD3 module.'}
