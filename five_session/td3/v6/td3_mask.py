"""Masking of firm names and tickers with role tokens (pure functions; no readers, no data). v6 (TD3F).

Rules are stated in td3_spec.MASKING_RULE. Matches are first replaced by private-use sentinels so that a later
rule can never match inside an inserted token; sentinels become tokens at the end.

v6 (review TD3C5_AUDIT.md, D1): a unit name made of one of the firm's tickers or initialisms (code prefixes whose rule
is 'ticker' or 'initialism', >= 2 letters, upper or Title case, not on CODE_PREFIX_STOP or ACRONYM_STOP) and a word of
the frozen corporate-unit list UNIT_WORDS is replaced as a whole by the firm's token after the name rules, i.e. where
the prefix is still readable ("ZB Ventures" -> "[X]"; a ticker masked by the ticker rules keeps the unit word:
"[X] Ventures"; Masker._unit_pass); residual() reports a readable pair ('initialism_unit').

v5 (review TD3C4_AUDIT.md, D1, D2, R1, N-a to N-d):
- D1 glued tokens: after every other pass, a token glued to letters or digits on either side (no separator) takes the
  whole alphanumeric word, digits included ("QRT515" masked at "QRT" -> "[X]", never "[X]515"); residual() reports
  any token still glued to a letter or digit.
- D1 house numbers: when a token absorbs a street word of its own firm (STREET_WORDS: "[X] Way"), a 1-6 digit house
  number right before the token is absorbed as well ("455 Zorbex Way" -> "[X]").
- D2 firm codes: a product or drug code PREFIX[-]digits[letters] whose PREFIX is one of the firm's code prefixes
  (code_prefix_terms: its tickers and RIC roots, initialisms of its names (camel-case parts count as words),
  leading-letter abbreviations of its first name word (two letters or more), its first name word, and the prefixes of
  the private table's code_prefixes section) is replaced as a whole by the firm's token before the name rules run
  ("ZRX-09", "Qlvn1181" -> "[X]"); codes on the frozen general-vocabulary stop list NON_FIRM_CODES are kept.
- D2 near names: the build adds to a firm's common names every two-word name from the CRSP or FactSet name
  histories whose first word is the firm's ticker, acronym or first name word minus one or two trailing letters, or
  plus one or two letters (td3_build.near_names); they are masked like common names.
- R1 embedded names: a distinctive name word of >= 5 letters at the start of a longer token followed, in the same
  case style, by up to four letters or by more letters that are not an English word masks the whole token
  ("Zorbexpro" -> "[X]"); a ticker or acronym starting an upper-case token followed by digits only masks it too.
- N-a ordinal dates ("the 30th of September", "30th September", "September the 30th") become [DATE]; N-b bracketed
  story IDs may start with P (the press-release IDs after "Source text for Eikon:") and runs of unbracketed alert
  numbers are removed; N-c the joint-partner and prefix passes run inside the product loop, so a partner that
  becomes adjacent only after a product is absorbed is replaced too.

v4 (review TD3C3_AUDIT.md, E1, E2, N1-N4, N6):
- E1: a frozen, source-noted private table (td3_spec.MASK_TABLES, 'firm_products') lists, per CRSP comnam prefix, the
  product and unit phrases that follow the firm word in a name owned by the firm, phrases that precede it, and
  joint-product partners. A role token immediately followed by one of its firm's phrases (any case unless the entry
  is marked capitalized, optional registered or trademark signs, optional possessive) absorbs the phrase
  ("[X] Zorbcloud Services" -> "[X]"); a preceding phrase of the firm is absorbed the same way; a joint-product
  partner joined to the token ("Quillco and [X]", "[X]-Quillco", "[X] and partner Quillco") is replaced as a whole by
  the token; an acronym made from the firm word and the absorbed words, or listed in the table, is masked in the same
  article when a phrase was absorbed there (Masker.mask_article). residual() reports a token still followed by a
  phrase of its own firm.
- E2: a source typo of a month before "<day>, [YEAR]" becomes [DATE] (words on NON_MONTH_STOP excepted), and a
  day-range remnant ("[DATE]-20", "[DATE] and 31") becomes [DATE].
- N1: the person guards are removed: every mention of a firm word is masked, also a person's surname after a given
  name ("Founder Michael [X]"); a masked surname does not give the hidden word back.
- N2: the place guard is removed: a firm word after a place word is masked ("Fire at Ohio [X] refinery"); place words
  themselves are still never single-word variants.
- N3: the fixed-phrase guards are restricted to the index, price or idiom phrases themselves (private table).
- N6: every firm-specific table (common names, products, joint products, fixed phrases keyed by a firm word) is in the
  private table file, loaded by the build with set_tables(); the defaults here are empty and the tests install
  invented tables. Examples in this file are invented.

v2 alias generation (review TD3C C1): CRSP names are normalized before variant generation. Runs of single
letters or digits collapse to an acronym written with periods ("Q R T SYSTEMS" -> "Q.R.T. SYSTEMS"), a single
letter stays an initial ("J."), single-letter ampersand names compact ("Q T & Q" -> "Q.T.&Q."), and the regex built
from a variant accepts the usual spellings (QRT, Q.R.T., Q. R. T., Q/R), abbreviations or their expansions,
apostrophe possessives inside words (GLORPINOS -> Glorpino's), joiners (space, hyphen, period, apostrophe, nothing)
and optional connectors (of, the, and, &) between words.

v3 (review TD3BC D1/D2): extra issuer tickers and RICs (restricted rules), names embedded in longer tokens
(camel-case compounds, edge acronyms in upper-case tokens, short derivations), edit-distance-1 misspellings of
distinctive name words, a context guard for single-word aliases (fixed phrases, hyphenated particles, person
names, place-led names, headquarters towns, English-word tickers in upper-case text) so that a mask never sits in
a phrase that gives the hidden word away, place words never used as single-word variants, a RIC written right after
a role token takes that token, a replacement log, bylines and nested contact blocks removed, years glued to a time
and generic bare-domain URLs masked. Examples in this file are invented.
"""
import re

LEGAL = {'INC', 'INCORPORATED', 'CORP', 'CORPORATION', 'CO', 'COMPANY', 'COS', 'COMPANIES', 'LTD', 'LIMITED', 'PLC',
         'HOLDINGS', 'HOLDING', 'HLDGS', 'HLDG', 'HLDNGS', 'GROUP', 'GRP', 'NV', 'SA', 'AG', 'SE', 'LLC', 'LP', 'TRUST',
         'TR', 'INTL', 'INTERNATIONAL', 'ENTERPRISES', 'INDUSTRIES', 'INDS', 'TECHNOLOGIES', 'TECHNOLOGY', 'SYSTEMS',
         'BANCORP', 'NEW', 'OLD', 'DEL', 'DE', 'MD', 'PA', 'NJ', 'NY', 'OH', 'VA', 'MN', 'TX', 'CA', 'FL', 'MA', 'IND',
         'THE', 'ADR', 'ADS', 'SPON', 'SPONSORED', 'CL', 'A', 'B', 'C', '&', 'AND', 'QUE', 'ONT', 'ALTA', 'CDA', 'SF',
         'NA', 'BV', 'SAB', 'CV', 'SPA', 'ASA', 'AB', 'AS', 'OYJ', 'KGAA', 'PBC', 'PARTNERS', 'PTNRS', 'PTNR', 'I',
         'II', 'III', 'NC', 'NE', 'NS', 'ME', 'WI', 'GA', 'MI', 'IL', 'MO', 'CT', 'COR', 'IN'}
# Frozen abbreviation map (CRSP comnam conventions). A word matches itself or any listed form.
ABBREV = {
    'INTL': ('INTERNATIONAL',), 'INTERNAT': ('INTERNATIONAL',), 'HLDGS': ('HOLDINGS',), 'HLDG': ('HOLDING',),
    'HLDNGS': ('HOLDINGS',), 'TR': ('TRUST',), 'LABS': ('LABORATORIES', 'LABORATORY'), 'LAB': ('LABORATORY',),
    'GRP': ('GROUP',), 'COS': ('COMPANIES', 'COMPANY'), 'SVCS': ('SERVICES',), 'SVC': ('SERVICE', 'SERVICES'),
    'MFG': ('MANUFACTURING',), 'TECH': ('TECHNOLOGIES', 'TECHNOLOGY'), 'TECHS': ('TECHNOLOGIES',),
    'INVT': ('INVESTMENT',), 'INVTS': ('INVESTMENTS',), 'INVS': ('INVESTORS', 'INVESTMENTS'), 'INV': ('INVESTMENT',),
    'INDS': ('INDUSTRIES',), 'IND': ('INDUSTRIES', 'INDUSTRIAL'), 'PPTYS': ('PROPERTIES',), 'PPTY': ('PROPERTY',),
    'PROP': ('PROPERTIES', 'PROPERTY'), 'PRODS': ('PRODUCTS',), 'PROD': ('PRODUCTS', 'PRODUCTION'),
    'ASSOC': ('ASSOCIATES', 'ASSOCIATION'), 'AMER': ('AMERICA', 'AMERICAN'), 'AMERN': ('AMERICAN',),
    'ACQ': ('ACQUISITION',), 'PWR': ('POWER',), 'ELEC': ('ELECTRIC',), 'MTG': ('MORTGAGE',),
    'PTNRS': ('PARTNERS',), 'PTNR': ('PARTNERS',), 'MED': ('MEDICAL',), 'RES': ('RESOURCES',),
    'AIRLS': ('AIRLINES',), 'TEL': ('TELEPHONE',), 'FINL': ('FINANCIAL',), 'CHEM': ('CHEMICAL', 'CHEMICALS'),
    'CHEMS': ('CHEMICALS',), 'SOL': ('SOLUTIONS',), 'SOLS': ('SOLUTIONS',), 'SYS': ('SYSTEMS',),
    'PHARM': ('PHARMACEUTICAL', 'PHARMACEUTICALS'), 'PHARMS': ('PHARMACEUTICALS',), 'BLDG': ('BUILDING',),
    'CDA': ('CANADA',), 'ENT': ('ENTERTAINMENT',), 'ENTMT': ('ENTERTAINMENT',), 'CMNTYS': ('COMMUNITIES',),
    'SOFTWR': ('SOFTWARE',), 'COMMS': ('COMMUNICATIONS',), 'COMMUN': ('COMMUNICATIONS',),
    'COMMUNS': ('COMMUNICATIONS',), 'MGMT': ('MANAGEMENT',), 'MGT': ('MANAGEMENT',), 'NATL': ('NATIONAL',),
    'BK': ('BANK',), 'BKS': ('BANKS',), 'SVGS': ('SAVINGS',), 'ASSN': ('ASSOCIATION',), 'INS': ('INSURANCE',),
    'MTRS': ('MOTORS',), 'PETE': ('PETROLEUM',), 'EXPL': ('EXPLORATION',), 'ENGR': ('ENGINEERING',),
    'ENGRG': ('ENGINEERING',), 'EQUIP': ('EQUIPMENT',), 'DIST': ('DISTRIBUTION',), 'ELECTRS': ('ELECTRONICS',),
    'TELECOMM': ('TELECOMMUNICATIONS',), 'BIOSCI': ('BIOSCIENCES',), 'THERAP': ('THERAPEUTICS',),
    'THERAPS': ('THERAPEUTICS',), 'RESTR': ('RESTAURANTS',), 'CTR': ('CENTER',), 'CTRS': ('CENTERS',),
    'MACH': ('MACHINES',), 'MACHS': ('MACHINES',), 'MACHY': ('MACHINERY',), 'INSTR': ('INSTRUMENTS',),
    'INSTRS': ('INSTRUMENTS',), 'TRANSN': ('TRANSPORTATION',), 'MATLS': ('MATERIALS',), 'MNG': ('MINING',),
    'DEV': ('DEVELOPMENT',), 'MKTS': ('MARKETS',), 'MKT': ('MARKET',), 'HLTH': ('HEALTH',),
    'HLTHCARE': ('HEALTHCARE',), 'BRDCSTG': ('BROADCASTING',), 'CORP': ('CORPORATION',), 'CO': ('COMPANY',),
    'INC': ('INCORPORATED',), 'LTD': ('LIMITED',), 'FED': ('FEDERAL',), 'SECS': ('SECURITIES',),
    'RLTY': ('REALTY',), 'MTLS': ('METALS',), 'WKS': ('WORKS',), 'SPLY': ('SUPPLY',), 'GEN': ('GENERAL',),
    'BANCSHS': ('BANCSHARES',), 'CMNTY': ('COMMUNITY',), 'FINCL': ('FINANCIAL',), 'TRANSPRT': ('TRANSPORT',),
    'SCIS': ('SCIENCES',), 'SCI': ('SCIENTIFIC', 'SCIENCES'), 'ELECTR': ('ELECTRONICS',),
    'HOSP': ('HOSPITALITY', 'HOSPITALS'), 'AUTOMTN': ('AUTOMATION',), 'MINS': ('MINERALS',),
    'RESH': ('RESEARCH',), 'STNLESS': ('STAINLESS',), 'ALY': ('ALLOY',), 'EST': ('ESTATE',),
    'EQ': ('EQUITY', 'EQUITIES'), 'AIRGROUP': ('AIR GROUP',), 'COR': ('CORP', 'CORPORATION'), 'IN': ('INC',),
    'HLDS': ('HOLDINGS',), 'CMNCTNS': ('COMMUNICATIONS',), 'COMMNS': ('COMMUNICATIONS',), 'TECHNOL': ('TECHNOLOGY',),
    'PHARMACEUTICAL': ('PHARMACEUTICALS',), 'BIOPHARM': ('BIOPHARMA', 'BIOPHARMACEUTICALS'),
    'ALUMINUM': ('ALUMINIUM',),
}
GENERIC = {'american', 'general', 'united', 'first', 'national', 'international', 'global', 'energy', 'financial',
           'technology', 'systems', 'group', 'industries', 'resources', 'capital', 'china', 'new', 'york', 'north',
           'south', 'west', 'east', 'air', 'water', 'power', 'gas', 'oil', 'bank', 'trust', 'health', 'care', 'food',
           'foods', 'home', 'auto', 'motor', 'motors', 'steel', 'paper', 'gold', 'silver', 'delta', 'alpha', 'us',
           'usa', 'the', 'company', 'services', 'solutions', 'products', 'partners', 'data', 'digital', 'media',
           'western', 'eastern', 'southern', 'northern', 'pacific', 'atlantic', 'central', 'union', 'standard',
           'royal', 'premier', 'advanced', 'applied', 'universal', 'continental', 'consolidated', 'allied', 'summit',
           'pioneer', 'liberty', 'eagle', 'world', 'mobile', 'solar', 'natural', 'life', 'federal', 'community',
           'citizens', 'peoples', 'savings', 'credit', 'santa', 'texas', 'california', 'florida', 'great', 'state',
           'city', 'county', 'america', 'europe', 'european', 'asia', 'canada', 'canadian', 'japan', 'investors',
           'investment', 'insurance', 'realty', 'properties', 'electric', 'medical', 'communications', 'total',
           'one', 'real', 'estate', 'equity', 'income', 'select', 'value', 'prime', 'plus', 'best', 'public', 'time',
           'corn', 'copper', 'iron', 'coal', 'cotton', 'sugar', 'wheat', 'soybean', 'cocoa', 'coffee', 'lumber',
           'aluminum', 'nickel', 'zinc', 'lithium', 'uranium',
           # v3: business adjectives and foreign generic company words, never single-word variants
           'corporate', 'commercial', 'industrial', 'strategic', 'integrated', 'grupo', 'groupe', 'gruppo', 'banco',
           'compania', 'compagnie', 'sociedad', 'societe', 'koninklijke', 'cia'}
# v3: place words (US states, large US cities, countries, regions) are never single-word variants: as single words
# they name the place ("in [X] expects", "[X] and Ohio"), and a mask there would give the place away.
PLACE_WORDS = frozenset({
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida',
    'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine',
    'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
    'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
    'wisconsin', 'wyoming', 'chicago', 'detroit', 'boston', 'houston', 'dallas', 'denver', 'atlanta', 'seattle',
    'phoenix', 'philadelphia', 'pittsburgh', 'cleveland', 'cincinnati', 'baltimore', 'miami', 'orlando', 'tampa',
    'portland', 'memphis', 'nashville', 'louisville', 'milwaukee', 'minneapolis', 'omaha', 'austin', 'charlotte',
    'sacramento', 'oakland', 'brooklyn', 'manhattan', 'honolulu', 'indianapolis', 'columbus', 'jacksonville',
    'china', 'japan', 'india', 'canada', 'mexico', 'brazil', 'germany', 'france', 'britain', 'england', 'italy',
    'spain', 'russia', 'korea', 'australia', 'israel', 'ireland', 'singapore', 'taiwan', 'switzerland',
    'netherlands', 'sweden', 'norway', 'denmark', 'finland', 'belgium', 'austria', 'poland', 'turkey', 'chile',
    'peru', 'colombia', 'argentina', 'indonesia', 'malaysia', 'thailand', 'vietnam', 'philippines', 'africa',
    'europe', 'asia', 'america', 'permian', 'appalachia', 'scandinavia', 'london', 'paris', 'tokyo', 'beijing',
    'shanghai', 'toronto', 'montreal', 'vancouver', 'calgary', 'sydney'})
# US states and Canadian provinces for the headquarters-town rule ("The Zelwick, Ohio-based company").
STATES = ('Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida',
          'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
          'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
          'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
          'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
          'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Ontario', 'Quebec',
          'Alberta', 'British Columbia', 'Manitoba', 'Saskatchewan', 'Nova Scotia', 'Ala', 'Ariz', 'Ark', 'Calif',
          'Colo', 'Conn', 'Del', 'Fla', 'Ga', 'Ill', 'Ind', 'Kan', 'Kans', 'Ky', 'La', 'Md', 'Mass', 'Mich', 'Minn',
          'Miss', 'Mo', 'Mont', 'Neb', 'Nev', 'N.H', 'N.J', 'N.M', 'N.Y', 'N.C', 'N.D', 'Okla', 'Ore', 'Pa', 'Penn',
          'R.I', 'S.C', 'S.D', 'Tenn', 'Tex', 'Vt', 'Va', 'Wash', 'W.Va', 'Wis', 'Wyo')
_STATES_RX = '(?:' + '|'.join(re.escape(s) for s in sorted(STATES, key=len, reverse=True)) + r')\.?'
# Common given names (general knowledge; upper case). v4 (N1): no longer a guard (a surname after a given name is
# masked); still used to keep given names out of the embedded-name and misspelling terms (_distinctive).
PERSON_GIVEN = frozenset("""
AARON ABIGAIL ADAM ADRIAN AHMED AISHA ALAN ALBERT ALEX ALEXANDER ALEXANDRA ALICE ALICIA ALLISON AMANDA AMIT AMY ANA
ANAND ANDREA ANDREW ANDY ANGELA ANIL ANITA ANN ANNA ANNE ANNIE ANTHONY ANTONIO ARJUN ARTHUR ASHLEY BARBARA BARRY BEN
BENJAMIN BETH BETTY BOB BONNIE BRAD BRADLEY BRANDON BRENDA BRIAN BRUCE BRYAN CARL CARLOS CAROL CAROLINE CAROLYN
CATHERINE CHANDRA CHARLES CHARLIE CHERYL CHRIS CHRISTIAN CHRISTINA CHRISTINE CHRISTOPHER CINDY CLAIRE CRAIG CYNTHIA
DAN DANIEL DANIELLE DAVID DEBORAH DEBRA DENISE DENNIS DEREK DIANA DIANE DONALD DONNA DOUG DOUGLAS DYLAN EDWARD
ELAINE ELENA ELIZABETH ELLEN EMILY EMMA ERIC ERIN EVAN FRANCIS FRANK FRED GARY GEORGE GERALD GLENN GORDON GREG
GREGORY HANNAH HAROLD HARRY HEATHER HELEN HENRY HOWARD IAN IRENE ISABEL JACOB JAMES JAMIE JANE JANET JASON JEFF
JEFFREY JENNIFER JEREMY JERRY JESSICA JILL JIM JOAN JOE JOEL JOHN JON JONATHAN JORGE JOSE JOSEPH JOSH JOSHUA JOYCE
JUAN JUDITH JULIA JULIE JUSTIN KAREN KATE KATHERINE KATHLEEN KATHY KEITH KELLY KEN KENNETH KEVIN KIM KIMBERLY
KRISTEN KYLE LARRY LAURA LAUREN LAWRENCE LEE LEO LINDA LISA LORI LOUIS LUCY LUIS LYNN MARC MARCUS MARGARET MARIA
MARIE MARIO MARTHA MARTIN MARY MATTHEW MEGAN MELISSA MICHAEL MICHELLE MIKE MOHAMMED NANCY NATALIE NATHAN NEIL
NICHOLAS NICOLE NOAH OLIVER OLIVIA PAMELA PATRICIA PATRICK PAUL PAULA PETER PHILIP PHILLIP PRIYA RACHEL RAHUL RAJ
RAJESH RALPH RANDY RAVI RAYMOND REBECCA RICHARD RICK RITA ROBERT ROGER RON RONALD ROSS RUSSELL RUTH RYAN SAM
SAMANTHA SAMUEL SANDRA SARA SARAH SCOTT SEAN SHANNON SHARON SHAWN SHEILA SIMON SOPHIA STACY STEPHANIE STEPHEN STEVE
STEVEN SUNIL SUSAN TAMMY TERESA TERRY THERESA THOMAS TIFFANY TIM TIMOTHY TINA TODD TOM TONY TRACY TRAVIS TYLER
VALERIE VICTOR VICTORIA VIKRAM VINCENT WALTER WAYNE WENDY WILLIAM YOLANDA ZACHARY""".split())
# v4 (N6): every firm-specific table is private. The frozen table file (td3_spec.MASK_TABLES, schema TABLE_SCHEMA,
# hash-bound in BUILD.json and the freeze) holds, each entry with a source note:
#   common_names   {CRSP comnam prefix: {'aliases': [...], 'source': ...}}: short, former and operating-company names;
#   firm_products  {CRSP comnam prefix: {'products': [{'phrase', 'acronym'?, 'capitalized'?, 'not_before'?}],
#                   'prefixes': [phrase], 'joint_partners': [{'partner': regex text, 'connector'?}], 'source': ...}}:
#                   E1 names owned by the firm (the comnam prefix matches a whole-word prefix of a CRSP name);
#   fixed_phrases  {ALIAS WORD: {'guards': [[left regex or null, right regex or null], ...], 'source': ...}}: a
#                   single-word alias inside one of these index, price, idiom or other-entity phrases is left as
#                   written (the left regex ends at the word, the right regex starts after it; case-insensitive).
# The defaults are empty; set_tables() installs a table (the build loads the frozen file; tests install invented
# tables). The dicts are updated in place so that every reference stays valid.
#   code_prefixes  {CRSP comnam prefix: {'prefixes': [...], 'source': ...}} (v5, TD3C4 D2): product and drug code
#                   prefixes of the firm that the code_prefix_terms rules do not derive (letters, >= 2).
TABLE_SCHEMA = 'TD3_MASK_TABLES_v2'
COMMON_NAMES = {}
FIXED_PHRASES = {}
FIRM_PRODUCTS = {}
CODE_PREFIXES = {}
_TABLE_STATE = {'sha256': None, 'source': 'empty default'}
# v5 (TD3C4 D1): street words; a house number before a token that absorbed one of its firm's street phrases is
# absorbed too.
STREET_WORDS = frozenset({'WAY', 'STREET', 'ST', 'AVENUE', 'AVE', 'ROAD', 'RD', 'DRIVE', 'DR', 'BOULEVARD', 'BLVD',
                          'PARKWAY', 'PKWY', 'LANE', 'LN', 'PLACE', 'PL', 'COURT', 'CT', 'PLAZA', 'CIRCLE', 'TERRACE',
                          'HIGHWAY', 'HWY', 'SQUARE', 'SQ'})
# v5 (TD3C4 D2): general-vocabulary codes that are never masked as a firm code (chemical formulas, particle sizes,
# respirator classes, disease and gene names, standards). Upper case, hyphen removed.
NON_FIRM_CODES = frozenset({'CO2', 'NO2', 'SO2', 'CH4', 'H2O', 'H2S', 'NH3', 'N2O', 'PM10', 'PM2', 'N95', 'KN95',
                            'FFP2', 'FFP3', 'COVID19', 'COV2', 'IL6', 'IL17', 'IL23', 'PD1', 'CD19', 'CD20', 'HER2',
                            'BRCA1', 'BRCA2', 'MP3', 'MP4', 'IPV4', 'IPV6', 'COP26', 'COP27', 'ISO9001', 'HFC134',
                            # credit-rating grades of a rating agency
                            'AA1', 'AA2', 'AA3', 'BAA1', 'BAA2', 'BAA3', 'BA1', 'BA2', 'BA3', 'CAA1', 'CAA2', 'CAA3'})
CODE_MAX_DIGITS = 8
# v6 (TD3C5 D1, correction 2): frozen corporate-unit words. A firm's ticker or initialism followed by one of these
# words names a unit of the firm ("ZB Ventures") and is masked whole. Source: the TD3C5 audit's list, with the singular
# forms added.
UNIT_WORDS = ('Ventures', 'Venture', 'Capital', 'Holdings', 'Holding', 'Partners', 'Partner', 'Investments',
              'Investment', 'Midstream', 'Energy', 'Financial', 'Technologies', 'Technology', 'Solutions', 'Solution',
              'Labs', 'Lab', 'Bio', 'Pharma', 'Therapeutics', 'Properties', 'Property', 'Realty', 'Services', 'Service',
              'Systems', 'System', 'Group', 'International', 'Global', 'Industries', 'Industry', 'Fund', 'Trust',
              'Bank')
# A single-word alias that is one of these verbs, followed by a hyphenated particle, is a compound ("dine-in",
# "plug-in", "sell-off"), not the firm.
PARTICLE_VERBS = frozenset({'DINE', 'PLUG', 'CHECK', 'TAKE', 'SIGN', 'LOG', 'PICK', 'DRIVE', 'WALK', 'CARRY', 'PLAY',
                            'LOCK', 'BUY', 'SELL', 'BACK', 'TOP', 'ROLL', 'START', 'SET', 'PAY', 'CASH', 'SHUT', 'LAY',
                            'PULL', 'PUSH', 'STAND', 'SPIN', 'BREAK', 'HOLD', 'LOOK', 'RUN', 'TURN', 'WIND', 'WORK',
                            'MAKE', 'SHAKE', 'LEAD', 'KNOCK', 'DROP', 'FOLLOW', 'SIT', 'CUT', 'BUILD', 'WRITE', 'PUT'})
PARTICLE_RX = re.compile(r'^-(?:in|out|up|off|on|down|over|back|through)\b', re.I)
ACRONYM_STOP = {'A', 'I', 'AI', 'IT', 'ON', 'ALL', 'ARE', 'CEO', 'CFO', 'COO', 'CTO', 'US', 'UK', 'EU', 'UN', 'GDP',
                'IPO', 'FDA', 'SEC', 'FED', 'FTC', 'DOJ', 'EPA', 'USA', 'NEW', 'ONE', 'NOW', 'BIG', 'OUT', 'CAN', 'FOR',
                'AND', 'THE', 'HAS', 'WAS', 'ANY', 'GO', 'SO', 'DO', 'BE', 'AT', 'BY', 'IN', 'OF', 'OR', 'TO', 'UP',
                'IS', 'AS', 'AN', 'WE', 'NO', 'OK', 'TV', 'PC', 'EV', 'EVS', 'CO', 'LP', 'PLC', 'INC', 'LTD', 'LLC',
                'EPS', 'ESG', 'LNG', 'OPEC', 'AM', 'PM', 'ET', 'GMT', 'UTC', 'EST', 'EDT', 'Q1', 'Q2', 'Q3', 'Q4', 'FY',
                'YOY', 'ETF', 'IMF', 'WHO', 'NATO', 'NYSE', 'HK', 'EUR', 'USD', 'GBP', 'JPY', 'CNY', 'BRIEF', 'UPDATE',
                'RPT', 'DJ', 'PE', 'VC', 'HR', 'PR', 'AG', 'SA', 'NV', 'SE', 'DE', 'MD', 'PA', 'NJ', 'NY', 'TX', 'CA',
                'FL', 'IL', 'OH', 'GA', 'NC', 'VA', 'WA', 'MA', 'MI', 'RE', 'SEE', 'NET', 'TOP', 'FOCAL', 'LINK',
                'DATE', 'YEAR', 'SUPPLIER', 'CUSTOMER', 'LINKED', 'URL', 'EMAIL', 'AGENCY',
                # v2: months, weekdays, macro and market acronyms that collide with collapsed CRSP initials
                'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUNE', 'JUL', 'JULY', 'AUG', 'SEP', 'SEPT', 'OCT', 'NOV',
                'DEC', 'MON', 'TUE', 'TUES', 'WED', 'THU', 'THUR', 'THURS', 'FRI', 'SAT', 'SUN', 'CPI', 'PPI', 'PMI',
                'ISM', 'API', 'NAV', 'AUM', 'GAAP', 'IFRS', 'SPAC', 'NFT', 'REIT', 'GSE', 'CMS', 'RBC', 'OTC',
                'LLP', 'LBO', 'ROE', 'ROI', 'ROA', 'EBIT', 'AGM', 'EGM', 'CAGR', 'YTD', 'ECB', 'BOE', 'BOJ', 'FOMC',
                'IEA', 'EIA', 'WTI', 'CDC', 'NIH', 'FAA', 'FCC', 'FERC', 'NRC', 'DOE', 'DOT', 'HHS', 'IRS', 'DHS',
                'DOD', 'NASA', 'USDA', 'OSHA', 'SPX', 'TSX', 'ASX', 'LSE', 'NA', 'BV', 'SAB',
                'CV', 'SPA', 'ASA', 'AB', 'PBC', 'LPG', 'CNG', 'PPE', 'ICU', 'SUV', 'SUVS', 'RNA', 'DNA', 'MRNA',
                'COVID', 'SARS', 'BA', 'NEWS', 'TECH', 'LIVE', 'SAYS', 'SEES', 'PER', 'MLN', 'BLN', 'PCT', 'BPS',
                'PRE', 'POST', 'NON', 'III', 'II', 'IV', 'VI', 'VII', 'VIII', 'IX', 'XI', 'XII'}
AMP_STOP = {'M&A', 'R&D', 'S&P', 'P&L', 'Q&A', 'B&B', 'P&C', 'E&P', 'O&G', 'L&D', 'F&B', 'G&A', 'D&O', 'R&R', 'T&E'}
# v4 (N6): only given names; a middle name used by one listed firm moved to the private common-name table.
GIVEN_NAMES = {'STEVEN', 'STEVE', 'WALT', 'ELI', 'CHARLES', 'ARTHUR', 'GEORGE', 'JOHN', 'WILLIAM', 'HENRY',
               'EDWARD', 'THOMAS', 'RICHARD', 'JOSEPH', 'RALPH', 'CALVIN', 'TOMMY', 'HARRY', 'FREDERICK', 'ALBERT',
               'WALTER', 'MICHAEL', 'DAVID', 'PETER', 'FRANK', 'LOUIS', 'SAMUEL', 'BENJAMIN', 'CORNELIUS'}
# v3: place and commodity words removed (as single words they name the place or the commodity).
OWNER_GENERIC = {'delta', 'alpha', 'eagle', 'liberty', 'pioneer', 'summit', 'royal', 'premier', 'universal',
                 'continental', 'consolidated', 'allied', 'solar', 'total'}
_MONTH_WORDS = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                'November', 'December', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sept', 'Sep', 'Oct', 'Nov', 'Dec')
# v2 (manual read): month names also in upper case (BRIEF bodies: "ON APRIL 28", "AS OF 30 JUNE")
MONTHS = '(?:' + '|'.join(sorted({w for m in _MONTH_WORDS for w in (m, m.upper())}, key=len, reverse=True)) + r')\.?'
EXCHANGES = r'(?:NYSE American|NYSE MKT|NYSE Arca|NYSE|NASDAQ|Nasdaq|AMEX|OTC|TSX)'
_B = r'(?<![A-Za-z0-9])'
_E = r'(?![A-Za-z0-9])'
_E_ACRO = r'(?![A-Za-z])'              # a standalone acronym may be followed by digits (drug codes such as QRX515)
_SENT_OPEN, _SENT_CLOSE = '', ''
_TOKEN = re.compile(r'\[[A-Z0-9_]+\]')
_NEUTRAL = ''
_ACRO = re.compile(r'(?:[A-Z0-9]\.){2,}')
_INIT = re.compile(r'[A-Z0-9]\.')
_AMP = re.compile(r'((?:[A-Z0-9]\.)+)&((?:[A-Z0-9]\.)+)')
TRUNCATION_LENGTH = 30                 # CRSP comnam is cut at 32 characters, FactSet names at about 30
# Two-word place names and phrases never used as a first-two-words variant (NEW YORK ZORBANE CORP must not mask "New York").
PHRASE_STOP = {'NEW YORK', 'UNITED STATES', 'NORTH AMERICA', 'NORTH AMERICAN', 'SOUTH AMERICA', 'SOUTH AMERICAN',
               'LATIN AMERICA', 'CENTRAL AMERICA', 'NEW JERSEY', 'NEW ENGLAND', 'NEW MEXICO', 'NEW ZEALAND',
               'NORTH CAROLINA', 'SOUTH CAROLINA', 'NORTH DAKOTA', 'SOUTH DAKOTA', 'WEST VIRGINIA', 'SOUTH AFRICA',
               'GREAT BRITAIN', 'HONG KONG', 'SAN FRANCISCO', 'LOS ANGELES', 'LAS VEGAS', 'SAN DIEGO', 'SAN JOSE',
               'SANTA CLARA', 'SANTA MONICA', 'SANTA FE', 'EAST COAST', 'WEST COAST', 'GULF COAST', 'MIDDLE EAST',
               'FAR EAST', 'UNITED KINGDOM', 'UNITED NATIONS', 'FIRST QUARTER', 'REAL ESTATE', 'NATURAL GAS',
               'HEALTH CARE', 'WALL STREET', 'AIR TRANSPORT', 'SILICON VALLEY', 'PERMIAN BASIN', 'WEST TEXAS'}
ABSORB_STOP = {'THE', 'OF', 'AND', 'A', 'AN', 'DE', 'DEL', 'DU', 'NEW', 'OLD', 'FOR', 'IN', 'ON', 'AT', 'TO', 'BY',
               'IS', 'AS', 'OR', 'IT', 'US', 'WE', 'ITS', 'HAS', 'WAS', 'ARE', 'SAID', 'SAYS'}
_LETTER_SEP = r"(?:\.\s?|\s|/|-)?"
_JOIN = r"(?:[\s\-.,'’/]*|[\s\-.,]+(?:of|the|and|&|de|du)[\s\-.]+)"
_AMP_JOIN = r'\s*(?:&|and|\+)\s*'


def _clean(name):
    name = re.sub(r"['’]", '', name or '')
    name = re.sub(r'[,.;:()"/]', ' ', name)
    name = re.sub(r'&', ' & ', name)
    return ' '.join(name.upper().split())


def normalize(comnam):
    """CRSP comnam -> word list. Acronyms are written 'Q.R.T.', initials 'J.', compact ampersands 'A.T.&T.'."""
    words = _clean(comnam).split()
    out, run = [], []

    def flush():
        if run:
            out.append(''.join(c + '.' for c in run))
            run.clear()
    for w in words:
        if len(w) == 1 and w.isalnum():
            run.append(w)
        else:
            flush()
            out.append(w)
    flush()
    def mark(w):
        return bool(_ACRO.fullmatch(w) or _INIT.fullmatch(w))
    merged = []
    i = 0
    while i < len(out):
        if i + 2 < len(out) and out[i + 1] == '&' and mark(out[i]) and mark(out[i + 2]):
            merged.append(out[i] + '&' + out[i + 2])
            i += 3
        else:
            merged.append(out[i])
            i += 1
    return merged


def _letters(w):
    return w.replace('.', '').replace('&', '')


def _is_short_mark(w):
    """Initial, acronym or compact ampersand of at most two letters (person initials such as J. M. or H.&R.)."""
    return bool(_INIT.fullmatch(w) or (_ACRO.fullmatch(w) and len(_letters(w)) <= 2) or
                (_AMP.fullmatch(w) and len(_letters(w)) <= 3))


def _strippable(w):
    return w in LEGAL or (_letters(w) in LEGAL and bool(_ACRO.fullmatch(w) or _INIT.fullmatch(w)))


def _strip(words):
    words = list(words)
    stages = []
    while words and _strippable(words[-1]):
        words = words[:-1]
        if words:
            stages.append(list(words))
    return words, stages


def _inverted(words):
    """Natural order of an inverted CRSP person name: SURNAME + initials/given names (+ rest) -> initials/given
    names + SURNAME (+ rest). Returns a list of variants (possibly empty)."""
    if len(words) < 2 or len(words[0]) < 3 or not words[0].isalpha() or words[0].lower() in GENERIC:
        return []
    seg = []
    for w in words[1:]:
        if _is_short_mark(w) or w in GIVEN_NAMES:
            seg.append(w)
        else:
            break
    if not seg:
        return []
    rest = [w for w in words[1 + len(seg):]]
    out = [' '.join(seg + [words[0]])]
    if rest and rest[-1] != '&':
        out.append(' '.join(seg + [words[0]] + rest))
    return out


FORM_WORDS = {'INC', 'INCORPORATED', 'CORP', 'CORPORATION', 'CO', 'COMPANY', 'COS', 'COMPANIES', 'LTD', 'LIMITED',
              'PLC', 'LLC', 'LP', 'NV', 'SA', 'AG', 'SE', 'NEW', 'OLD', 'DEL', 'DE', 'CL', 'ADR', 'ADS', 'SPON', 'A', 'B',
              'C', 'I', 'II', 'III', 'NA', 'BV', 'SAB', 'CV', 'SPA', 'ASA', 'AB', 'AS', 'OYJ', 'KGAA', 'PBC', 'IN', 'COR',
              'MD', 'PA', 'NJ', 'NY', 'OH', 'VA', 'MN', 'TX', 'CA', 'FL', 'MA', 'IND', 'QUE', 'ONT', 'ALTA', 'CDA', 'SF',
              'NC', 'NE', 'NS', 'ME', 'WI', 'GA', 'MI', 'IL', 'MO', 'CT'}


def word_initials(comnam):
    """Initials of a name's words (connectors skipped, trailing corporate-form words dropped, acronyms contributing
    all their letters): QUANWOOD REAL ESTATE INVT TR -> QREIT, NEW ZORQ COMMUNITY BANCORP INC -> NZCB. The build
    keeps 3-6 letter results that are not English words and not on ACRONYM_STOP."""
    words = [w for w in normalize(comnam) if w not in ('THE', 'OF', 'AND', '&', 'DE', 'DU')]
    while words and (words[-1] in FORM_WORDS or _letters(words[-1]) in FORM_WORDS):
        words = words[:-1]
    if len(words) < 2:
        return ''
    out = ''
    for w in words:
        if _ACRO.fullmatch(w) or _AMP.fullmatch(w) or _INIT.fullmatch(w):
            out += _letters(w)
        else:
            out += w[0]
    return out if out.isalpha() else ''


def common_aliases(comnams):
    keys = set()
    for c in comnams:
        norm = ' '.join((c or '').upper().split())
        for k, v in COMMON_NAMES.items():
            if norm.startswith(k):
                keys.update(v)
    return keys


def _check_regex(text, where):
    try:
        re.compile(text, re.I)
    except re.error as e:
        raise ValueError(f'Invalid regex in mask table ({where}): {e}') from None


def validate_tables(tables):
    """Schema check of a mask table (v4). Every entry needs a non-empty source note. Returns the table."""
    if not isinstance(tables, dict) or tables.get('schema') != TABLE_SCHEMA:
        raise ValueError('Mask table schema must be ' + TABLE_SCHEMA)
    for section in ('common_names', 'firm_products', 'fixed_phrases', 'code_prefixes'):
        if not isinstance(tables.get(section), dict):
            raise ValueError('Mask table section missing: ' + section)
        for key, entry in tables[section].items():
            if not isinstance(key, str) or not key.strip() or key != ' '.join(key.upper().split()):
                raise ValueError(f'Mask table key must be upper case with single spaces: {section}/{key!r}')
            if not isinstance(entry, dict) or not str(entry.get('source', '')).strip():
                raise ValueError(f'Mask table entry without a source note: {section}/{key}')
    for key, e in tables['common_names'].items():
        if not e.get('aliases') or not all(isinstance(a, str) and a.strip() for a in e['aliases']):
            raise ValueError('common_names entry without aliases: ' + key)
    for key, e in tables['firm_products'].items():
        for p in e.get('products', []):
            if not isinstance(p, dict) or not str(p.get('phrase', '')).strip():
                raise ValueError('firm_products phrase missing: ' + key)
            if p.get('not_before'):
                _check_regex(p['not_before'], key)
            if p.get('acronym') is not None and not re.fullmatch(r'[A-Z][A-Z0-9]{1,7}', p['acronym']):
                raise ValueError('firm_products acronym must be upper case: ' + key)
            for flag in ('capitalized', 'model'):
                if p.get(flag) is not None and not isinstance(p[flag], bool):
                    raise ValueError(f'firm_products {flag} must be true or false: ' + key)
            unknown = set(p) - {'phrase', 'acronym', 'capitalized', 'not_before', 'model'}
            if unknown:
                raise ValueError(f'firm_products unknown fields {sorted(unknown)}: ' + key)
        for p in e.get('prefixes', []):
            if not isinstance(p, str) or not p.strip():
                raise ValueError('firm_products prefix must be a phrase: ' + key)
        for j in e.get('joint_partners', []):
            if not isinstance(j, dict) or not str(j.get('partner', '')).strip():
                raise ValueError('firm_products joint partner missing: ' + key)
            _check_regex(j['partner'], key)
        if not (e.get('products') or e.get('prefixes') or e.get('joint_partners')):
            raise ValueError('firm_products entry is empty: ' + key)
    for key, e in tables['fixed_phrases'].items():
        guards = e.get('guards')
        if not guards or not all(isinstance(g, (list, tuple)) and len(g) == 2 and any(g) for g in guards):
            raise ValueError('fixed_phrases entry needs [left, right] guards: ' + key)
        for g in guards:
            for rx in g:
                if rx is not None:
                    _check_regex(rx, key)
    for key, e in tables['code_prefixes'].items():          # v5 (TD3C4 D2)
        p = e.get('prefixes')
        if not p or not all(isinstance(x, str) and re.fullmatch(r'[A-Z]{2,8}', x) for x in p):
            raise ValueError('code_prefixes entry needs upper-case letter prefixes of 2-8 letters: ' + key)
    return tables


def set_tables(tables, sha256=None, source=None):
    """Install a mask table (validated). The module dicts are replaced in place."""
    validate_tables(tables)
    COMMON_NAMES.clear()
    COMMON_NAMES.update({k: tuple(e['aliases']) for k, e in tables['common_names'].items()})
    FIXED_PHRASES.clear()
    FIXED_PHRASES.update({k: tuple((g[0], g[1]) for g in e['guards']) for k, e in tables['fixed_phrases'].items()})
    FIRM_PRODUCTS.clear()
    FIRM_PRODUCTS.update({k: {'products': [dict(p) for p in e.get('products', [])],
                              'prefixes': list(e.get('prefixes', [])),
                              'joint_partners': [dict(j) for j in e.get('joint_partners', [])]}
                          for k, e in tables['firm_products'].items()})
    CODE_PREFIXES.clear()
    CODE_PREFIXES.update({k: tuple(e['prefixes']) for k, e in tables['code_prefixes'].items()})
    _TABLE_STATE.update(sha256=sha256, source=source)
    _COMPILED.clear()
    return tables


def clear_tables():
    """Empty tables (the module default)."""
    COMMON_NAMES.clear()
    FIXED_PHRASES.clear()
    FIRM_PRODUCTS.clear()
    CODE_PREFIXES.clear()
    _TABLE_STATE.update(sha256=None, source='empty default')


def table_state():
    return dict(_TABLE_STATE, common_names=len(COMMON_NAMES), firm_products=len(FIRM_PRODUCTS),
                fixed_phrases=len(FIXED_PHRASES), code_prefixes=len(CODE_PREFIXES))


def _prefix_match(norm, key):
    return norm == key or norm.startswith(key + ' ')


def firm_products(comnams):
    """E1 (v4): the product, prefix and joint-partner entries of every FIRM_PRODUCTS key that is a whole-word prefix
    of one of the firm's CRSP names. Returns {'products': [...], 'prefixes': [...], 'joint_partners': [...],
    'keys': [...]} (lists sorted, duplicates removed)."""
    out = {'products': {}, 'prefixes': set(), 'joint_partners': {}, 'keys': set()}
    for c in comnams:
        norm = ' '.join(_clean(c).split())
        for k, e in FIRM_PRODUCTS.items():
            if not _prefix_match(norm, k):
                continue
            out['keys'].add(k)
            initial = k[0]
            for p in e['products']:
                item = {'phrase': p['phrase'], 'acronyms': sorted(set(_product_acronyms(initial, p))),
                        'capitalized': bool(p.get('capitalized')), 'not_before': p.get('not_before') or None,
                        'model': bool(p.get('model'))}
                out['products'][json_key(item)] = item
            out['prefixes'].update(e['prefixes'])
            for j in e['joint_partners']:
                item = {'partner': j['partner'], 'connector': j.get('connector') or None}
                out['joint_partners'][json_key(item)] = item
    return {'products': [out['products'][k] for k in sorted(out['products'])], 'prefixes': sorted(out['prefixes']),
            'joint_partners': [out['joint_partners'][k] for k in sorted(out['joint_partners'])],
            'keys': sorted(out['keys'])}


def json_key(item):
    return '|'.join(f'{k}={item[k]}' for k in sorted(item))


# v5 (TD3C4 D2): words that are never code prefixes (general vocabulary: fiscal and calendar words, function words,
# places and agencies), so that "FY22", "NO3" or "US-1" are never read as a firm code.
CODE_PREFIX_STOP = frozenset({'FY', 'CY', 'NO', 'NOS', 'US', 'USA', 'UK', 'EU', 'UN', 'COVID', 'SARS', 'TOP', 'NEW',
                              'THE', 'AND', 'FOR', 'PER', 'PRE', 'POST', 'AM', 'PM', 'VS', 'ID', 'OR', 'AT', 'BY', 'IN',
                              'ON', 'OF', 'TO', 'UP', 'IS', 'AS', 'AN', 'WE', 'BE', 'DO', 'GO', 'SO', 'IT', 'MY', 'HE',
                              'ME', 'IF', 'QTR', 'QTRLY', 'TIER', 'PHASE', 'BA', 'XBB',
                              # date words: "mid-2021", "Jan-22", "Spring-2022" are dates, not codes
                              'MID', 'EARLY', 'LATE', 'END', 'FALL', 'SPRING', 'SUMMER', 'WINTER', 'AUTUMN', 'JAN',
                              'JANUARY', 'FEB', 'FEBRUARY', 'MAR', 'MARCH', 'APR', 'APRIL', 'MAY', 'JUN', 'JUNE', 'JUL',
                              'JULY', 'AUG', 'AUGUST', 'SEP', 'SEPT', 'SEPTEMBER', 'OCT', 'OCTOBER', 'NOV', 'NOVEMBER',
                              'DEC', 'DECEMBER', 'YEAR', 'YEARS', 'CIRCA'})
_CONNECT = ('THE', 'OF', 'AND', '&', 'DE', 'DU', 'DES', 'LA', 'LE', 'DEL')


def camel_words(name):
    """Upper-case words of a name with camel-case parts split ('QolVane Bio, Inc.' -> QOL VANE BIO INC);
    apostrophes dropped. Only a mixed-case name has camel boundaries, so CRSP upper-case names are unchanged."""
    s = re.sub(r"['’]", '', name or '')
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)
    return [w for w in re.split(r'[^A-Za-z0-9]+', s.upper()) if w]


def _initialisms(words):
    """Initials of the first k >= 2 significant words (connectors skipped, trailing corporate-form words dropped; an
    acronym such as 'Q.R.T.' contributes all its letters)."""
    words = [w for w in words if w not in _CONNECT]
    while words and (words[-1] in FORM_WORDS or _letters(words[-1]) in FORM_WORDS):
        words = words[:-1]
    out = set()
    acc = ''
    for k, w in enumerate(words):
        acc += _letters(w) if (_ACRO.fullmatch(w) or _AMP.fullmatch(w) or _INIT.fullmatch(w)) else w[:1]
        if k >= 1 and acc.isalpha():
            out.add(acc)
    if words and _ACRO.fullmatch(words[0]) and _letters(words[0]).isalpha():
        out.add(_letters(words[0]))
    return out


def code_prefix_terms(comnams, fs_names=(), tickers=(), english=frozenset()):
    """v5 (TD3C4 D2): code prefixes of one firm (upper-case letters, >= 2) with the rules that give them:
    'ticker' (every ticker, sibling or FactSet ticker and RIC root of letters only); 'initialism' (initials of the first
    k >= 2 significant words of a CRSP or FactSet name, camel-case parts of a mixed-case FactSet name counting as words,
    a collapsed acronym contributing all its letters); 'leading' (the first 2..n letters of the first significant word
    of a CRSP name, or of a FactSet name when that word is distinctive, n = its length, when it has >= 3 letters and is
    not a generic or place word); 'table' (the private table's code_prefixes entry whose key is a whole-word prefix of
    a CRSP name). Prefixes on CODE_PREFIX_STOP are dropped. Returns {prefix: sorted rules}."""
    out = {}

    def add(p, rule):
        p = p.upper()
        if len(p) >= 2 and p.isalpha() and p not in CODE_PREFIX_STOP:
            out.setdefault(p, set()).add(rule)
    for t in tickers:
        if t and re.fullmatch(r'[A-Za-z]{2,8}', t):
            add(t, 'ticker')
    for c in comnams:
        words = normalize(c)
        if words and words[0] == 'THE':
            words = words[1:]
        for p in _initialisms(words):
            add(p, 'initialism')
        first = words[0] if words else ''
        if len(first) >= 3 and first.isalpha() and first.lower() not in GENERIC and first.lower() not in PLACE_WORDS:
            for k in range(2, len(first) + 1):
                add(first[:k], 'leading')
    for n in fs_names:
        words = camel_words(n)
        if words and words[0] == 'THE':
            words = words[1:]
        for p in _initialisms(words):
            add(p, 'initialism')
        first = words[0] if words else ''
        if len(first) >= 3 and first.isalpha() and _distinctive(first, english, 3):
            for k in range(2, len(first) + 1):
                add(first[:k], 'leading')
    for c in comnams:
        norm = ' '.join(_clean(c).split())
        for k, prefixes in CODE_PREFIXES.items():
            if _prefix_match(norm, k):
                for p in prefixes:
                    add(p, 'table')
    return {p: sorted(r) for p, r in sorted(out.items())}


def code_regex(prefixes):
    """Regex text of a firm code: one of the prefixes (any case; checked as upper or Title case by code_ok), an optional
    hyphen or dash, 1-6 digits, up to three letters, and optional further '-<digits><letters>' parts; not inside a
    longer word or number."""
    alts = '|'.join(re.escape(p) for p in sorted(prefixes, key=lambda x: (-len(x), x)))
    part = r'\d{1,' + str(CODE_MAX_DIGITS) + r'}[A-Za-z]{0,3}'
    return (r'(?<![A-Za-z0-9])(?P<pre>' + alts + r')(?P<sep>[\-–]?)' + part + r'(?:[\-–/]' + part + r')*'
            r'(?![A-Za-z0-9])(?![.,]\d)')


_PROJECT_TAIL = (r'(?:[ \t]+(?:Parent|Merger[ \t]+Sub(?:sidiary)?|Holdings?|Holdco|Bidco|Topco|Midco|Buyer|Purchaser|'
                 r'Intermediate|Acquisition(?:[ \t]+(?:Corp|Co|LLC|Inc)\.?)?))?')


def project_regex(prefixes):
    """v5 (TD3E reader read): a deal code name 'Project <prefix>' (with an optional Parent, Merger Sub, Holdings,
    Bidco, Topco, Midco, Buyer, Purchaser, Intermediate or Acquisition tail) whose prefix, in upper or Title case, is
    one of the firm's code prefixes ("Project QV Parent" for QOLVANE ... -> "[X]")."""
    alts = '|'.join(re.escape(p) for p in sorted(prefixes, key=lambda x: (-len(x), x)))
    return r'(?<![A-Za-z0-9])Project[ \t]+(?P<pre>' + alts + r')' + _PROJECT_TAIL + r'(?![A-Za-z0-9])'


def code_ok(m):
    """A matched code counts when its prefix is written in upper case or Title case and the code is not on the
    general-vocabulary stop list NON_FIRM_CODES."""
    pre = m.group('pre')
    if not (pre.isupper() or (pre[:1].isupper() and pre[1:].islower())):
        return False
    flat = re.sub(r'[\-–/]', '', m.group(0)).upper()
    stem = re.match(r'[A-Z]+\d+', flat)
    return flat not in NON_FIRM_CODES and (stem is None or stem.group(0) not in NON_FIRM_CODES)


def unit_prefixes(codes):
    """v6 (TD3C5 D1): the code prefixes (code_prefix_terms output {prefix: rules}) usable before a corporate-unit word:
    rule 'ticker' or 'initialism', >= 2 letters, not on CODE_PREFIX_STOP or ACRONYM_STOP."""
    return sorted(p for p, rules in codes.items() if len(p) >= 2 and set(rules) & {'ticker', 'initialism'}
                  and p not in CODE_PREFIX_STOP and p not in ACRONYM_STOP)


def unit_regex(prefixes):
    """v6 (TD3C5 D1): a firm prefix (ticker or initialism) followed by a corporate-unit word (UNIT_WORDS), with an
    optional possessive; not inside a longer word."""
    alts = '|'.join(re.escape(p) for p in sorted(prefixes, key=lambda x: (-len(x), x)))
    units = '|'.join(re.escape(u) for u in sorted(UNIT_WORDS, key=lambda x: (-len(x), x)))
    return (r'(?<![A-Za-z0-9])(?P<pre>' + alts + r')[ \t]+(?P<unit>' + units + r")(?:['’]s)?(?![A-Za-z0-9])")


def unit_ok(m):
    """The prefix is written in upper or Title case and the unit word is not all lower case."""
    pre, unit = m.group('pre'), m.group('unit')
    return (pre.isupper() or (pre[:1].isupper() and pre[1:].islower())) and unit != unit.lower()


def _product_acronyms(initial, product):
    """Acronyms masked when the phrase is absorbed in an article: the table's own acronym and the firm initial plus
    the initials of the phrase's alphabetic words (&/and skipped), when that has >= 3 letters."""
    out = []
    if product.get('acronym'):
        out.append(product['acronym'])
    words = [w for w in re.split(r'[\s\-/]+', product['phrase']) if w and w.lower() not in ('&', 'and', '+')]
    if words and all(w[0].isalpha() for w in words):
        acro = (initial + ''.join(w[0] for w in words)).upper()
        if len(acro) >= 3 and acro.isalpha() and acro not in ACRONYM_STOP:
            out.append(acro)
    return out


def name_variants(comnam):
    """Name variants of one CRSP comnam (see MASKING_RULE), written in the normalized form."""
    if not isinstance(comnam, str) or not comnam.strip():
        return set()
    full = normalize(comnam)
    out = {' '.join(full)}
    if len(comnam.strip()) >= TRUNCATION_LENGTH and len(comnam.split()) >= 3:
        out |= name_variants(' '.join(comnam.split()[:-1]))   # stored names are cut at ~30-32 characters
    words, stages = _strip(full)
    out |= {' '.join(s) for s in stages if s[-1] != '&'}
    for s in stages:
        out.update(_inverted(s[1:] if s[0] == 'THE' else s))
    if words and words[0] == 'THE' and len(words) > 1:
        words = words[1:]
        out.add(' '.join(words))
    if not words:
        return set()
    first = words[0]
    if len(first) >= 4 and first.isalpha() and first.lower() not in GENERIC:
        out.add(first)
    if _ACRO.fullmatch(first) and (len(_letters(first)) >= 3 or any(c.isdigit() for c in _letters(first))) and \
            _letters(first) not in ACRONYM_STOP:
        out.add(first)
    if len(words) >= 2 and words[1] not in LEGAL and ' '.join(words[:2]) not in PHRASE_STOP:
        two = ' '.join(words[:2])
        if len(_letters(two)) >= 7:
            out.add(two)
    if len(words) == 3 and words[1] == '&' and all(w.isalpha() and len(w) >= 2 for w in (words[0], words[2])):
        if words[0][0] + '&' + words[2][0] not in AMP_STOP:
            out.add(words[0][0] + '.&' + words[2][0] + '.')
    out.update(_inverted(words))
    out.update(_inverted(full[1:] if full and full[0] == 'THE' else full))

    def long_enough(v):
        letters = _letters(v).replace(' ', '')
        if _AMP.fullmatch(v):
            return len(letters) >= 2
        return len(letters) >= 3 or (len(letters) == 2 and any(c.isdigit() for c in letters)
                                     and any(c.isalpha() for c in letters))
    keep = set()
    for v in out:
        if not long_enough(v) or v.lower() in GENERIC or v in LEGAL:
            continue
        if ' ' not in v and v.lower() in PLACE_WORDS:
            continue        # v3: a place word is never a single-word variant
        if _AMP.fullmatch(v) and v.replace('.', '') in AMP_STOP:
            continue
        if _ACRO.fullmatch(v) and (len(_letters(v)) < 2 or _letters(v) in ACRONYM_STOP or
                                   (len(_letters(v)) == 2 and not any(c.isdigit() for c in _letters(v)))):
            continue        # standalone acronym: >= 3 letters (2 only with a digit: Z9), not a common acronym or month
        keep.add(v)
    return keep


def _acro_pattern(letters, possessive=True):
    parts = [re.escape(c) for c in letters]
    if possessive and len(letters) >= 3 and letters.endswith('S'):
        return _LETTER_SEP.join(parts[:-1]) + r"(?:['’]|" + _LETTER_SEP + r')S\.?'
    return _LETTER_SEP.join(parts) + r'\.?'


ACCENTS = {'A': 'AÀÁÂÃÄÅĀ', 'E': 'EÈÉÊËĒĖĘ', 'I': 'IÌÍÎÏĪ', 'O': 'OÒÓÔÕÖØŌ', 'U': 'UÙÚÛÜŪ', 'C': 'CÇ', 'N': 'NÑ',
           'S': 'SŠ', 'Z': 'ZŽ', 'Y': 'YÝŸ'}


def _letters_pattern(text):
    """Escaped text whose letters also match their accented forms (ZORVELEZ matches ZORVELĒZ)."""
    return ''.join('[' + ACCENTS[c.upper()] + ']' if c.upper() in ACCENTS else re.escape(c) for c in text)


def _plain_pattern(w, optional_s=False):
    forms = {w} | set(ABBREV.get(w, ()))
    pats = []
    for f in sorted(forms, key=len, reverse=True):
        if len(f) >= 3 and f.endswith('S') and f.isalpha():
            pats.append(_letters_pattern(f[:-1]) + (r"(?:['’]?S)?" if optional_s and len(f) >= 5 else r"['’]?S"))
        else:
            pats.append(_letters_pattern(f))
    core = pats[0] if len(pats) == 1 else '(?:' + '|'.join(pats) + ')'
    return core + (r'\.?' if w in ABBREV else '')


def _word_pattern(w, optional_s=False):
    m = _AMP.fullmatch(w)
    if m:
        return _acro_pattern(_letters(m.group(1)), False) + r'\s?&\s?' + _acro_pattern(_letters(m.group(2)), False)
    if _ACRO.fullmatch(w):
        return _acro_pattern(_letters(w))
    if _INIT.fullmatch(w):
        return re.escape(w[0]) + r'\.?'
    return _plain_pattern(w, optional_s)


def variant_regex(alias, acronym=False):
    """Regex text for one variant (normalized form, or free text for common-name aliases). In a multi-word variant a
    word ending in S also matches without the S ('ZORVATED MATERIALS' matches 'Zorvated Material')."""
    words = alias.split()
    multi = len([w for w in words if w != '&']) > 1
    out = []
    for i, w in enumerate(words):
        if w == '&':
            out.append(_AMP_JOIN)
            continue
        if out and words[i - 1] != '&':
            out.append(_JOIN)
        out.append(_word_pattern(w, multi))
    return _B + ''.join(out) + r"(?:['’]s)?" + (_E_ACRO if acronym else _E)


def _kind(alias):
    if _ACRO.fullmatch(alias):
        return 'acronym'
    if _AMP.fullmatch(alias):
        return 'amp'
    return 'name'


def owner_terms(comnams):
    """Owner rule: the first name word when it is a distinctive generic word (OWNER_GENERIC). Masked (when not all
    lower case) only in an article chosen for that firm; elsewhere it stays, since it is a common word."""
    out = set()
    for comnam in comnams:
        words, _ = _strip(normalize(comnam or ''))
        if words and words[0] == 'THE' and len(words) > 1:
            words = words[1:]
        if words and words[0].lower() in OWNER_GENERIC:
            out.add(words[0])
    return out


def loose_terms(comnam):
    """Indicator terms for the residual report (not masking rules): the first word even when generic."""
    words, _ = _strip(normalize(comnam or ''))
    if words and words[0] == 'THE':
        words = words[1:]
    return {words[0]} if words and len(words[0]) >= 4 and words[0].isalpha() else set()


_SENT_CHARS = re.compile('[-]')
DROP = ''                         # placeholder of a deleted headquarters-town word (removed with its comma)


def _upper_segment(left, right):
    """True when the three words on each side of a match are all upper case (BRIEF bodies), where capitals carry no
    ticker signal."""
    lw = re.findall(r"[A-Za-z][A-Za-z'.&\-]*", _SENT_CHARS.sub(' ', left))[-3:]
    rw = re.findall(r"[A-Za-z][A-Za-z'.&\-]*", _SENT_CHARS.sub(' ', right))[:3]
    words = [w for w in lw + rw if len(re.sub(r'[^A-Za-z]', '', w)) >= 2]
    return len(words) >= 2 and all(w == w.upper() for w in lw + rw)


def context_guard(word, m, ticker=False, english_word=False):
    """v3 (TD3BC D1c), v4 (TD3C3 N1-N3): decide whether a single-word alias occurrence may be masked. Returns 'keep'
    (leave it as written: a fixed index, price, idiom or other-entity phrase of the private table, a hyphenated verb
    compound, an English-word ticker in upper-case text), 'drop' (a headquarters town before ', <State>' or
    '<State>-based': the word is deleted, since a mask there would show that the hidden name is a town) or None
    (mask). v4: no person guard (N1: a surname after a given name and its later mentions are masked) and no place
    guard (N2: a firm word after a place word is masked; place words themselves are never single-word variants)."""
    left = m.string[max(0, m.start() - 60):m.start()]
    right = m.string[m.end():m.end() + 60]
    for lrx, rrx in FIXED_PHRASES.get(word, ()):
        if (lrx is None or re.search(lrx, left, re.I)) and (rrx is None or re.search(rrx, right, re.I)):
            return 'keep'
    if word in PARTICLE_VERBS and PARTICLE_RX.match(right):
        return 'keep'
    if re.match(r'\s*,\s*' + _STATES_RX + r'(?:-based\b|\s*(?:[,.;)]|$))', right) or \
            re.match(r'\s+' + _STATES_RX + r'-based\b', right):
        return 'drop'
    if ticker and english_word and len(word) >= 3 and _upper_segment(left, right):
        return 'keep'
    return None


def _guard_word(alias):
    """Upper-case letters of a single-word alias (acronym periods removed), or None for multi-word aliases."""
    a = alias.replace('.', '').replace('’', "'")
    return a.upper() if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9'&\-]*", a) else None


class _Rule:
    __slots__ = ('kind', 'alias', 'index', 'regex', 'group', 'word', 'english')

    def __init__(self, kind, alias, index, regex, group=0, english=frozenset()):
        self.kind, self.alias, self.index, self.regex, self.group = kind, alias, index, regex, group
        self.word = _guard_word(alias) if kind not in ('ric', 'ticker_ric_form', 'ticker_exchange', 'ticker_paren',
                                                       'ticker_cashtag', 'amp') else None
        self.english = bool(self.word) and (self.word.lower() in english or
                                            (self.word.endswith('S') and self.word[:-1].lower() in english))

    def accepted(self, m):
        """True (mask), False (leave) or 'drop' (delete a headquarters-town word)."""
        text = m.group(0)
        if self.kind in ('name', 'near_name'):
            if text == text.lower() and all(w.lower() in GENERIC for w in self.alias.split()):
                return False     # a phrase of generic words ("air products") counts only when not all lower case
            ok = (' ' in self.alias and len(self.alias.replace('.', '')) >= 6) or text != text.lower()
            if self.kind == 'near_name' and text == text.lower():
                ok = False       # v5: a near name counts only when not all lower case
        elif self.kind in ('acronym', 'amp'):
            ok = text != text.lower()
        elif self.kind == 'common':
            ok = len(self.alias.replace(' ', '')) <= 3 or text != text.lower()   # short aliases: exact-case regex
        else:
            ok = True
        if not ok:
            return False
        if self.word is not None:
            g = context_guard(self.word, m, ticker=self.kind.startswith('ticker'), english_word=self.english)
            if g == 'keep':
                return False
            if g == 'drop':
                return 'drop'
        return True


_COMPILED = {}


def _compile(pattern, flags=0):
    """Memoized re.compile (every article builds a Masker; the stdlib cache holds only 512 patterns)."""
    key = (pattern, flags)
    rx = _COMPILED.get(key)
    if rx is None:
        rx = _COMPILED[key] = re.compile(pattern, flags)
    return rx


def _name_rule(alias, index, english=frozenset()):
    kind = _kind(alias)
    return _Rule(kind, alias, index, _compile(variant_regex(alias, kind == 'acronym'), re.I), english=english)


def _common_rule(alias, index, english=frozenset()):
    if len(alias.replace(' ', '')) <= 3:
        regex = _compile(_B + re.escape(alias) + r"(?:['’]s)?" + _E)
    else:
        words = alias.split()
        regex = _compile(_B + _JOIN.join(re.escape(w) for w in words) + r"(?:['’]s)?" + _E, re.I)
    return _Rule('common', alias, index, regex, english=english)


_SUFFIXES = (('ies', 'y'), ('es', ''), ('s', ''), ('ed', ''), ('ed', 'e'), ('ing', ''), ('ing', 'e'), ('ers', ''),
             ('er', ''), ('ly', ''))


def _is_english(word, english):
    """English word or a regular inflection of one (plural, past, gerund, agent noun, adverb)."""
    w = word.lower()
    if w in english:
        return True
    return any(w.endswith(a) and len(w) - len(a) >= 3 and w[:-len(a)] + b in english for a, b in _SUFFIXES)


def _ticker_rules(ticker, index, english=frozenset(), extra=False):
    """extra (v3): a ticker of an issuer sibling or a FactSet-only ticker. It is always masked in RIC form, after an
    exchange prefix, in parentheses and as a cashtag, but standalone or in Title case only when it has >= 3 letters
    and is not an English word or a common acronym."""
    t = re.escape(ticker)
    ric_form = (r'\.(?:N|O|OQ|A|P|PK|TO|L)(?![A-Za-z0-9.])' if len(ticker) == 1 else r'\.[A-Za-z]{1,3}' + _E)
    rules = [_Rule('ticker_ric_form', ticker, index, _compile(_B + t + ric_form)),   # 'U' never matches 'U.S.'
             _Rule('ticker_exchange', ticker, index, _compile(r'(' + EXCHANGES + r'\s*:\s*)' + t + _E), 1),
             _Rule('ticker_paren', ticker, index, _compile(r'\(\s*' + t + r'\s*\)')),
             _Rule('ticker_cashtag', ticker, index, _compile(r'\$' + t + _E))]
    if extra and not (len(ticker) >= 3 and ticker.isalpha() and not _is_english(ticker, english)):
        return rules
    if len(ticker) >= 2 and ticker.isalnum() and ticker.isupper() and ticker not in ACRONYM_STOP:
        rules.append(_Rule('ticker_standalone', ticker, index, _compile(r'(?<![A-Za-z0-9.$])' + t + _E),
                           english=english))
    title = ticker[:1] + ticker[1:].lower()
    if len(ticker) >= 3 and ticker.isalpha() and ticker.isupper() and ticker not in ACRONYM_STOP and \
            ticker.lower() not in english:
        # v2: title-case ticker (news headlines title-case every word: "Quorn And Zbxq", "(Zbxq)")
        rules.append(_Rule('ticker_title', title, index, _compile(r'(?<![A-Za-z0-9.$])' + re.escape(title) + _E),
                           english=english))
    return rules


def lev1(a, b):
    """True when Levenshtein distance(a, b) == 1 (one substitution, insertion or deletion)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _name_words(aliases):
    """Upper-case words of a firm's name variants and common aliases (acronym forms excluded)."""
    out = set()
    for v in list(aliases.get('names', ())) + list(aliases.get('common', ())):
        for w in re.split(r"[\s\-/]+", v.upper()):
            w = re.sub(r"['’]", '', w)
            if _ACRO.fullmatch(w) or _AMP.fullmatch(w) or _INIT.fullmatch(w):
                continue
            w = w.strip('.,&')
            if w.isalpha():
                out.add(w)
    return out


def _distinctive(w, english, min_len):
    lw = w.lower()
    return (len(w) >= min_len and w.isalpha() and not _is_english(w, english) and lw not in GENERIC and
            lw not in PLACE_WORDS and w not in LEGAL and w not in FORM_WORDS and w not in ABBREV and
            w not in ACRONYM_STOP and w.upper() not in PERSON_GIVEN)


def embedded_terms(aliases, english=frozenset()):
    """v3 (TD3BC D1 item 2): terms looked for inside longer tokens: tickers and acronyms of >= 3 letters that are not
    English words or common acronyms ('acro'), and distinctive name words of >= 4 letters ('word')."""
    acro = set()
    for t in list(aliases.get('tickers', ())) + list(aliases.get('tickers_extra', ())):
        if len(t) >= 3 and t.isalpha() and t.isupper() and t not in ACRONYM_STOP and not _is_english(t, english):
            acro.add(t)
    for v in aliases.get('names', ()):
        if _ACRO.fullmatch(v):
            letters = _letters(v)
            if len(letters) >= 3 and letters.isalpha() and letters not in ACRONYM_STOP and \
                    not _is_english(letters, english):
                acro.add(letters)
    words = {w for w in _name_words(aliases) if _distinctive(w, english, 4)}
    return sorted([(t, 'acro') for t in acro] + [(w, 'word') for w in words - acro])


def fuzzy_words(aliases, english=frozenset()):
    """v3 (TD3BC D1 item 3): distinctive name words of >= 6 letters checked for edit-distance-1 misspellings."""
    return sorted(w for w in _name_words(aliases) if _distinctive(w, english, 6))


_ENGLISH_REF = [frozenset()]      # set by Masker (English list)


def embedded_hit(token, term, kind):
    """True when the term sits inside the longer token as a separable part: an upper-case acronym at the token start
    or end or after a lower-to-upper boundary (ZetaQRX, ZETAQRX, QRXP); a name word at the token start followed by a
    camel-case boundary, a digit or at most two more letters (Zelwickon), or after a lower-to-upper boundary."""
    n, k = len(token), len(term)
    if k >= n:
        return False
    low, lt = token.lower(), term.lower()
    i = low.find(lt)
    while i >= 0:
        j = i + k
        seg = token[i:j]
        camel_start = i > 0 and token[i - 1].islower() and token[i].isupper()
        camel_end = j < n and token[j - 1].islower() and token[j].isupper()
        if kind == 'acro':
            rest = token[:i] + token[j:]
            if seg.isupper() and token.isupper():
                # upper-case token: the acronym starts it with at most two more letters (QRXP, QRXS) or with digits
                # only (v5: QRX1181), or the rest is itself an English word (ZETAQRX = ZETA + QRX)
                if (i == 0 and (len(rest) <= 2 or rest.isdigit())) or \
                        ((i == 0 or j == n) and len(rest) >= 3 and rest.isalpha() and rest.lower() in _ENGLISH_REF[0]):
                    return True
            elif seg.isupper() and (camel_start or (i == 0 and (token[j:j + 1].isdigit() or
                                                                 re.match(r'[A-Z][a-z]', token[j:j + 2])))):
                return True
            elif i == 0 and seg[:1].isupper() and seg[1:].islower() and len(rest) >= 3 and rest.isalpha() and \
                    rest.lower() in _ENGLISH_REF[0]:
                return True          # Title-case product written from the acronym and an English word (Qrxpro)
        else:
            rest = token[j:]
            if i == 0 and (camel_end or rest[:1].isdigit() or (rest.isalpha() and len(rest) <= 2)):
                return True
            # v5 (TD3C4 R1): a name word of >= 5 letters starting a longer token followed by more letters in the same
            # case style (Zorbexpro, ZORBEXPRO): up to four letters, or more letters that are not an English word
            # (a firm word plus an English word, such as a compound noun, is not a product name of the firm)
            if i == 0 and k >= 5 and rest.isalpha() and \
                    (len(rest) <= 4 or not _is_english(rest, _ENGLISH_REF[0])) and \
                    ((seg[:1].isupper() and seg[1:].islower() and rest.islower()) or (seg.isupper() and rest.isupper())):
                return True
            if camel_start and (j == n or camel_end or token[j].isdigit()):
                return True
        i = low.find(lt, i + 1)
    return False


def absorb_words(aliases):
    """Name words of a firm for the trailing-word rule: words of every name variant and common alias (acronyms as
    letters, abbreviations with their expansions), >= 2 characters, not on ABSORB_STOP."""
    words = set()
    for v in list(aliases.get('names', ())) + list(aliases.get('common', ())):
        for w in v.upper().split():
            base = w.replace('.', '').replace('&', '') if (_ACRO.fullmatch(w) or _AMP.fullmatch(w)) else w
            base = re.sub(r"['’]", '', base)
            for f in {base} | set(ABBREV.get(base, ())):
                for part in f.split():
                    if len(part) >= 2 and part.isalnum() and part not in ABSORB_STOP:
                        words.add(part)
    return words


def _ric_rule(ric, index):
    return _Rule('ric', ric, index, _compile(_B + re.escape(ric) + _E))


_TLD = r'(?:com|net|org|gov|edu|io|co|ly|bg|gl|me|tv|info|biz|us|uk|ca|de|fr|jp|cn|hk|ai|app|news|rs|to|fm|sh|be|it)'
STORY_FIELDS = re.compile(r'STORY_(?:NUMBER|DATE|TIME)\s*:\s*\[[^\]\n]*\]')
# v5 (TD3C4 N-b): a bracketed story ID may also start with "P" (the press-release IDs after "Source text for Eikon:");
# it is removed like the other bracketed story IDs (the label stays, as for every other removed ID).
STORY_IDS = re.compile(r'\[(?:ID:\s*)?P?n(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,14}(?:\]|(?=\s*(?:\.\.\.)?\s*$))'
                       r'|\[ID:[^\]\n]*$|\[P?n(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{1,13}(?=\s*\.\.\.\s*$)')
# v5: a run of three or more unbracketed news-agency alert numbers (three letters, three digits, two letters or digits,
# as in a list of retracted alerts) is removed like the bracketed story IDs.
ALERT_ID_RUN = re.compile(r'(?<![A-Za-z0-9])(?:[A-Z]{3}\d{3}[A-Z0-9]{2}(?:[ \t,;]+|(?=\s*(?:\.\.\.)?\s*$))){3,}')
# v3 (TD3BC D2): any label.tld/path with a 2-6 letter lower-case TLD (shorturl.at/...), plus the frozen TLD list in any
# case (v2 behaviour).
BARE_URL = re.compile(r'(?<![\w@./])(?:[A-Za-z0-9-]+\.)+(?:[a-z]{2,6}|(?i:' + _TLD[3:-1] + r'))/[^\s)\]]*')
# Independent residual URL pattern for verify (does not reuse the masker's TLD list).
URL_RESIDUAL = re.compile(r'(?:https?://|www\.)|(?<![\w@])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[a-z]{2,6}/\S')
CONTACT = re.compile(r'\(\((?:[^()]|\([^()]*\))*(?:\)\)|$)')          # nested contact blocks ((... (0) ...; ...))
_NAME_WORD = r"(?:[A-Z][a-z][\w'’\-]*|[A-Z]\.)"
BYLINE = re.compile(r"\bBy " + _NAME_WORD + r"(?: " + _NAME_WORD + r"){0,3}(?:(?:, | and )" + _NAME_WORD +
                    r"(?: " + _NAME_WORD + r"){0,3}){0,5}\s+(?=(?:[A-Z][^\n]{0,60}?,?\s+)?" + MONTHS +
                    r"\s+\d{1,2}\s*\(Reuters)")


def scrub_boilerplate(text):
    text = re.sub(r'(?:https?://|www\.)\S+', '[URL]', text, flags=re.I)
    text = re.sub(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+', '[EMAIL]', text)
    text = BARE_URL.sub('[URL]', text)
    text = STORY_FIELDS.sub(' ', text)
    text = re.sub(r'Further company coverage:.*?(?=\(\(|$)', ' ', text, flags=re.S)
    text = re.sub(r'Source text for Eikon:\s*\[ID:[^\]]*\]', ' ', text)
    text = STORY_IDS.sub(' ', text)
    text = ALERT_ID_RUN.sub(' ', text)                    # v5
    text = re.sub(r'\[ID:[^\]]*\]', ' ', text)
    text = CONTACT.sub(' ', text)
    text = re.sub(r'\((?:Reporting|Editing|Additional reporting|Writing|Compiled|Research) by[^)]*\)', ' ', text,
                  flags=re.I)
    text = BYLINE.sub('', text)            # v3: "By <reporters> <DATELINE>, <date> (Reuters)" bylines
    text = re.sub(r'^[A-Z][A-Za-z .,/\'-]{2,40},?\s+' + MONTHS + r'\s+\d{1,2}\s*\(Reuters\)\s*-\s*', '', text)
    text = re.sub(r'\(Reuters(?: Breakingviews)?\)\s*-?\s*', '', text)
    return text


AGENCY = re.compile(_B + r'(?:Thomson\s+)?Reuters' + _E, re.I)


def mask_agency(text):
    """The news agency name (also with its owner's first word, when that is not a masked firm: the name rules run first) becomes
    [NEWS_AGENCY] (not all-lower-case occurrences)."""
    return AGENCY.sub(lambda m: '[NEWS_AGENCY]' if m.group(0) != m.group(0).lower() else m.group(0), text)


_UNITS = r'(?!\s*(?:%|percent|pct|cents?|million|billion|mln|bln|bps|basis|per\b))'
# v4 (E2): frozen non-month stop list for the month-typo rule (general vocabulary of numbered items).
NON_MONTH_STOP = ('no', 'nos', 'number', 'form', 'forms', 'item', 'items', 'rule', 'rules', 'section', 'sections',
                  'article', 'articles', 'chapter', 'part', 'parts', 'phase', 'phases', 'stage', 'stages', 'model',
                  'models', 'series', 'class', 'tier', 'level', 'grade', 'version', 'release', 'gen', 'generation',
                  'unit', 'units', 'train', 'trains', 'line', 'lines', 'block', 'blocks', 'lot', 'lots', 'page',
                  'pages', 'note', 'notes', 'exhibit', 'annex', 'appendix', 'schedule', 'volume', 'vol', 'issue',
                  'route', 'highway', 'terminal', 'reactor', 'plant', 'covid', 'top', 'type', 'group', 'team',
                  'game', 'games', 'round', 'episode', 'act', 'title', 'proposition',
                  'measure', 'amendment', 'resolution', 'case', 'docket', 'count', 'counts')
DATE_PATTERNS = (
    ('iso_date', re.compile(r'(?<!\d)(?:19|20)\d\d[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])(?!\d)'),
     '[DATE]'),
    ('numeric_date', re.compile(r'(?<![\d/.])\d{1,2}/\d{1,2}/(?:(?:19|20)\d\d|\d\d)(?![\d/])'), '[DATE]'),
    ('dotted_date', re.compile(r'(?<![\d.])\d{1,2}\.\d{1,2}\.(?:19|20)\d\d(?!\d)'), '[DATE]'),
    # v5 (TD3C4 N-a): ordinal day before or after a month ("the 30th of September 2021", "30th September",
    # "September the 30th") becomes [DATE]
    ('ordinal_day_month', re.compile(r'\b\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?' + MONTHS +
                                     r'(?:,?\s+(?:19|20)\d\d)?\b', re.I), '[DATE]'),
    ('month_the_ordinal', re.compile(r'\b' + MONTHS + r'\s+the\s+\d{1,2}(?:st|nd|rd|th)\b(?:,?\s+(?:19|20)\d\d\b)?',
                                     re.I), '[DATE]'),
    ('month_day', re.compile(r'\b' + MONTHS + r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+(?:19|20)\d\d)?\b'), '[DATE]'),
    ('day_month', re.compile(r'\b\d{1,2}\s+' + MONTHS + r'(?:\s+(?:19|20)\d\d)?\b'), '[DATE]'),
    ('month_apostrophe_year', re.compile(r'\b' + MONTHS + r"\s?['’]\d{2}\b"), '[DATE]'),
    ('season_apostrophe_year', re.compile(r"\b(?:Spring|Summer|Fall|Autumn|Winter|CY)\s?['’]\d{2}\b", re.I), '[DATE]'),
    # v3 (TD3BC D2): a year glued to a clock time ("<year>12:32 p.m.")
    ('year_time', re.compile(r'(?<![\d.,])(?:19|20)\d\d(?=\d{1,2}:\d\d)'), '[YEAR] '),
    ('year4', re.compile(r'(?<![\d.,])(?:19|20)\d\d(?!\d)'), '[YEAR]'),
    ('fiscal_year2', re.compile(r"\bFY\s?['’]?\d{2}\b"), 'FY [YEAR]'),
    ('quarter_sep_year2', re.compile(r"\b([QH][1-4]|[1-4][QH])\s?(?:['’]|FY\s?|-)\s?\d{2}\b"), r'\1 [YEAR]'),
    ('quarter_compact_year2', re.compile(r'\b([1-4][QH])(\d{2})\b'), r'\1 [YEAR]'),
    ('quarter_space_year2', re.compile(r'\b([QH][1-4])\s?\d{2}\b(?![\d.,%])' + _UNITS), r'\1 [YEAR]'),
    # v5 (TD3E reader read): a year cut to three digits by the source after a masked date ("[DATE] 202" at the end of
    # a text or before punctuation) becomes [YEAR]
    ('year_fragment', re.compile(r'(?<=\[DATE\])(,?[ \t]+)(?:19|20)\d(?!\d)(?=[ \t]*(?:$|[.;,)\]]))'), r'\1[YEAR]'),
    # v4 (TD3C3 E2, recommended remnant rule): a day range left after a masked date ("[DATE]-20, [YEAR]",
    # "[DATE] And 31, [YEAR]") becomes [DATE]. The audit's pattern, matched case-insensitively, with an en dash added,
    # and not before a decimal or thousands digit (a negative earnings figure after a date: "[DATE] -0.79").
    ('day_range', re.compile(r'\[DATE\]\s*(?:-|–|to|and|through)\s*\d{1,2}\b(?![.,:]\d)', re.I), '[DATE]'),
    # v4 (TD3C3 E2): a source typo or unlisted spelling of a month before "<day>, [YEAR]" ("Oco 1, [YEAR]") becomes
    # [DATE]; [YEAR] is kept. The audit's pattern; the year part is a lookahead so that it stays in place. Words on the
    # frozen non-month stop list (NON_MONTH_STOP: numbered items such as "Phase 3, [YEAR]") are not dates.
    ('month_typo', re.compile(r'\b(?!(?i:' + '|'.join(NON_MONTH_STOP) + r')\b)[A-Za-z]{1,9}\.?\s+\d{1,2}'
                              r'(?:st|nd|rd|th)?(?=,\s*\[YEAR\])'), '[DATE]'),
)


def mask_dates(text):
    for _, rx, repl in DATE_PATTERNS:
        text = rx.sub(repl, text)
    return text


def date_residuals(text):
    """Names of date patterns (and story boilerplate, URLs) still present in final text (verify check, C2/D2)."""
    hits = [name for name, rx, _ in DATE_PATTERNS if rx.search(text)]
    if re.search(r"['’]\d{2}\b", text) and re.search(r"(?:\b" + MONTHS + r"|\b[QH][1-4]|\bFY)\s?['’]\d{2}\b", text):
        hits.append('apostrophe_year2')
    if STORY_FIELDS.search(text) or re.search(r'STORY_(?:NUMBER|DATE|TIME)', text):
        hits.append('story_fields')
    if STORY_IDS.search(text) or ALERT_ID_RUN.search(text):
        hits.append('story_ids')
    if BARE_URL.search(text) or URL_RESIDUAL.search(text):
        hits.append('url')
    return hits


def truncate(text, n):
    """First n characters; a partial trailing word (cut mid-word) is dropped and '...' appended."""
    text = text or ''
    if len(text) <= n:
        return text
    cut = text[:n]
    if text[n].isalnum() and cut[-1:].isalnum():
        m = re.search(r'[^\W_]+$', cut)
        cut = cut[:m.start()] if m else cut
    return cut.rstrip() + ' ...'


_WORD_TOKEN = re.compile(r"(?<![A-Za-z0-9-])([A-Za-zÀ-ÖØ-öø-ž][A-Za-z0-9À-ÖØ-öø-ž]*)((?:['’][sS])?)"
                         r"(?![A-Za-z0-9-])")
_RIC_AFTER = re.compile(r'([-])(\s*)(<\s*[A-Za-z0-9]{1,10}(?:[._][A-Za-z]{1,3})?\s*>|'
                        r'\(\s*[A-Za-z0-9]{1,10}[._][A-Za-z]{1,3}\s*\))')


_SIGNS = r'(?:[ \t]*[®™])*'            # optional registered / trademark signs
_PHRASE_SEP = r'(?:[ \t]+|[ \t]*[\-–][ \t]*)'
# A product phrase marked 'model' also absorbs a following model designation (up to three parts: a number of at most
# four digits with an optional letter, SE, XR, XS, X, Series <n>, Pro, Max, Plus, mini, Ultra, Air), so that no model
# number of the product is left next to the token ("[X] Zorbphone 14 Pro Max" -> "[X]").
_MODEL_SUFFIX = (r'(?:[ \t]+(?:Series[ \t]+\d{1,2}|\d{1,4}[A-Za-z]?|SE|XR|XS|X|Pro|Max|Plus|mini|Mini|Ultra|Air)'
                 r'(?![A-Za-z0-9])){0,3}' + _SIGNS)


def _phrase_pattern(phrase, capitalized=False):
    """Regex text of a table phrase: words joined by spaces or hyphens; '&', 'and' and '+' are interchangeable; an
    optional registered or trademark sign after each word. capitalized: only Title-case or upper-case spellings."""
    words = phrase.split()
    out = []
    for i, w in enumerate(words):
        if w.lower() in ('&', 'and', '+'):
            out.append(r'[ \t]*(?:&|and|\+)[ \t]*')
            continue
        if out and words[i - 1].lower() not in ('&', 'and', '+'):
            out.append(_PHRASE_SEP)
        if capitalized:
            forms = sorted({w[:1].upper() + w[1:], w.upper()}, key=len, reverse=True)
            out.append('(?-i:(?:' + '|'.join(re.escape(f) for f in forms) + '))')
        else:
            out.append(re.escape(w))
        out.append(_SIGNS)
    return ''.join(out)


def is_street_phrase(phrase):
    """v5 (TD3C4 D1): a table phrase whose last word is a street word ('Way', 'Road', ...)."""
    words = re.findall(r'[A-Za-z]+', phrase or '')
    return bool(words) and words[-1].upper() in STREET_WORDS


_HOUSE = r'(?:(?P<house>(?<![A-Za-z0-9.,$#\-])\d{1,6}[A-Za-z]?)[ \t]+)?'


def _product_regex(anchor, phrases):
    """E1: anchor (a sentinel or a rendered token) + optional possessive and signs + one of the firm's phrases (longest
    first; named groups p0, p1, ...; a phrase's not_before lookahead). v5 (TD3C4 D1): an optional house number of 1-6
    digits right before the anchor (group 'house'); it is absorbed only with a street phrase (is_street_phrase)."""
    order = sorted(range(len(phrases)), key=lambda i: (-len(phrases[i]['phrase']), phrases[i]['phrase']))
    alts = []
    for i in order:
        p = phrases[i]
        alts.append(f'(?P<p{i}>' + _phrase_pattern(p['phrase'], p.get('capitalized')) +
                    (_MODEL_SUFFIX if p.get('model') else '') + ')' +
                    (f"(?!{p['not_before']})" if p.get('not_before') else ''))
    return _compile(_HOUSE + anchor + r"(?P<poss>['’]s)?" + _SIGNS + _PHRASE_SEP + r'(?:[®™][ \t]*)*(?:' +
                    '|'.join(alts) + r')(?![A-Za-z0-9])', re.I)


def _prefix_regex(anchor, prefixes):
    """E1: one of the firm's preceding phrases + space or hyphen + anchor."""
    if not prefixes:
        return None
    alts = '|'.join(_phrase_pattern(p) for p in sorted(prefixes, key=lambda x: (-len(x), x)))
    return _compile(r'(?<![A-Za-z0-9])(?P<p>' + alts + ')' + _PHRASE_SEP + anchor, re.I)


# Connector between a token and its joint-product partner: a hyphen, dash, slash or plus; '&'; 'and' (optionally
# 'and its/their/<nationality> partner'); or a plain space ("Quillco Zorbtech vaccine").
_JOINT_CONNECTOR = (r'(?:[ \t]*[\-–/+][ \t]*|[ \t]*&[ \t]*|[ \t]+and[ \t]+(?:(?:its|their|German|U\.S\.|US|'
                    r'American|French|British|Swiss|Japanese|Chinese)[ \t]+)*(?:partners?[ \t]+)?|[ \t]+)')
_PARTNER_SUFFIX = r'(?:[ \t]+(?:SE|Inc\.?|Corp\.?|AG|plc|PLC|Co\.?|Ltd\.?|N\.?V\.?))?'


def _joint_regexes(anchor, joints):
    """E1: [(partner-before-token regex, token-before-partner regex)] for each joint-product partner."""
    out = []
    for j in joints:
        partner = '(?P<partner>' + j['partner'] + ')'
        conn = j.get('connector') or _JOINT_CONNECTOR
        before = _compile(r'(?<![A-Za-z0-9])' + partner + _PARTNER_SUFFIX + r"(?:['’]s)?" + conn + anchor, re.I)
        after = _compile(anchor + r"(?P<poss>['’]s)?" + conn + partner + _PARTNER_SUFFIX + r"(?P<pp>['’]s)?"
                         r'(?![A-Za-z0-9])', re.I)
        out.append((before, after))
    return out


class Masker:
    """entries: [(token, {'names', 'common', 'tickers', 'tickers_extra', 'rics', 'embedded', 'fuzzy', 'products'})] in
    precedence order ('embedded': [(term, 'acro'|'word')], 'fuzzy': [word], 'products': firm_products() output; all
    optional). v4: mask_article() masks a headline and body together (E1 acronyms)."""

    def __init__(self, entries, english=frozenset()):
        self.tokens = []
        self.collisions = []
        self.absorb = []
        self.embedded = []
        self.fuzzy = []
        self.english = english
        _ENGLISH_REF[0] = english
        self.log = []
        self.products = []          # v4 (E1): per index (product absorb regex, phrase list) or None
        self.prefix_rx = []         # v4 (E1): per index preceding-phrase regex or None
        self.joint_rx = []          # v4 (E1): per index [(before regex, after regex)]
        self.active = {}            # v4 (E1): index -> acronyms masked in the current article
        self.prod_meta = []         # v4 (E1): per index firm_products() output
        self.code_rx = []           # v5 (D2): per index firm-code regex or None
        self.project_rx = []        # v5: per index deal code-name ('Project <prefix>') regex or None
        self.unit_rx = []           # v6 (TD3C5 D1): per index initialism-unit regex or None
        self.code_prefixes = []     # v5 (D2): per index sorted code prefixes
        owner = {}
        rules = []
        for index, (token, aliases) in enumerate(entries):
            if not _TOKEN.fullmatch(token):
                raise ValueError('Invalid role token')
            self.tokens.append(token)
            words = sorted(absorb_words(aliases), key=lambda w: (-len(w), w))
            self.absorb.append(_compile(re.escape(self._sentinel(index)) + r"((?:(?:[ \t]*,[ \t]*|[ \t]+|[ \t]*&[ \t]*)"
                                        r'(?:' + '|'.join(re.escape(w) for w in words) + r')\.?(?![A-Za-z0-9\-]))+)'
                                        r"(?:['’]s(?![A-Za-z0-9]))?", re.I) if words else None)
            self.embedded.append([(t.upper(), k) for t, k in aliases.get('embedded', ())])
            self.fuzzy.append([w.upper() for w in aliases.get('fuzzy', ())])
            prod = aliases.get('products') or {}
            self.prod_meta.append(prod)
            phrases = list(prod.get('products', ()))
            self.products.append((_product_regex(re.escape(self._sentinel(index)), phrases), phrases)
                                 if phrases else None)
            self.prefix_rx.append(_prefix_regex(re.escape(self._sentinel(index)), prod.get('prefixes', ())))
            self.joint_rx.append(_joint_regexes(re.escape(self._sentinel(index)), prod.get('joint_partners', ())))
            cp = sorted({p.upper() for p in aliases.get('code_prefixes', ()) if p})
            self.code_prefixes.append(cp)
            self.code_rx.append(_compile(code_regex(cp), re.I) if cp else None)
            self.project_rx.append(_compile(project_regex(cp), re.I) if cp else None)
            up = sorted({p.upper() for p in aliases.get('unit_prefixes', ()) if p})
            self.unit_rx.append(_compile(unit_regex(up), re.I) if up else None)
            for kind, values in (('name', aliases.get('names', ())), ('common', aliases.get('common', ())),
                                 ('near', aliases.get('near', ())),      # v5 (D2): near names, after own names
                                 ('ticker', aliases.get('tickers', ())),
                                 ('ticker_extra', aliases.get('tickers_extra', ())), ('ric', aliases.get('rics', ()))):
                for v in sorted(set(values)):
                    if not isinstance(v, str) or not v.strip():
                        continue
                    key = ('ticker' if kind.startswith('ticker') else kind if kind == 'ric' else 'name',
                           v if kind.startswith('ticker') or kind == 'ric' else v.lower())
                    if key in owner:
                        if owner[key] != index:
                            self.collisions.append({'kind': kind, 'alias': v, 'kept': self.tokens[owner[key]],
                                                    'dropped': token})
                        continue
                    owner[key] = index
                    if kind == 'name':
                        rules.append(_name_rule(v, index, english))
                    elif kind == 'near':
                        rules.append(_Rule('near_name', v, index, _compile(variant_regex(v), re.I), english=english))
                    elif kind == 'common':
                        rules.append(_common_rule(v, index, english))
                    elif kind == 'ticker':
                        rules.extend(_ticker_rules(v, index, english))
                    elif kind == 'ticker_extra':
                        rules.extend(_ticker_rules(v, index, english, extra=True))
                    else:
                        rules.append(_ric_rule(v, index))
        order = {'ric': 0, 'ticker_ric_form': 1, 'ticker_exchange': 2, 'ticker_paren': 3, 'ticker_cashtag': 4,
                 'name': 5, 'common': 5, 'amp': 5, 'near_name': 5, 'acronym': 6, 'ticker_standalone': 7,
                 'ticker_title': 8}
        # longest first; an acronym or compact ampersand counts by its letters ('Q.X.Z.' = 3), so a full RIC such as
        # 'QXZ.N' is masked before the bare acronym 'QXZ'
        self.rules = sorted(rules, key=lambda r: (-(len(_letters(r.alias)) if r.kind in ('acronym', 'amp')
                                                    else len(r.alias)), order[r.kind], r.alias, r.index))

    @staticmethod
    def _sentinel(index):
        return _SENT_OPEN + chr(0xE100 + index) + _SENT_CLOSE

    def _render(self, s):
        for index, token in enumerate(self.tokens):
            s = s.replace(self._sentinel(index), token)
        return s.replace(DROP, '')

    def _note(self, index, kind, alias, m, start=None, end=None, **extra):
        """Replacement log entry (private build metadata; used by the adjacency review)."""
        a = m.start() if start is None else start
        b = m.end() if end is None else end
        self.log.append({'token': self.tokens[index], 'kind': kind, 'alias': alias,
                         'match': self._render(m.string[a:b]),
                         'left': self._render(m.string[max(0, a - 60):a]),
                         'right': self._render(m.string[b:b + 60]), **extra})

    def _apply(self, rule, text):
        count = 0
        sentinel = self._sentinel(rule.index)

        def repl(m):
            nonlocal count
            verdict = rule.accepted(m)
            if not verdict:
                return m.group(0)
            if verdict == 'drop':
                self._note(rule.index, 'dropped_hq_town', rule.alias, m)
                return DROP
            count += 1
            if rule.group:
                self._note(rule.index, rule.kind, rule.alias, m, start=m.end(1))
                return m.group(1) + sentinel
            self._note(rule.index, rule.kind, rule.alias, m)
            return sentinel
        return rule.regex.sub(repl, text), count

    def _token_hit(self, word):
        """(index, kind, term) of the first firm whose embedded term or misspelled name word matches this token."""
        if len(word) < 4 or word == word.lower() or _is_english(word, self.english):
            return None
        up = word.upper()
        for index in range(len(self.tokens)):
            for term, k in self.embedded[index]:
                if term in up and embedded_hit(word, term, k):
                    return index, 'embedded', term
            if word[0].isupper() and word.isalpha():
                for w in self.fuzzy[index]:
                    # a 6-letter word counts only an inserted or deleted letter; >= 7 letters also a substitution
                    if up[0] == w[0] and abs(len(up) - len(w)) <= 1 and (len(w) >= 7 or len(up) != len(w)) and \
                            lev1(up, w):
                        return index, 'fuzzy_ed1', w
        return None

    def _token_pass(self, text, counts):
        """v3: names embedded in longer tokens and edit-distance-1 misspellings (the whole token is masked)."""
        if not any(self.embedded) and not any(self.fuzzy):
            return text

        def repl(m):
            hit = self._token_hit(m.group(1))
            if hit is None:
                return m.group(0)
            index, kind, term = hit
            if context_guard(term, m) == 'keep':
                return m.group(0)
            counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
            self._note(index, kind, term, m)
            return self._sentinel(index)
        return _WORD_TOKEN.sub(repl, text)

    def _code_pass(self, text, counts):
        """v5 (TD3C4 D2): a product or drug code whose prefix is one of the firm's code prefixes is replaced as a whole
        by the firm's token ("ZRX-09", "Qlvn1181", "QX-6137" -> "[X]"); firms in precedence order."""
        for index, rx in enumerate(self.code_rx):
            if rx is None:
                continue

            def rc(m, index=index):
                if not code_ok(m):
                    return m.group(0)
                counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                self._note(index, 'firm_code', m.group('pre').upper(), m)
                return self._sentinel(index)

            def rp(m, index=index):
                pre = m.group('pre')
                if not (pre.isupper() or (pre[:1].isupper() and pre[1:].islower())):
                    return m.group(0)
                counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                self._note(index, 'firm_project_name', pre.upper(), m)
                return self._sentinel(index)
            text = self.project_rx[index].sub(rp, rx.sub(rc, text))
        return text

    _GLUED = re.compile(r'(?<![^\W_])([^\W_]*)([-])([^\W_]*)(?![^\W_])')

    def _unit_pass(self, text, counts):
        """v6 (TD3C5 D1): a readable firm ticker or initialism followed by a corporate-unit word ("ZB Ventures") is
        replaced as a whole by the firm's token; runs after the name rules, so it acts only where the prefix is still
        readable."""
        for index, rx in enumerate(self.unit_rx):
            if rx is None:
                continue

            def ru(m, index=index):
                if not unit_ok(m):
                    return m.group(0)
                counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                self._note(index, 'initialism_unit', m.group('pre').upper() + ' ' + m.group('unit'), m)
                return self._sentinel(index)
            text = rx.sub(ru, text)
        return text

    def _glued_pass(self, text, counts):
        """v5 (TD3C4 D1): a token glued to letters or digits (no separator) takes the whole alphanumeric word, digits
        included ("QRT515" with "QRT" masked -> "[X]", never "[X]515")."""
        def rg(m):
            if not m.group(1) and not m.group(3):
                return m.group(0)
            index = ord(m.group(2)[1]) - 0xE100
            counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
            self._note(index, 'glued_word', m.group(1) + '<token>' + m.group(3), m)
            return m.group(2)
        return self._GLUED.sub(rg, text)

    def _ric_after_token(self, text, counts):
        """v3: a RIC written right after a role token ("[X] <QRT.L>", "[X] (1234.T)") takes that token."""
        def repl(m):
            index = ord(m.group(1)[1]) - 0xE100
            counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
            self._note(index, 'ric_after_token', m.group(3), m, start=m.start(3))
            inner = m.group(3).strip()
            return m.group(1) + m.group(2) + inner[0] + m.group(1) + inner[-1]
        return _RIC_AFTER.sub(repl, text)

    def _name_absorb(self, text):
        for index, rx in enumerate(self.absorb):
            if rx is not None:   # trailing name words left after a token join it ("[X] Outlet Centers Inc" -> "[X]")
                def absorb(m, index=index):
                    self._note(index, 'absorb', 'trailing name words', m)
                    return self._sentinel(index)
                text = rx.sub(absorb, text)
        return text

    def _joint_pass(self, text, counts):
        """v4 (E1): a joint-product partner joined to a token ("Quillco and [X]", "[X]-Quillco", "[X] and partner
        Quillco") is replaced as a whole by the token (a possessive of the token or partner is kept)."""
        for index, pairs in enumerate(self.joint_rx):
            for before, after in pairs:
                def rb(m, index=index):
                    counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                    self._note(index, 'joint_partner', m.group('partner'), m)
                    return self._sentinel(index)

                def ra(m, index=index):
                    counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                    self._note(index, 'joint_partner', m.group('partner'), m)
                    return self._sentinel(index) + (m.group('pp') or m.group('poss') or '')
                text = after.sub(ra, before.sub(rb, text))
        return text

    def _prefix_pass(self, text, counts):
        """v4 (E1): a phrase of the firm right before its token ("Fulfilled by [X]") joins the token."""
        for index, rx in enumerate(self.prefix_rx):
            if rx is not None:
                def rp(m, index=index):
                    counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                    self._note(index, 'prefix_absorb', m.group('p'), m)
                    return self._sentinel(index)
                text = rx.sub(rp, text)
        return text

    def _product_pass(self, text, counts):
        """v4 (E1): a phrase of the firm right after its token (any case unless marked capitalized; optional
        registered or trademark signs and possessive) joins the token: "[X] Zorbcloud Services" -> "[X]"."""
        for index, item in enumerate(self.products):
            if item is None:
                continue
            rx, phrases = item

            def rp(m, index=index, phrases=phrases):
                k = next(i for i in range(len(phrases)) if m.group(f'p{i}') is not None)
                counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                street = is_street_phrase(phrases[k]['phrase'])
                house = m.group('house')
                self._note(index, 'product_absorb', phrases[k]['phrase'], m, acronyms=phrases[k]['acronyms'],
                           house_number_absorbed=bool(street and house))
                keep = ''
                if house is not None and not street:      # the number stays with the space before the token
                    keep = m.string[m.start():m.end('house')] + re.match(r'[ \t]*', m.string[m.end('house'):]).group(0)
                return keep + self._sentinel(index) + (m.group('poss') or '')
            text = rx.sub(rp, text)
        return text

    def _acronym_pass(self, text, counts):
        """v4 (E1): acronyms of a phrase absorbed in this article (self.active) are masked as the firm's token."""
        for index, acronyms in sorted(self.active.items()):
            for a in sorted(acronyms, key=lambda x: (-len(x), x)):
                forms = [re.escape(a)] + ([re.escape(a[0] + a[1:].lower())] if a.isalpha() and
                                          not _is_english(a, self.english) else [])
                rx = _compile(r'(?<![A-Za-z0-9.$\-])(?:' + '|'.join(forms) + r")(?:['’]s)?(?![A-Za-z0-9])")

                def ra(m, index=index, a=a):
                    counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                    self._note(index, 'product_acronym', a, m)
                    return self._sentinel(index)
                text = rx.sub(ra, text)
        return text

    def mask_article(self, headline, body):
        """v4 (E1): mask a headline and body as one article. When a product phrase with acronyms is absorbed in either
        field, both fields are masked again with those acronyms active. Returns (headline, body, counts_headline,
        counts_body, log_headline, log_body); self.active holds the article's active acronyms by index."""
        self.active = {}
        h, c1 = self.mask(headline)
        l1 = self.log
        b, c2 = self.mask(body)
        l2 = self.log
        active = {}
        index_of = {t: i for i, t in enumerate(self.tokens)}
        for e in l1 + l2:
            if e['kind'] == 'product_absorb' and e.get('acronyms'):
                active.setdefault(index_of[e['token']], set()).update(e['acronyms'])
        if active:
            self.active = active
            h, c1 = self.mask(headline)
            l1 = self.log
            b, c2 = self.mask(body)
            l2 = self.log
        return h, b, c1, c2, l1, l2

    def _prefixes(self, index):
        return tuple(self.prod_meta[index].get('prefixes', ()))

    def _joints(self, index):
        return tuple(self.prod_meta[index].get('joint_partners', ()))

    def active_acronyms(self):
        """{token: sorted acronyms} of the current article (stored in the build metadata for verify)."""
        return {self.tokens[i]: sorted(a) for i, a in sorted(self.active.items())}

    def set_active_acronyms(self, by_token):
        index_of = {t: i for i, t in enumerate(self.tokens)}
        self.active = {index_of[t]: set(a) for t, a in (by_token or {}).items() if t in index_of}

    def mask(self, text):
        """Returns (masked text, replacements by token); self.log holds this call's replacement log. v4: joint
        partners, preceding phrases, product phrases and active acronyms (E1) after the name rules."""
        self.log = []
        text = scrub_boilerplate(text or '')
        counts = {}
        text = self._code_pass(text, counts)                   # v5 (D2): whole firm codes first
        for rule in self.rules:
            text, k = self._apply(rule, text)
            if k:
                counts[self.tokens[rule.index]] = counts.get(self.tokens[rule.index], 0) + k
        text = self._token_pass(text, counts)
        text = self._unit_pass(text, counts)                   # v6 (TD3C5 D1): initialism + corporate-unit word
        for _ in range(4):                                     # v5 (N-c): joint and prefix passes inside the loop
            before = text
            text = self._joint_pass(text, counts)
            text = self._prefix_pass(text, counts)
            text = self._product_pass(self._name_absorb(text), counts)
            if text == before:
                break
        text = self._acronym_pass(text, counts)
        text = self._ric_after_token(text, counts)
        text = self._glued_pass(text, counts)                  # v5 (D1): no token glued to a letter or digit
        text = re.sub(DROP + r'[ \t]*,?[ \t]*', '', text)       # deleted headquarters-town word and its comma
        text = mask_agency(text)
        text = mask_dates(text)
        for index, token in enumerate(self.tokens):
            text = text.replace(self._sentinel(index), token)
        if _SENT_OPEN in text or _SENT_CLOSE in text or DROP in text:
            raise ValueError('Unresolved masking sentinel')
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text, counts

    def residual(self, text):
        """Strict scan: every masking rule re-applied to final text (tokens neutralized), including the context guard
        and the embedded-name and misspelling checks; v4 (E1): a token still followed by a product phrase of its own
        firm, preceded by its own prefix phrase, joined to its joint-product partner, or an active acronym of the
        article. v5 (TD3C4 D1, D2): a token glued to a letter or digit ('glued_token') and a firm code of a masked firm
        ('firm_code'). Must be empty."""
        neutral = _TOKEN.sub(_NEUTRAL, text)
        hits = []
        for index, token in enumerate(self.tokens):
            anchor = re.escape(token)
            if re.search(r'(?<=[^\W_])' + anchor + r'|' + anchor + r'(?=[^\W_])', text):
                hits.append({'token': token, 'kind': 'glued_token', 'alias': 'glued'})
            rx = self.code_rx[index]
            if rx is not None:
                for m in rx.finditer(neutral):
                    if code_ok(m):
                        hits.append({'token': token, 'kind': 'firm_code', 'alias': m.group('pre').upper()})
                for m in self.project_rx[index].finditer(neutral):
                    pre = m.group('pre')
                    if pre.isupper() or (pre[:1].isupper() and pre[1:].islower()):
                        hits.append({'token': token, 'kind': 'firm_code', 'alias': 'Project ' + pre.upper()})
            ux = self.unit_rx[index]
            if ux is not None:                                 # v6 (TD3C5 D1)
                for m in ux.finditer(neutral):
                    if unit_ok(m):
                        hits.append({'token': token, 'kind': 'initialism_unit',
                                     'alias': m.group('pre').upper() + ' ' + m.group('unit')})
            item = self.products[index]
            if item is not None and _product_regex(anchor, item[1]).search(text):
                hits.append({'token': token, 'kind': 'product_after_token', 'alias': 'firm_products'})
            pr = self.prefix_rx[index]
            if pr is not None and _prefix_regex(anchor, self._prefixes(index)).search(text):
                hits.append({'token': token, 'kind': 'prefix_before_token', 'alias': 'firm_products'})
            for before, after in _joint_regexes(anchor, self._joints(index)):
                if before.search(text) or after.search(text):
                    hits.append({'token': token, 'kind': 'joint_partner_adjacent', 'alias': 'firm_products'})
        for index, acronyms in self.active.items():
            for a in acronyms:
                if re.search(r'(?<![A-Za-z0-9.$\-])' + re.escape(a) + r'(?![A-Za-z0-9])', neutral):
                    hits.append({'token': self.tokens[index], 'kind': 'product_acronym', 'alias': a})
        for rule in self.rules:
            for m in rule.regex.finditer(neutral):
                if rule.accepted(m):
                    hits.append({'token': self.tokens[rule.index], 'kind': rule.kind, 'alias': rule.alias})
        for m in _WORD_TOKEN.finditer(neutral):
            hit = self._token_hit(m.group(1))
            if hit is not None and context_guard(hit[2], m) != 'keep':
                hits.append({'token': self.tokens[hit[0]], 'kind': hit[1], 'alias': hit[2]})
        if AGENCY.search(neutral) and any(m.group(0) != m.group(0).lower() for m in AGENCY.finditer(neutral)):
            hits.append({'token': '[NEWS_AGENCY]', 'kind': 'agency', 'alias': 'Reuters'})
        return hits


def loose_scan(text, entries, comnams):
    """Indicator scan for the audit (not a masking rule): case-insensitive name variants and first words, and
    every ticker (including common acronyms) as an upper-case standalone word. Hits are possible residuals."""
    neutral = _TOKEN.sub(_NEUTRAL, text)
    hits = []
    for token, aliases in entries:
        terms = {v for v in aliases.get('names', ()) if not _ACRO.fullmatch(v) and not _AMP.fullmatch(v)}
        for c in comnams.get(token, ()):
            terms |= loose_terms(c)
        for v in sorted(terms):
            if re.search(variant_regex(v), neutral, re.I):
                hits.append({'token': token, 'kind': 'name_ci', 'alias': v})
        for t in sorted(set(aliases.get('tickers', ()))):
            if re.search(_B + re.escape(t) + _E, neutral):
                hits.append({'token': token, 'kind': 'ticker_any', 'alias': t})
    return hits
