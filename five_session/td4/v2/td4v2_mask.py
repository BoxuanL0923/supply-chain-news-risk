"""TD4 masker v2: the cleared TD3 v6 masker (td3_mask.Masker; the TD3 file is unchanged) subclassed with the rules the
four TD4 masking reviews required (td4m M1-M4, 2026-11-06). Pure functions and classes: no reader, no data. The frozen
word tables (common lower-case words, CRSP word document frequencies) and the private TD4 mask table are passed in by
the build (td4v2_build2.install). Examples in this file are invented.

Rule groups (review class in brackets; every group has a residual check in MaskerV2.residual or date_residuals_v2 and
an independent check in td4v2_scan):
  T  tickers: the English-word guard uses td3_mask._is_english plus the frozen common-word table, and a ticker that is
     also the firm's initialism is never kept (M1 B1); a ticker that heads a brief bullet or header and comes before a
     company verb is masked even when it is a month or dictionary word (M2, M4 A1, M3 P4); in the firm's own article
     its tickers, initialisms, first name words and collapsed acronyms (stop-list words included) are masked as a
     subject, and its Title-case tickers are masked (M1 B9, M2, M3 P2, P4, M4 A1).
  I  initialisms: two-letter initials of the first two core words (M2, M4 A1: ZORBEX QUILL -> 'ZQ'), initialisms of
     the first k >= 3 name words, initialism + the next name word ('ZQ Partners'), ampersand initialisms (M1 B7).
  N  name variants: words with digits in any case (M1 B3, M3 P1, P9), single letter + next word (M1 B3), the first
     three words when the second is an initial, numbers before the first word, 'U.S. <W>' from UNITED STATES names
     (M1 B5), camel-case and English-word splits of one-word names with a space or hyphen (M1 B4, M2 spaced names),
     leading parts of one-word names at a natural boundary (M1 B10, M3 P7), distinctive non-first name words (M1 B6),
     the core word after a generic first word (M2), a short first word or ticker glued to a name-word prefix or a
     frozen suffix (M3 P3, M4 A1: QX + bank -> 'Qxbank'), lower-case distinctive words (M2), first core words in
     company context (possessive, company verb; M2 english_word_short_name), weekday and month words never one-word
     aliases (M2), 2025 renames from the dated hand table (M2, M4).
  E  embedded names: a first core word or FactSet camel part at a camel boundary ('ZorbNet'), or glued in an
     upper-case token when not a common word (M4 A2), a distinctive word followed by any letters (M1 B12), a ticker
     glued after an English word ('Betzrbx'; M2); in the firm's own article its first core word followed by any letters.
  C  give-backs: a one-word alias absorbs an adjacent capitalised word in running text instead of leaving it next to
     the token (M2 name_part_remainder, '<Place> [X]', '[X] City'); dictionary-word aliases are kept in frozen
     collocations, two-word place phrases, Reuters slugs, index-movement lists and after adverbs in upper-case text
     outside company context (M2, M3 P5, M4 A3); the index and exchange guard does not apply to a firm's own aliases
     in its own article except for exact index names (M1 B9, M2).
  U  units and models: extended corporate-unit words and '<prefix> <Word> <legal suffix>' for non-English prefixes
     (M1 B8); a token followed by unit nouns and a legal suffix absorbs them; table acronyms active in the firm's own
     articles and the table's 'always' acronyms everywhere (M4 A4); an aircraft model number after a token (M4 A4, M2
     model give-back); hyphenated own name words next to a token (M1 B13, M3 P8); a RIC exchange suffix after a token
     (M2); firm codes whose prefix is a subsequence of a ticker or of the first name word keeping its first letter
     (>= 3 letters wherever the firm is a token, 2 letters in its own article; M1 B2, M3 P6).
  S  templates: the subject of an earnings-summary or preview template in the firm's own article (M1 B11).
  D  dates: a dateline day glued to the next word, 'MONTH, D', ordinals in any case, 'CY26' and apostrophe years
     (M2, M4 A5); '(Reuters) -' becomes a space.
  F  story fields (D8 v2): each unbracketed STORY_NUMBER/DATE/TIME field is blanked by its own value pattern, with
     spaces of equal length and no backtracking into the whitespace before '['; residuals for label stubs and clock
     times with seconds (TD4C B1, M1, M3, M4).
  A  mis-tag notices ('was inadvertently tagged to') are not used as a shown article (M1 advisory).
"""
import functools
import re

import td3_mask as M
from td3_mask import (ACRONYM_STOP, AMP_STOP, LEGAL, FORM_WORDS, GENERIC, PLACE_WORDS, OWNER_GENERIC, ABBREV,
                      PERSON_GIVEN, PHRASE_STOP, CODE_PREFIX_STOP, NON_FIRM_CODES, MONTHS, STATES, _STATES_RX,
                      normalize, camel_words, _letters, _is_english, _distinctive, _ACRO, _AMP, _INIT, _B, _E,
                      _compile, variant_regex, _upper_segment, DROP)

VERSION = 'TD4-mask-v2'
TABLE_SCHEMA_V2 = 'TD4_MASK_TABLES_v2'
COMMON_MIN = 50                 # a word is a common English word when it occurs >= 50 times written in lower case in
                                # the frozen 2021-2022 article corpus (td4v2_words.COMMON_WORDS file)
DF_MAX_NONFIRST = 5             # a non-first name word is distinctive when at most 5 CRSP PERMNOs carry it in a name

WEEKDAYS = frozenset({'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'})
MONTH_WORDS = frozenset({'JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER',
                         'OCTOBER', 'NOVEMBER', 'DECEMBER', 'JAN', 'FEB', 'MAR', 'APR', 'JUN', 'JUL', 'AUG', 'SEP',
                         'SEPT', 'OCT', 'NOV', 'DEC'})
INDEX_WORDS = frozenset({'DOW', 'NASDAQ', 'NYSE', 'RUSSELL', 'FTSE', 'NIKKEI', 'STOXX', 'DAX', 'CAC', 'KOSPI', 'SENSEX',
                         'NIFTY', 'TSX', 'ASX', 'MSCI', 'S&P', 'HANG'})
# Reuters story slugs ("UPDATE 1-INSIGHT-...", "LIVE MARKETS-..."): a dictionary-word alias in a slug is kept
SLUG_RX = re.compile(r'^(?:(?:UPDATE|RPT|REFILE|CORRECTED|EXCLUSIVE)[ \t]*\d*-)*$')
# a token-adjacent capitalised word that is never absorbed (function words, pronouns)
FUNCTION_CAPS = frozenset({'The', 'A', 'An', 'In', 'On', 'At', 'For', 'Of', 'And', 'Or', 'To', 'By', 'With', 'From',
                           'As', 'But', 'If', 'When', 'While', 'After', 'Before', 'Since', 'Its', 'Their', 'His', 'Her',
                           'Our', 'This', 'That', 'These', 'Those', 'It', 'We', 'They', 'He', 'She', 'I', 'Is', 'Was',
                           'Are', 'Were', 'Has', 'Have', 'Had', 'Will', 'Would', 'Can', 'Could', 'Not', 'No', 'Yes',
                           'Mr', 'Ms', 'Mrs', 'Dr', 'Inc', 'Corp', 'Co', 'Ltd', 'Plc', 'LLC', 'Group', 'Holdings',
                           'Source', 'Text', 'Further', 'Reporting', 'Editing', 'Writing', 'Additional', 'Compiled',
                           'Click', 'See', 'Read', 'Here', 'Note', 'Notes', 'Details', 'Link', 'Updates', 'Adds',
                           'Through', 'Description', 'Former', 'Like', 'Unlike', 'Per', 'Including', 'Between',
                           'Which', 'What', 'Who', 'Whom', 'Why', 'How', 'Where', 'Buy', 'Sell', 'Hold', 'Neutral',
                           'Says', 'Said', 'Say', 'Over', 'Under', 'Into', 'About', 'Via', 'Vs', 'Amid', 'Than',
                           'Also', 'Both', 'All', 'Any', 'Each', 'More', 'Most', 'New'})
# first core words never used by the company-context rule (too generic to name one firm)
CORE_STOP = frozenset({'INTERNATIONAL', 'NATIONAL', 'GLOBAL', 'GENERAL', 'FIRST', 'NEW', 'STANDARD', 'UNIVERSAL',
                       'CONSOLIDATED', 'INTEGRATED', 'ADVANCED', 'CENTRAL', 'WESTERN', 'EASTERN', 'SOUTHERN',
                       'NORTHERN', 'PACIFIC', 'ATLANTIC', 'COMMUNITY', 'PEOPLES', 'HOME', 'REAL', 'CAPITAL',
                       'DIGITAL', 'DATA', 'GREAT', 'PREMIER', 'SELECT', 'MIDDLE', 'NORTH', 'SOUTH', 'EAST', 'WEST'})
# two-word phrases in which a one-word alias is never the firm (TD3 PHRASE_STOP plus adjective forms)
PHRASE_STOP_V2 = PHRASE_STOP | frozenset({'MIDDLE EASTERN', 'UNITED ARAB', 'NORTH AMERICAN', 'SOUTH AMERICAN',
                                          'LATIN AMERICAN', 'SOUTH KOREAN', 'NORTH KOREAN', 'SOUTH AFRICAN',
                                          'FAR EASTERN', 'NEAR EAST', 'GOLDEN STATE', 'WHITE HOUSE', 'SUPREME COURT',
                                          'FEDERAL RESERVE', 'EUROPEAN UNION', 'PRIME MINISTER', 'STATE DEPARTMENT'})
# frozen collocations: a dictionary-word alias inside one of these is kept as written (M2, M3 P5, M4 A3); (word or None
# for any dictionary word, left regex ending at the word or None, right regex starting after it or None)
_PRICE_L = (r'(?:purchase|offering|closing|exercise|strike|conversion|target|share|stock|sale|sales|selling|asking|'
            r'average|floor|reserve|list|bid|spot|base|unit|issue|redemption|tender|initial|final|net|cash|market|retail|'
            r'wholesale|contract|consensus|transaction|deal|acquisition|subscription|reference|settlement|realized|'
            r'realised|higher|lower|oil|gas|gold|fuel|food|home|house|housing|power|electricity|commodity|import|export)'
            r'[ \t]+$')
_PRICE_R = (r'^[ \t]+(?:of|per|at|target|targets|range|war|wars|cut|cuts|hike|hikes|increase|increases|index|level|'
            r'levels|action|momentum|pressure|data|cap|caps|ceiling|floor|tag|point|points|growth|inflation|rise|'
            r'rises|drop|drops|decline|declines|move|moves|swing|swings|spike|spikes|outlook|forecast)\b')
COLLOCATIONS = (
    ('RESEARCH', None, r'^[ \t]+(?:roundup|and[ \t]+development|&[ \t]*development|note|notes|report|reports|firm|'
                       r'firms|analysts?|house|houses|desk)\b'),
    ('BRITISH', None, r'^[ \t]+thermal\b'),
    ('PRICE', _PRICE_L, None),
    ('PRICE', None, _PRICE_R),
    ('BELL', r'(?:closing|opening|taco|liberty|school|church|alarm|door|diving|dumb|blue|cow|jingle|tinker)[ \t]+$', None),
    ('BELL', None, r'^[ \t]+(?:curve|curves|ringer|ringing|rang|rings|weather|bottom|bottoms|jar|jars|pepper)\b'),
    ('PERFORMANCE', None, r'^[ \t]+(?:year|years|period|periods|share|shares|unit|units|bonus|bonuses|metric|metrics|'
                          r'fee|fees|obligation|obligations|criteria|goal|goals|target|targets|review|reviews|'
                          r'condition|conditions|based|data|issues?|problems?)\b'),
    ('PERFORMANCE', r'(?:for|strong|weak|financial|operating|past|share|stock|investment|fund|relative|economic|sales|'
                    r'business|solid|poor|good|better|best|record|overall|peak|high|top|its|their|the)[ \t]+$', None),
    ('OCCIDENTAL', r'[A-Z][a-z]+[ \t-]+$', None),
    ('ORIENTAL', r'[A-Z][a-z]+[ \t-]+$', None),
    ('OFFICE', r'(?:oval|back|head|home|post|box|patent|press|front|field|branch|regional|main)[ \t]+$', None),
    ('DOLLAR', None, r'^[ \t]+(?:index|bills?|bonds?|terms|amounts?|value|values|figures?|denominated|strength|'
                     r'weakness|rose|fell|firmed|weakened|gained|slipped|up|down|edged|eased|strengthened|rallied|'
                     r'versus|vs|against)\b'),
    ('DOLLAR', r'(?:U\.S\.|US|Canadian|Australian|Hong Kong|Singapore|Taiwan|New Zealand|strong|weak|stronger|weaker|'
               r'the|a|per|one|every|billion|million|trillion)[ \t]+$', None),
    (None, r'\bfor[ \t]+$', r'^[ \t]+years?\b'),
)
# index-movement context (an index word in a market list): kept outside the firm's own article, and in its own article
# when another index word is within 60 characters
INDEX_MOVE_R = re.compile(r'^[ \t]*(?:(?:was|is|were)[ \t]+)?(?:up|down|rose|fell|gained|lost|slipped|climbed|edged|'
                          r'added|ended|closed|jumped|dropped|declined|advanced|futures|e-minis?|industrials|average|'
                          r'composite|index|[+\-−]?[ \t]*\d)', re.I)
INDEX_CONT_R = re.compile(r'^[ \t]*(?:Composite|COMPOSITE|-?100\b|Industrial[ \t]+Average|INDUSTRIAL[ \t]+AVERAGE|'
                          r'Jones[ \t]+Industrial|JONES[ \t]+INDUSTRIAL|Jones[ \t]+(?:Transport|Utilit)|500\b|/[A-Z]{2,4}\b|'
                          r'futures|e-mini|Biotechnology[ \t]+Index|MidCap|SmallCap|Global[ \t]+1200|industrials|'
                          r'Industrials|Transportation[ \t]+Average)')
_INDEX_NEAR = re.compile(r'\b(?:S&P|Nasdaq|NASDAQ|Dow|DOW|Russell|RUSSELL|DJIA|FTSE|STOXX|Nikkei|sectors?)\b')

# company context
COMPANY_VERBS = ('said|says|say|reported|reports|report|posted|posts|announced|announces|announce|expects|expected|'
                 'forecasts?|raised|raises|lowered|lowers|beats?|missed|misses|plans|planned|agreed|agrees|filed|files|'
                 'completed|completes|entered|enters|received|receives|priced|prices|launched|launches|launch|named|names|'
                 'appointed|appoints|declared|declares|incurs|incurred|maintains|maintained|resumed|resumes|resuming|'
                 'sees|saw|shares|stock|stocks|ceo|cfo|chief|executive|executives|spokesperson|spokesman|spokeswoman|'
                 'board|chairman|president|management|confirmed|confirms|denied|denies|declined|declines|added|adds|'
                 'told|wrote|reached|signed|signs|sued|sues|won|wins|bought|buys|sold|sells|acquired|acquires|owns|'
                 'opened|opens|closed|closes|cut|cuts|hired|hires|delayed|delays|halted|halts|recalled|recalls|'
                 'warned|warns|stated|states|disclosed|discloses|unveiled|unveils|introduced|introduces|issued|issues|'
                 'agrees|partnered|partners|teamed|teams|invested|invests|offered|offers|aims|seeks|sought|wants|'
                 'is|was|has|had|will|would')
COMPANY_RIGHT = re.compile(r"^(?:[ \t]*:|[ \t]+[-–][ \t]|[ \t]*,[ \t]+(?:the|which|whose|a|an)\b|[ \t]+(?:INC|CORP|CO|"
                           r"LTD|PLC|LLC|LP|SA|AG|NV|SE|GROUP|HOLDINGS|Inc|Corp|Co|Ltd|Plc|plc|Group|Holdings)\b|"
                           r"[ \t]+(?:" + COMPANY_VERBS + r")\b)", re.I)
COMPANY_LEFT = re.compile(r'(?:according[ \t]+to|told|reported[ \t]+by|said[ \t]+in|wrote[ \t]+in|published[ \t]+by|'
                          r'interview[ \t]+with|shares[ \t]+of|stake[ \t]+in|rivals?|peers?|such[ \t]+as|including|'
                          r'unit[ \t]+of|owner|parent|versus|vs\.?)[ \t]+(?:the[ \t]+)?$', re.I)
_SENT = '[-]'          # a TD3 sentinel is U+E000, U+E100 + index, U+E001
SUBJECT_LEFT = re.compile(r'(?:^|\n|\*[ \t]*|BRIEF-|BUZZ-|(?:UPDATE|RPT|REFILE|CORRECTED)[ \t]*\d*-|'
                          r'[ \t]*[-–:][ \t]*|\[[A-Z0-9_]+\][ \t]*[-–:][ \t]*)$')
# a dictionary-word alias in upper-case text right after one of these words is an adjective or adverb, not the firm
AUX_LEFT = frozenset({'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING', 'CURRENTLY', 'NOT', 'STILL', 'ALSO', 'NOW',
                      'JUST', 'ALREADY', 'WILL', 'WOULD', 'CAN', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'GO', 'GOES',
                      'WENT', 'GOING', 'GET', 'GETS', 'GOT', 'VERY', 'MORE', 'MOST', 'LESS', 'SO', 'TOO', 'REMAINS',
                      'REMAIN', 'STAYS', 'STAY', 'BECOME', 'BECOMES', 'BECAME', 'FULLY'})
VERBISH_NEXT = re.compile(r'^[ \t]+[A-Z]{3,}(?:S|ED|ING)\b')

# T/I: two-letter words that are never two-letter initialisms (ratings, units, common abbreviations)
TWO_STOP = frozenset({'AA', 'BB', 'CC', 'DD', 'HQ', 'IR', 'OS', 'VR', 'AR', 'UI', 'UX', 'HD', 'ID', 'IP', 'IQ', 'PS',
                      'TB', 'GB', 'MB', 'KB', 'KM', 'KG', 'LB', 'MW', 'GW', 'KW', 'OZ', 'PT', 'DR', 'MR', 'ST', 'RD',
                      'DC', 'AC', 'EM', 'FX', 'IB', 'ML', 'MM', 'QE', 'RV', 'SF', 'TX', 'VP', 'XL', 'XR', 'ZZ', 'OP',
                      'RX', 'DX', 'BC', 'AD', 'MP', 'MA', 'GP', 'LA', 'SK', 'NZ', 'SG', 'HK'})
# N: words with digits that are never digit name variants
DIGIT_STOP = frozenset({'3D', '2D', '4D', '4G', '5G', '6G', '3G', '2G', 'B2B', 'B2C', 'P2P', 'M2M', 'D2C', 'C2C',
                        '401K', '1Q', '2Q', '3Q', '4Q', '1H', '2H', 'Q1', 'Q2', 'Q3', 'Q4', 'H1', 'H2', 'COVID19',
                        '24X7', '24/7', '1ST', '2ND', '3RD', '4TH', '5TH', '6TH', '7TH', '8TH', '9TH', '10TH', 'MP3',
                        'MP4', 'A1', 'A2', 'B1', 'B2', 'G7', 'G20', 'K12', 'W2', 'F1', 'E3', 'S1', 'PM10'})
# N: frozen suffixes of glued short-name brands (first word or ticker of 2-4 letters + suffix)
GLUE_SUFFIXES = ('bank', 'pay', 'soft', 'tech', 'semi', 'net', 'link', 'care', 'med', 'bio', 'pharma', 'gen', 'labs',
                 'lab', 'fin', 'cap', 'cloud', 'data', 'shop', 'store', 'mart', 'hub', 'pro', 'plus', 'life', 'health',
                 'rx', 'dx', 'tx', 'trade', 'invest', 'card', 'cash', 'coin', 'chain', 'max', 'one', 'go', 'ai')
# U: extended corporate-unit words (M1 B8) and unit nouns for a legal-suffix tail after a token
UNIT_WORDS_V2 = M.UNIT_WORDS + ('Finance', 'Financing', 'Funding', 'Operating', 'Operations', 'Payments', 'Payment',
                                'Leasing', 'Credit', 'Securities', 'Markets', 'Management', 'Asset', 'Assets', 'Brands',
                                'Retail', 'Media', 'Networks', 'Communications', 'Logistics', 'Resources', 'Power',
                                'Utilities', 'Insurance', 'Re', 'Reinsurance', 'Healthcare', 'Medical', 'Software',
                                'Digital', 'Exploration', 'Production', 'Pipeline', 'Infrastructure', 'Development',
                                'Research', 'Laboratories', 'Manufacturing', 'Engineering', 'Electronics', 'Aerospace',
                                'Defense', 'Marine', 'Shipping', 'Sciences', 'Life', 'Lending', 'Mortgage', 'Servicing',
                                'Treasury', 'Overseas', 'Cloud', 'Data')
LEGAL_TAIL = (r'(?:Inc|Incorporated|LLC|L\.L\.C\.|LP|L\.P\.|Ltd|Limited|Co|Corp|Corporation|plc|PLC|SA|S\.A\.|AG|NV|'
              r'N\.V\.|BV|B\.V\.|GmbH|SE|Pty|Pvt|SpA|S\.p\.A\.|SAS|AB|ASA|Oyj|KK|K\.K\.)\.?')
_UNIT_NOUN = '(?:' + '|'.join(re.escape(u) for u in sorted(set(UNIT_WORDS_V2) | {
    'Holdings', 'Group', 'International', 'Global', 'Americas', 'America', 'Europe', 'Asia', 'Pacific', 'Canada',
    'India', 'China', 'Japan', 'Korea', 'UK', 'USA', 'US', 'Australia', 'Brasil', 'Brazil', 'Mexico', 'Germany',
    'France', 'Italia', 'Espana', 'Nederland', 'Deutschland', 'Ireland', 'Singapore', 'Products', 'Industries',
    'Chemicals', 'Materials', 'Metals', 'Mining', 'Oil', 'Gas', 'Motors', 'Automotive', 'Airlines', 'Stores', 'Foods',
    'Transport', 'Transportation', 'Semiconductor', 'Devices', 'Instruments', 'Telecom', 'Wireless', 'Entertainment',
    'Studios', 'Pictures', 'Records', 'Publishing', 'Hotels', 'Resorts', 'Gaming', 'Interactive', 'Labs', 'Bio',
    'Pharmaceuticals', 'Biosciences', 'Diagnostics', 'Equipment', 'Machinery', 'Construction', 'Homes', 'Realty',
    'Trust', 'Bancorp', 'Bank', 'Banking', 'Capital', 'Fund', 'Funds', 'Advisors', 'Advisers', 'Investors',
    'Solutions', 'Services', 'Systems', 'Technologies', 'Technology', 'Energy', 'Electric', 'Water', 'Renewables',
    'Solar', 'Wind', 'Nuclear'}, key=lambda x: (-len(x), x))) + ')'
# U: aircraft model number after a token (a 3-digit model with a '-<digits>' variant or MAX, a 7x7 family number, or a
# 3-digit model followed by an aircraft noun)
MODEL_AFTER = (r'(?P<model>[ \t]+(?:\d{3}(?:-\d{1,4}[A-Z]{0,3}(?:[ \t]?(?:ER|LR|XR|F))?|[ \t]?MAX(?:[ \t]?\d{1,2})?)'
               r'|7[0-8]7s?(?![\d,.])|\d{3}s?(?=[ \t]+(?i:regional[ \t]+jets?|jets?|aircraft|airplanes?|planes?|'
               r'airliners?|freighters?|turboprops?|helicopters?|business[ \t]+jets?))))')
# S: earnings-summary and preview templates (the subject is the firm the article was chosen for)
TEMPLATE_RX = (re.compile(r'^(?P<s>[A-Z0-9][^\n]{1,90}?)[ \t]+reports[ \t]+results[ \t]+for[ \t]+the[ \t]+'
                          r'(?:quarter|period|year|half|fiscal|nine|six|three|twelve)'),
               re.compile(r'(?:^|\n)\*[ \t]*(?P<s>[A-Z0-9][^\n*]{1,90}?)[ \t]+is[ \t]+expected[ \t]+to[ \t]+'
                          r'(?:show|report|post)'),
               re.compile(r"for[ \t]+(?P<s>[A-Z0-9][^\n*]{1,90}?)[ \t]+is[ \t]+for[ \t]+a\b"))
# A: mis-tag withdrawal notices
MISTAG = re.compile(r'\b(?:inadvertently|incorrectly|wrongly|mistakenly|erroneously)[ \t]+(?:tagged|issued|published|'
                    r'sent|attached|transmitted|coded|linked)\b|\btagged[ \t]+(?:in[ \t]+error|incorrectly|wrongly|'
                    r'mistakenly)\b|\bwas[ \t]+tagged[ \t]+to\b[^\n]{0,80}\bunrelated\b', re.I)

# ------------------------------------------------------------------------------------------------- F story fields
STORY_V2 = (
    ('NUMBER', re.compile(r'STORY_NUMBER[ \t]*:[ \t]*(?!\[)[A-Za-z0-9][A-Za-z0-9_\-]{3,24}')),
    ('DATE', re.compile(r'STORY_DATE[ \t]*:[ \t]*(?!\[)(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|'
                        r'\d{1,2}[\-.]\d{1,2}[\-.]\d{2,4}|[A-Za-z]{3,9}\.?[ \t]+\d{1,2},?[ \t]+\d{4}|'
                        r'\d{1,2}[ \t]+[A-Za-z]{3,9}\.?[ \t]+\d{4})')),
    ('TIME', re.compile(r'STORY_TIME[ \t]*:[ \t]*(?!\[)\d{1,2}:?\d{2}(?::\d{2})?(?:[ \t]*[AaPp]\.?[Mm]\.?)?'
                        r'(?:[ \t]*(?:GMT|UTC|ET|EST|EDT|BST|CET))?')),
)
STORY_STUB = re.compile(r'(?<![A-Za-z])(?:STORY_|ORY_(?:NUMBER|DATE|TIME)|RY_(?:NUMBER|DATE|TIME)|Y_(?:NUMBER|DATE|TIME)|'
                        r'_(?:NUMBER|DATE|TIME)\b)|(?<![A-Za-z])(?:T?IME|ME)[ \t]*:[ \t]*\d{1,2}:?\d{2}')
CLOCK_SECONDS = re.compile(r'(?<![\d:])\d{1,2}:\d{2}:\d{2}(?![\d:])')


def story_fields_v2(text):
    """D8 v2 (TD4C B1): every unbracketed STORY_NUMBER, STORY_DATE and STORY_TIME field is replaced by spaces of the
    same length, each by its own value pattern ([ \\t]* only, so a match never starts where a '[' follows the
    whitespace and never runs into the next field). Returns (text, replacements)."""
    n = 0
    for _, rx in STORY_V2:
        def rep(m):
            return ' ' * (m.end() - m.start())
        text, k = rx.subn(rep, text or '')
        n += k
    return text, n


# ------------------------------------------------------------------------------------------------------- D dates
_MONTHS_ANY = '(?i:' + '|'.join(sorted({w for w in M._MONTH_WORDS}, key=len, reverse=True)) + r')\.?'
DATE_PRE = (
    # a dateline day glued to the next word ("July 1Zorbex"): a space is inserted (the day is then a date)
    ('v2_glued_dateline', re.compile(r'\b(' + MONTHS + r')([ \t]+)(\d{1,2})(?=[A-Z][a-z])'), r'\1\2\3 '),
    ('v2_month_comma_day', re.compile(r'\b' + _MONTHS_ANY + r',[ \t]*\d{1,2}(?i:st|nd|rd|th)?(?![\d%]|[.,]\d|[ \t]*'
                                      r'(?:%|percent|pct))\b'), '[DATE]'),
    ('v2_ordinal_anycase', re.compile(r'\b' + _MONTHS_ANY + r'[ \t]+\d{1,2}(?:ST|ND|RD|TH|St|Nd|Rd|Th|sT|nD|rD|tH)\b'),
     '[DATE]'),
    ('v2_day_month_anycase', re.compile(r'\b\d{1,2}(?i:st|nd|rd|th)?[ \t]+(?i:January|February|March|April|June|July|'
                                        r'August|September|October|November|December)\b'), '[DATE]'),
    ('v2_cy_year', re.compile(r"\b(CY|FY|MY)[ \t]?['’]?(\d{2})\b(?![\d.,%])"), r'\1 [YEAR]'),
    ('v2_apostrophe_year', re.compile(r"(?<![\w'’])['’]\d{2}s?\b(?![ \t]*(?:%|percent|pct|ft|feet|inch|inches|lb))"),
     '[YEAR]'),
)


def mask_dates_v2(text):
    for _, rx, repl in DATE_PRE:
        text = rx.sub(repl, text)
    return M.mask_dates(text)


def date_residuals_v2(text):
    """TD3 date_residuals plus the v2 date patterns, story-field stubs and clock times with seconds."""
    hits = M.date_residuals(text)
    for name, rx, _ in DATE_PRE:
        if name != 'v2_glued_dateline' and rx.search(text):
            hits.append(name)
    if re.search(r'\b' + MONTHS + r'[ \t]+\d{1,2}(?=[A-Za-z])', text):
        hits.append('v2_glued_dateline')
    if STORY_STUB.search(text):
        hits.append('v2_story_stub')
    if CLOCK_SECONDS.search(text):
        hits.append('v2_clock_seconds')
    return hits


def scrub_v2(text):
    """TD3 scrub_boilerplate with '(Reuters) -' replaced by a space (TD3 removed it, which can glue a dateline day to
    the next word) and the unbracketed story fields blanked (D8 v2; a no-op after the pre-truncation hook)."""
    text, _ = story_fields_v2(text or '')
    text = re.sub(r'\((Reuters(?: Breakingviews)?)\)[ \t]*-?[ \t]*', ' (\\1) - ', text)
    return M.scrub_boilerplate(text)


# ----------------------------------------------------------------------------------------------------- word tables
_STATE = {'common': frozenset(), 'df': {}, 'tables': None, 'last_day': None}


def install(common, df, tables, last_day):
    """Frozen word tables and the private TD4 mask table, set by the build before any alias is generated."""
    _STATE.update(common=frozenset(common), df=dict(df), tables=tables, last_day=last_day)


def common_word(w):
    w = (w or '').lower()
    return w in _STATE['common']


def dict_like(word, english):
    """A dictionary, place, index, weekday or month word: as a one-word alias it needs company context."""
    w = (word or '').upper().replace('.', '')
    lw = w.lower()
    return (w in WEEKDAYS or w in MONTH_WORDS or w in INDEX_WORDS or lw in GENERIC or lw in OWNER_GENERIC or
            lw in PLACE_WORDS or _is_english(w, english) or common_word(w) or
            (w.endswith('S') and common_word(w[:-1])))


def validate_tables_v2(t):
    if not isinstance(t, dict) or t.get('schema') != TABLE_SCHEMA_V2:
        raise ValueError('TD4 v2 mask table schema must be ' + TABLE_SCHEMA_V2)
    for sec in ('common_names', 'owner_aliases', 'always_acronyms', 'renames'):
        if not isinstance(t.get(sec), dict):
            raise ValueError('TD4 v2 mask table section missing: ' + sec)
        for k, e in t[sec].items():
            if not isinstance(e, dict) or not str(e.get('source', '')).strip():
                raise ValueError(f'TD4 v2 mask table entry without a source note: {sec}/{k}')
    for k, e in t['renames'].items():
        if not re.fullmatch(r'\d+', k) or not e.get('names') or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', e.get('dated', '')):
            raise ValueError('rename entry needs a PERMNO key, names and a date: ' + k)
    return t


def _table_entries(section, comnams):
    """Entries of a comnam-prefix keyed v2 table section for a firm (whole-word prefix of a CRSP name)."""
    t = _STATE['tables'] or {}
    out = []
    for c in comnams:
        norm = ' '.join(M._clean(c).split())
        for k, e in (t.get(section) or {}).items():
            if norm == k or norm.startswith(k + ' '):
                out.append(e)
    return out


# ----------------------------------------------------------------------------------------------- v2 alias helpers
def _core_words(comnam):
    """Significant words of a CRSP name (THE dropped, trailing legal words stripped)."""
    words = [w for w in normalize(comnam) if w not in ('&',)]
    if words and words[0] == 'THE':
        words = words[1:]
    words, _ = M._strip(words)
    return words


def _first_core(comnam):
    """First core word; for a name that starts with a two-word place or phrase on PHRASE_STOP (NEW YORK ...), the word
    after it."""
    w = _core_words(comnam)
    if len(w) >= 3 and ' '.join(w[:2]) in PHRASE_STOP:
        return w[2]
    return w[0] if w else ''


def _splits(word, english, fs_names):
    """Camel-case splits of a one-word name from mixed-case FactSet names and English-word splits A+B."""
    out = set()
    for n in fs_names:
        for tok in re.findall(r'[A-Za-z]+', n or ''):
            if tok.upper() == word and re.search(r'[a-z][A-Z]', tok):
                parts = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tok).upper().split()
                if len(parts) >= 2:
                    out.add(' '.join(parts))
    if len(word) >= 6 and word.isalpha() and not _is_english(word, english):
        for i in range(2, len(word) - 2):
            a, b = word[:i], word[i:]
            if a.lower() in english and b.lower() in english and len(b) >= 3 and (len(a) >= 3 or a in ('GO', 'MY')):
                out.add(a + ' ' + b)
    return out


def _subsequences(s, lo=2, hi=4):
    """Letter subsequences of s that keep its first letter, of length lo..hi."""
    s = s.upper()
    out = set()

    def rec(i, acc):
        if lo <= len(acc) <= hi:
            out.add(acc)
        if len(acc) >= hi:
            return
        for j in range(i, len(s)):
            rec(j + 1, acc + s[j])
    if s:
        rec(1, s[0])
    return out


def add_v2(al, comnams, fs_names, sib_comnams, english, permno=None):
    """v2 aliases of one firm on top of td3_build.aliases_for's dict (returned as a new dict; TD3's lists are kept and
    extended). Pure function of the firm's names, tickers, the frozen word tables and the v2 mask table."""
    al = {k: (list(v) if isinstance(v, list) else v) for k, v in al.items()}
    names_all = list(comnams) + list(sib_comnams)
    tickers = sorted(set(al.get('tickers', ())) | set(al.get('tickers_extra', ())))
    v2 = {'rename_names': [], 'digit': [], 'compounds': [], 'core_ctx': [], 'init2': [], 'lower_ok': [],
          'camel_terms': [], 'lead_terms': [], 'suffix_terms': [], 'own_words': [], 'own_title': [],
          'own_codes': [], 'codes3': [], 'own_lead': [], 'owner_aliases': [], 'always_acronyms': [],
          'own_acronyms': [], 'initialisms': [], 'hyphen_words': [], 'dropped_weekday_month': []}
    # renames (dated hand table; used for periods whose last day is in or after the rename year)
    t = _STATE['tables'] or {}
    rn = (t.get('renames') or {}).get(str(int(permno))) if permno is not None else None
    if rn and _STATE['last_day'] and rn['dated'][:4] <= _STATE['last_day'][:4]:
        v2['rename_names'] = list(rn['names'])
        names_all += list(rn['names'])
        for c in rn['names']:                    # a dictionary word alone is not a variant (own-article rules apply)
            al['names'] = sorted(set(al['names']) | {v for v in M.name_variants(c)
                                                     if ' ' in v or not dict_like(v, english)})
        tickers = sorted(set(tickers) | set(rn.get('tickers', ())))
    names = set(al.get('names', ()))
    extra_names, near = set(), set(al.get('near', ()))
    for c in names_all + list(fs_names):
        # N: words with letters and digits (any case), from CRSP and FactSet names
        for w in (w for w in re.split(r'[^A-Za-z0-9&]+', M._clean(c)) if w):
            if re.search(r'[A-Z]', w) and re.search(r'\d', w) and len(w) >= 2 and w not in DIGIT_STOP and \
                    w not in NON_FIRM_CODES and not re.fullmatch(r'\d+(?:ST|ND|RD|TH|S)', w):
                v2['digit'].append(w)
    for c in list(fs_names):
        # N: distinctive non-first words of FactSet names (issuer-linked entities)
        for w in _core_words(c)[1:]:
            if w.isalpha() and _distinctive(w, english, 5) and not common_word(w) and \
                    _STATE['df'].get(w, 0) <= DF_MAX_NONFIRST and w not in ABBREV:
                extra_names.add(w)
    for c in names_all:
        cw = _core_words(c)
        if not cw:
            continue
        first = _first_core(c)
        # N: single letter + next word ("Y MABS"); first three words when the second is an initial ("X A Y")
        if len(cw) >= 2 and _INIT.fullmatch(cw[0]) and cw[1].isalpha() and len(cw[1]) >= 3:
            extra_names.add(cw[0] + ' ' + cw[1])
        if len(cw) >= 3 and _INIT.fullmatch(cw[1]) and cw[0].isalpha() and cw[2].isalpha():
            extra_names.add(' '.join(cw[:3]))
        # N: numbers before the first word ("1 800 ZORB")
        if len(cw) >= 2 and re.fullmatch(r'[\d.]+', cw[0]):
            k = next((i for i, w in enumerate(cw) if w.isalpha()), None)
            if k is not None and k <= 3 and len(cw[k]) >= 3:
                extra_names.add(' '.join(cw[:k + 1]))
        # N: U.S. <W> from UNITED STATES names
        if len(cw) >= 3 and cw[0] == 'UNITED' and cw[1] == 'STATES':
            extra_names.add('U.S. ' + cw[2])
            if len(cw) >= 4 and cw[3] not in LEGAL:
                extra_names.add('U.S. ' + cw[2] + ' ' + cw[3])
        # N: the first core word after a generic first word or a two-word place ("BANCO ZORBEX", "NEW YORK ZORBEX")
        if first and first != cw[0] and first.isalpha() and _distinctive(first, english, 4) and \
                not dict_like(first, english):
            extra_names.add(first)
        if cw[0].lower() in GENERIC and len(cw) >= 2 and cw[1].isalpha() and _distinctive(cw[1], english, 4) and \
                not dict_like(cw[1], english) and cw[1] not in M.UNIT_WORDS:
            extra_names.add(cw[1])
        # I: initialisms of the first k words (TD3 rule on the full name), initialism + rest, two-letter initials
        nw = [w for w in normalize(c)]
        if nw and nw[0] == 'THE':
            nw = nw[1:]
        inis = M._initialisms(nw)
        v2['initialisms'] += sorted(inis)
        for ini in inis:
            if len(ini) >= 3 and ini not in ACRONYM_STOP and not _is_english(ini, english) and \
                    not common_word(ini) and ini.lower() not in GENERIC:
                extra_names.add('.'.join(ini) + '.')
        alpha = [w for w in cw if w not in M._CONNECT]
        for k in range(2, min(len(alpha), 4)):
            if all(w[:1].isalpha() for w in alpha[:k]):
                ini = ''.join(_letters(w) if (_ACRO.fullmatch(w) or _INIT.fullmatch(w)) else w[0] for w in alpha[:k])
                rest = alpha[k]
                if ini.isalpha() and rest.isalpha() and len(rest) >= 4 and rest not in LEGAL and \
                        ini not in ACRONYM_STOP | TWO_STOP:
                    extra_names.add('.'.join(ini) + '. ' + rest)
        if len(alpha) >= 2 and alpha[0].isalpha() and alpha[1][:1].isalpha() and not _ACRO.fullmatch(alpha[1]):
            ini2 = alpha[0][0] + alpha[1][0]
            if ini2 not in ACRONYM_STOP | TWO_STOP | CODE_PREFIX_STOP and ini2 not in LEGAL and \
                    ini2 not in FORM_WORDS and ini2 not in tickers and \
                    not (alpha[0].lower() in GENERIC and alpha[1].lower() in GENERIC):
                v2['init2'].append(ini2)
        if _ACRO.fullmatch(cw[0]) and len(_letters(cw[0])) == 2 and _letters(cw[0]).isalpha():
            ini2 = _letters(cw[0])
            if ini2 not in ACRONYM_STOP | TWO_STOP | CODE_PREFIX_STOP and ini2 not in LEGAL and ini2 not in tickers:
                v2['init2'].append(ini2)
        # I: ampersand initialisms ("X Y & Z" -> XY&Z) and an ampersand first word ("X & Y SERVICES" -> X&Y)
        if _AMP.fullmatch(cw[0]) and cw[0].replace('.', '') not in AMP_STOP:
            extra_names.add(cw[0])
        words = normalize(c)
        if '&' in words:
            i = words.index('&')
            before = [w for w in words[max(0, i - 3):i] if w not in M._CONNECT and w != 'THE']
            after, _ = M._strip([w for w in words[i + 1:i + 3] if w not in M._CONNECT])
            if before and after and all(w[:1].isalpha() for w in before + after):
                a = ''.join(_letters(w) if _ACRO.fullmatch(w) or _INIT.fullmatch(w) else w[0] for w in before)
                b = ''.join(_letters(w) if _ACRO.fullmatch(w) or _INIT.fullmatch(w) else w[0] for w in after)
                if a.isalpha() and b.isalpha() and len(a + b) >= 3 and (a + '&' + b) not in AMP_STOP:
                    extra_names.add('.'.join(a) + '.&' + '.'.join(b) + '.')
        # N: splits and leading parts of a one-word first name word (camel-case or English boundary)
        if first and first.isalpha() and len(first) >= 6:
            sps = _splits(first, english, fs_names)
            v2['splits'] = sorted(set(v2.get('splits', [])) | sps)
            for i in range(4, len(first) - 2):
                p, r = first[:i], first[i:]
                boundary = (len(r) >= 3 and _is_english(r, english)) or any(sp.startswith(p + ' ') for sp in sps)
                if boundary and _distinctive(p, english, 4) and not common_word(p):
                    extra_names.add(p)
        # N: distinctive non-first name words (M1 B6)
        for w in cw[1:]:
            if w.isalpha() and w != first and _distinctive(w, english, 5) and not common_word(w) and \
                    _STATE['df'].get(w, 0) <= DF_MAX_NONFIRST and w not in ABBREV:
                extra_names.add(w)
        # N: glued short-name compounds (first word or ticker of 2-4 letters + name-word prefix or frozen suffix)
        heads = set()
        if first and first.isalpha() and 2 <= len(first) <= 4:
            heads.add(first)
        for tk in tickers:
            if tk.isalpha() and 2 <= len(tk) <= 4:
                heads.add(tk.upper())
        nxt = cw[1] if len(cw) >= 2 and cw[1].isalpha() else ''
        sufs = {s.upper() for s in GLUE_SUFFIXES} | {nxt[:k] for k in range(3, len(nxt) + 1)}
        for h in heads:
            for s in sufs:
                cpd = h + s
                if len(cpd) >= 5 and not _is_english(cpd, english) and not common_word(cpd) and cpd not in names:
                    v2['compounds'].append(cpd)
        # N: first core words that TD3 does not use alone: masked in company context (M2 short names)
        if first and first.isalpha() and len(first) >= 3 and first not in WEEKDAYS and first not in MONTH_WORDS:
            v2['core_ctx'].append(first)
        # T: own-article words (first core word incl. generic, place and legal-list words; collapsed acronym)
        if first:
            ow = _letters(first) if _ACRO.fullmatch(first) else first.replace('.', '')
            if len(ow.replace('&', '')) >= 2:
                v2['own_words'].append(ow)
        if first and first.isalpha() and len(first) >= 5:
            v2['own_lead'].append(first)
        # E: camel terms (first core word, any but generic and place words, >= 4 letters)
        if first and first.isalpha() and len(first) >= 4 and first.lower() not in GENERIC and \
                first.lower() not in PLACE_WORDS and first not in CORE_STOP:
            v2['camel_terms'].append(first)
    fs_parts = set()
    unit_up = {u.upper() for u in M.UNIT_WORDS}
    for n in fs_names:                                   # E: FactSet camel-case parts
        for tok in re.findall(r'[A-Za-z]+', n or ''):
            if re.search(r'[a-z][A-Z]', tok):
                for part in re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tok).split():
                    pu = part.upper()
                    if len(part) >= 4 and part.isalpha() and pu not in LEGAL and pu.lower() not in GENERIC and \
                            pu.lower() not in PLACE_WORDS and pu not in CORE_STOP and pu not in unit_up:
                        v2['camel_terms'].append(pu)
                        fs_parts.add(pu)
    # E: distinctive words followed by any letters (B12); tickers glued after an English word
    v2['lead_terms'] = sorted({w for w, k in al.get('embedded', ()) if k == 'word' and len(w) >= 5 and
                               not common_word(w)})
    v2['camel_caps_terms'] = sorted({t for t in v2['camel_terms'] if not common_word(t) and
                                     (not _is_english(t, english) or t in fs_parts)})
    v2['suffix_terms'] = sorted({tk.upper() for tk in tickers if tk.isalpha() and len(tk) >= 3 and
                                 not _is_english(tk, english) and not common_word(tk)})
    # T: own-article tickers (all), initialisms incl. stop-list ones
    v2['own_words'] += [tk.upper() for tk in tickers if re.fullmatch(r'[A-Za-z]{2,6}', tk)]
    v2['own_words'] += [i for i in v2['initialisms'] if 2 <= len(i) <= 6]
    v2['own_title'] = sorted({tk.upper() for tk in tickers if tk.isalpha() and len(tk) >= 3 and not common_word(tk)})
    # U: code prefixes from subsequences of tickers (all lengths) and of the first name word (2-4 letters) that keep
    # the first letter; 3 letters or more in every article where the firm is a token, 2 letters in its own article
    codes = set()
    for tk in tickers:
        if tk.isalpha() and len(tk) >= 3:
            codes |= _subsequences(tk, 2, len(tk))
    for c in names_all:
        f = _first_core(c)
        if f.isalpha() and len(f) >= 4 and f.lower() not in GENERIC and f.lower() not in PLACE_WORDS:
            codes |= _subsequences(f, 2, 4)
    codes = {p for p in codes if p not in CODE_PREFIX_STOP and p not in MONTH_WORDS and not _is_english(p, english)
             and not common_word(p) and p not in ACRONYM_STOP}
    v2['codes3'] = sorted(p for p in codes if len(p) >= 3)
    v2['own_codes'] = sorted(p for p in codes if len(p) == 2)
    # N: lower-case acceptance for distinctive non-English single-word variants (>= 5 letters, not common)
    lower_ok = set()
    for v in names | extra_names:
        if ' ' not in v and v.isalpha() and len(v) >= 5 and _distinctive(v, english, 5) and not common_word(v):
            lower_ok.add(v)
    # table entries (private, firm-specific)
    for e in _table_entries('common_names', names_all):
        al['common'] = sorted(set(al.get('common', ())) | set(e['aliases']))
    for e in _table_entries('owner_aliases', names_all):
        v2['owner_aliases'] += list(e['aliases'])
    for e in _table_entries('always_acronyms', names_all):
        v2['always_acronyms'] += list(e['acronyms'])
    for p in (al.get('products') or {}).get('products', ()):
        v2['own_acronyms'] += [a for a in p.get('acronyms', ()) if len(a) >= 3]
    # weekday and month words are never one-word aliases (M2)
    wm = WEEKDAYS | MONTH_WORDS
    drop = {v for v in names | extra_names if ' ' not in v and v.replace('.', '') in wm}
    v2['dropped_weekday_month'] = sorted(drop | {v for v in al.get('common', ()) if v.upper() in wm} |
                                         {v for v in near if v.upper() in wm})
    al['names'] = sorted((names | extra_names) - drop)
    al['common'] = sorted(v for v in al.get('common', ()) if v.upper() not in wm)
    al['near'] = sorted(v for v in near - set(al['names']) if v.upper() not in wm)
    v2['splits'] = sorted(v for v in v2.get('splits', []) if v not in set(al['names']) and v not in set(al['near']))
    # hyphen-join words: the firm's absorb words of >= 3 letters
    v2['hyphen_words'] = sorted(w for w in M.absorb_words(al) if len(w) >= 3 and w.isalpha())
    for k in ('digit', 'compounds', 'core_ctx', 'init2', 'camel_terms', 'own_words', 'own_lead', 'owner_aliases',
              'always_acronyms', 'own_acronyms', 'initialisms'):
        v2[k] = sorted(set(v2[k]))
    v2['lower_ok'] = sorted(lower_ok)
    v2['core_ctx'] = [w for w in v2['core_ctx'] if w not in set(al['names'])]
    al['v2'] = v2
    return al


# --------------------------------------------------------------------------------------------------------- rules
_ABSORB_KINDS = ('name', 'common', 'near_name', 'core_ctx', 'own_word', 'digit_name', 'compound', 'owner_alias')
_CAPWORD = r"[A-Z][a-z]{1,24}(?:-[A-Z][a-z]{1,24})?"


class _RuleV2(M._Rule):
    __slots__ = ('own', 'dlike', 'v2', 'mk', 'never_keep', 'lower_ok')

    @classmethod
    def from_base(cls, base, mk):
        r = cls.__new__(cls)
        for s in M._Rule.__slots__:
            setattr(r, s, getattr(base, s))
        r.mk, r.v2 = mk, None
        r.own = base.index in mk.own_idx
        r.dlike = bool(r.word) and dict_like(r.word, mk.english)
        r.never_keep = False
        r.lower_ok = False
        return r

    @classmethod
    def new(cls, kind, alias, index, regex, mk, v2):
        r = cls.__new__(cls)
        r.kind, r.alias, r.index, r.regex, r.group = kind, alias, index, regex, 0
        r.word = M._guard_word(alias)
        r.english = bool(r.word) and _is_english(r.word, mk.english)
        r.mk, r.v2 = mk, v2
        r.own = index in mk.own_idx
        r.dlike = bool(r.word) and dict_like(r.word, mk.english)
        r.never_keep, r.lower_ok = False, False
        return r

    # -- v2 guards (shared by the TD3 kinds and the v2 kinds)
    def _v2_keep(self, m):
        """True when a dictionary-word alias sits in a frozen collocation, a slug, an index-movement list or upper-case
        text outside company context (outside the firm's own article)."""
        if not self.word or not self.dlike:
            return False
        s = m.string
        left, right = s[max(0, m.start() - 80):m.start()], s[m.end():m.end() + 80]
        w = self.word.replace("'", '')
        if ' ' not in self.alias.strip():
            nx = re.match(r'[ \t]+([A-Za-z]+)', right)
            pv = re.search(r'([A-Za-z]+)[ \t]+$', left)
            if (nx and (w + ' ' + nx.group(1).upper()) in PHRASE_STOP_V2) or \
                    (pv and (pv.group(1).upper() + ' ' + w) in PHRASE_STOP_V2):
                return True
        for cw, lrx, rrx in COLLOCATIONS:
            if cw is not None and cw != w:
                continue
            if (lrx is None or re.search(lrx, left, re.I if cw not in ('OCCIDENTAL', 'ORIENTAL') else 0)) and \
                    (rrx is None or re.search(rrx, right, re.I)):
                return True
        line_left = left.split('\n')[-1]
        if m.group(0).isupper() and (re.match(r'^-[A-Z]', right) or re.match(r'^[ \t]+MARKETS-', right)) and \
                SLUG_RX.match(line_left):
            return True
        if w in INDEX_WORDS and INDEX_MOVE_R.match(right):
            if not self.own or _INDEX_NEAR.search(left[-60:] + ' ' + right[:60].replace(m.group(0), '')):
                return True
        if not self.own and self.kind in ('name', 'common', 'near_name') and m.group(0).isupper():
            lw = re.search(r'([A-Z]+)[ \t]+$', left)
            if lw and lw.group(1) in AUX_LEFT and not self._company(m):
                return True
        return False

    def _company(self, m):
        s = m.string
        left, right = s[max(0, m.start() - 60):m.start()], s[m.end():m.end() + 60]
        if re.search(r"['’][sS]$", m.group(0)) or COMPANY_RIGHT.match(right) or COMPANY_LEFT.search(left):
            return True
        return bool(SUBJECT_LEFT.search(left) and (VERBISH_NEXT.match(right) or COMPANY_RIGHT.match(right)))

    def _td3_guard(self, m):
        """td3_mask.context_guard with the v2 changes: the private fixed-phrase guards and the stop lists do not keep a
        firm's own alias in its own article unless an exact index name follows; a ticker that is also the firm's
        initialism, or used in company context, is not kept in upper-case text."""
        word = self.word
        left = m.string[max(0, m.start() - 60):m.start()]
        right = m.string[m.end():m.end() + 60]
        for lrx, rrx in M.FIXED_PHRASES.get(word, ()):
            if (lrx is None or re.search(lrx, left, re.I)) and (rrx is None or re.search(rrx, right, re.I)):
                if not self.own or word not in INDEX_WORDS or INDEX_CONT_R.match(right):
                    return 'keep'
        if word in M.PARTICLE_VERBS and M.PARTICLE_RX.match(right):
            return 'keep'
        if re.match(r'\s*,\s*' + _STATES_RX + r'(?:-based\b|\s*(?:[,.;)]|$))', right) or \
                re.match(r'\s+' + _STATES_RX + r'-based\b', right):
            return 'drop'
        if self.kind.startswith('ticker') and self.english and len(word) >= 3 and _upper_segment(left, right) and \
                not self.never_keep and not self._company(m):
            return 'keep'
        return None

    def accepted(self, m):
        text = m.group(0)
        k = self.kind
        if self.v2 is not None:
            return self._accept_v2(m)
        if k in ('name', 'near_name'):
            if text == text.lower() and all(w.lower() in GENERIC for w in self.alias.split()):
                return False
            ok = (' ' in self.alias and len(self.alias.replace('.', '')) >= 6) or text != text.lower() or \
                (self.lower_ok and k == 'name')
            if k == 'near_name' and text == text.lower():
                ok = False
        elif k in ('acronym', 'amp'):
            ok = text != text.lower()
        elif k == 'common':
            ok = len(self.alias.replace(' ', '')) <= 3 or text != text.lower()
        else:
            ok = True
        if not ok:
            return False
        if self.word is not None:
            if self._v2_keep(m):
                return False
            g = self._td3_guard(m)
            if g == 'keep':
                return False
            if g == 'drop':
                return 'drop'
        return True

    def _accept_v2(self, m):
        k, text = self.kind, m.group(0)
        s = m.string
        left, right = s[max(0, m.start() - 60):m.start()], s[m.end():m.end() + 60]
        if k == 'split_name':
            return text != text.lower() and not self._v2_keep(m)
        if k in ('digit_name', 'compound'):
            return not self._v2_keep(m) and M.context_guard(self.word or '', m) != 'keep'
        if k == 'core_ctx':
            if self.word in CORE_STOP:
                return False
            lw = re.search(r'([A-Z][A-Za-z&.\-]*)[ \t-]+$', left)
            if lw and lw.group(1) not in FUNCTION_CAPS and \
                    not re.search(r'(?:^|[.!?:;\n*]|' + _SENT + r')[ \t]*' + re.escape(lw.group(1)) + r'[ \t-]+$', left):
                return False                     # part of a longer name ("Zorb United", "USW United")
            nw = re.match(r'[ \t]+([A-Z][A-Za-z]*)', right)
            if nw and not COMPANY_RIGHT.match(right) and not re.search(r"['’][sS]$", text):
                return False                     # followed by a capitalised word: another name ("<Name> Airlines")
            if self._v2_keep(m):
                return False
            return self._company(m)
        if k == 'init2':
            return not re.match(r'^[-–]?\d', right) and not re.search(r'\d[-–]?$', left)
        if k in ('own_word', 'ctx_ticker'):
            w = self.word.replace("'", '').replace('’', '')
            if w.rstrip('S') in MONTH_WORDS or w in MONTH_WORDS:
                if re.match(r'^\.?[ \t]*\d', right):
                    return False
            if w in LEGAL and re.search(r'[A-Z][a-z]+[ \t]+$', left):
                return False
            if INDEX_CONT_R.match(right):
                return False
            nxt = re.match(r'[ \t]+([A-Z]+)\b', right)
            if nxt and (w + ' ' + nxt.group(1)) in PHRASE_STOP:
                return False
            subject = bool(SUBJECT_LEFT.search(left))
            comp = bool(re.search(r"['’][sS]$", text) or COMPANY_RIGHT.match(right) or VERBISH_NEXT.match(right))
            if k == 'ctx_ticker':
                return subject and comp
            strong = text.isupper() and not common_word(w) and not _is_english(w, self.mk.english) and len(w) >= 2
            if strong:
                return True
            if common_word(w) or _is_english(w, self.mk.english):
                return subject and comp
            return subject or comp
        if k == 'own_title':
            if re.match(r'^\.', right) and text in M.STATES:
                return False
            return True
        if k == 'owner_alias':
            if ' ' not in self.alias and dict_like(self.alias, self.mk.english):
                pw = re.search('(' + _CAPWORD + r')[ \t]+$', left)
                if pw and pw.group(1) not in FUNCTION_CAPS and \
                        not re.search(r'(?:^|[.!?:;\n*]|' + _SENT + r')[ \t]*' + _CAPWORD + r'[ \t]+$', left):
                    return False                 # part of another name ("Zorb United"), not after "Of", "The" ...
                nxt = re.match(r'[ \t]+([A-Z][A-Za-z]+)\b', right)
                if nxt and (self.alias.upper() + ' ' + nxt.group(1).upper()) in PHRASE_STOP:
                    return False
            return not self._v2_keep(m)
        return True


class MaskerV2(M.Masker):
    """TD3 v6 Masker plus the v2 rules. own: the role tokens of the firms the article was chosen for."""

    def __init__(self, entries, english=frozenset(), own=()):
        self.own_idx = set()
        own = set(own or ())
        tokens = [t for t, _ in entries]
        self.own_idx = {i for i, t in enumerate(tokens) if t in own}
        super().__init__(entries, english)
        self.v2 = [dict(a.get('v2') or {}) for _, a in entries]
        self.english = english
        owner = {}
        rules = []
        for r in self.rules:
            r2 = _RuleV2.from_base(r, self)
            key = ('ticker', r.alias) if r.kind.startswith('ticker') or r.kind == 'ric' else ('name', r.alias.lower())
            owner.setdefault(key, r.index)
            v2 = self.v2[r.index]
            if r.kind.startswith('ticker') and r.alias.upper() in set(v2.get('initialisms', ())):
                r2.never_keep = True
            if r.kind.startswith('ticker') and r2.word:          # M1 B1: stem >= 3 and common-word guard
                w = r2.word
                r2.english = (_is_english(w, english) and (common_word(w) or (w.endswith('S') and common_word(w[:-1]))))
            if r.kind == 'name' and r.alias in set(v2.get('lower_ok', ())):
                r2.lower_ok = True
            rules.append(r2)

        def add(kind, alias, index, regex, v2tag='v2'):
            key = ('name', alias.lower())
            if key in owner and owner[key] != index:
                self.collisions.append({'kind': kind, 'alias': alias, 'kept': self.tokens[owner[key]],
                                        'dropped': self.tokens[index]})
                return
            owner[key] = index
            rules.append(_RuleV2.new(kind, alias, index, regex, self, v2tag))
        for index, v2 in enumerate(self.v2):
            for w in v2.get('digit', ()):
                add('digit_name', w, index, _compile(variant_regex(w), re.I))
            cps = [w for w in v2.get('compounds', ()) if owner.get(('name', w.lower()), index) == index]
            if cps:                                       # one alternation per firm (performance)
                cps = tuple(sorted(cps, key=lambda x: (-len(x), x)))
                add('compound', cps[0], index, _rx_compound(cps))
            for w in v2.get('splits', ()):
                add('split_name', w, index, _rx_split(w))
            for w in v2.get('core_ctx', ()):
                add('core_ctx', w, index, _rx_core(w))
            for w in v2.get('init2', ()):
                add('init2', w, index, _rx_init2(w))
            own = index in self.own_idx
            for w in v2.get('own_words', ()) if own else ():
                add('own_word', w, index, _rx_own_word(w))
            for w in v2.get('own_title', ()) if own else ():
                add('own_title', w[:1] + w[1:].lower(), index, _rx_own_title(w[:1] + w[1:].lower()))
            for a in v2.get('owner_aliases', ()) if own else ():
                add('owner_alias', a, index, _rx_owner_alias(a))
            if not own:                                     # M2: a stop-list or English ticker heading a brief bullet
                tks = set(entries[index][1].get('tickers', ()))
                for tk in v2.get('own_words', ()):
                    if tk in tks and (tk in ACRONYM_STOP or _is_english(tk, english) or tk in MONTH_WORDS):
                        add('ctx_ticker', tk, index, _rx_ctx_ticker(tk))
        order = {'ric': 0, 'ticker_ric_form': 1, 'ticker_exchange': 2, 'ticker_paren': 3, 'ticker_cashtag': 4,
                 'name': 5, 'common': 5, 'amp': 5, 'near_name': 5, 'digit_name': 5, 'compound': 5, 'owner_alias': 5,
                 'split_name': 5,
                 'acronym': 6, 'ticker_standalone': 7, 'own_word': 7, 'ctx_ticker': 7, 'init2': 7,
                 'ticker_title': 8, 'own_title': 8, 'core_ctx': 9}
        self.rules = sorted(rules, key=lambda r: (-(len(_letters(r.alias)) if r.kind in ('acronym', 'amp')
                                                    else len(r.alias)), order[r.kind], r.alias, r.index))
        # U: own-article firm-code prefixes (subsequences) and extended unit words
        self.hyphen_rx = []
        self.unit_rx2 = []
        for index in range(len(self.tokens)):
            cp = set(self.code_prefixes[index]) | set(self.v2[index].get('codes3', ()))
            if index in self.own_idx:
                cp |= set(self.v2[index].get('own_codes', ()))
            cp = tuple(sorted(cp))
            self.code_prefixes[index] = list(cp)
            self.code_rx[index], self.project_rx[index] = _rx_code(cp) if cp else (None, None)
            up = tuple(sorted({p.upper() for p in entries[index][1].get('unit_prefixes', ()) if p and
                               not _is_english(p, english) and not common_word(p)}))
            self.unit_rx2.append(_rx_unit2(up) if up else None)            # TD3 unit_rx is kept as is
            words = tuple(sorted(self.v2[index].get('hyphen_words', ()), key=len, reverse=True))
            self.hyphen_rx.append(_rx_hyphen(index, words) if words else None)
        self.other_words = {}
        for index, (_, a) in enumerate(entries):
            for v in list(a.get('names', ())) + list(a.get('common', ())):
                if ' ' not in v:
                    self.other_words.setdefault(v.upper().replace('.', ''), set()).add(index)

    # ----------------------------------------------------------------------------------- C adjacency (give-backs)
    def _adjacent(self, rule, m, a, b):
        """A one-word alias absorbs a capitalised word right before it (in running text) or right after it (when the
        next word is lower case or punctuation), so that no part of a longer proper name is left next to the token."""
        if rule.kind not in _ABSORB_KINDS or ' ' in rule.alias.strip() or rule.word is None:
            return a, b, None
        s = m.string
        a2, b2, what = a, b, []
        lo = max(0, a - 80)
        left = s[lo:a]
        right = s[b:b + 80]
        after_lower = bool(re.match(r"(?:['’]s)?(?:[ \t]+[a-z]|[ \t]*[.,;:)!?\"”]|[ \t]*$)", right))
        lm = re.search(r'(?:(?<=[^A-Za-z0-9\-’\'])|^)(' + _CAPWORD + r')([ \t]+|-)$', left)
        if lm and not (lm.start(1) == 0 and lo > 0 and s[lo - 1].isalnum()):
            w0 = lm.group(1)
            prev = left[:lm.start(1)].rstrip()
            pw = re.search(r"([A-Za-z’']+)$", prev)
            running = bool(pw and (pw.group(1).islower() or pw.group(1) in FUNCTION_CAPS)) or \
                prev.endswith((',', '(', ';', '"', '“'))
            start = (prev == '' and lo == 0) or prev.endswith(('.', '!', '?', ':', '\n', '*'))
            if (running or (start and after_lower)) and w0 not in FUNCTION_CAPS and \
                    w0.split('-')[0] not in FUNCTION_CAPS and not self._other_alias(w0, rule.index) and \
                    w0.upper() not in WEEKDAYS | MONTH_WORDS:
                a2 = a - (len(left) - lm.start(1))
                what.append('before:' + w0)
        rm = re.match(r'([ \t]+|-)(' + _CAPWORD + r')(?=[ \t]+[a-z]|[ \t]*[.,;)!?"”])', right)
        if rm and not re.search(r"['’]s$", m.group(0)) and not re.search(_CAPWORD + r'[ \t]+$', left):
            w2 = rm.group(2)
            if w2 not in FUNCTION_CAPS and not self._other_alias(w2, rule.index) and \
                    w2.upper() not in WEEKDAYS | MONTH_WORDS:
                b2 = b + rm.end(2)
                what.append('after:' + w2)
        elif not rule.dlike:                    # a two-digit number completing another name ("Zorbex 66")
            nm = re.match(r'[ \t]+\d{2}(?![\d.,%:])(?![ \t]*(?:%|percent|pct|million|billion|mln|bln|cents?|years?|'
                          r'days?|weeks?|months?|hours?|minutes?|points?|bps|basis|per|stores|employees|people|jobs|'
                          r'units)\b)(?=[ \t]+[a-z]|[ \t]*[.,;:)]|[ \t]*$)', right)
            if nm and not re.search(r"['’]s$", m.group(0)):
                b2 = b + nm.end()
                what.append('after_number:' + nm.group(0).strip())
        return a2, b2, what or None

    def _other_alias(self, word, index):
        owners = self.other_words.get(word.upper().replace('-', ' ').replace('.', ''), set())
        return bool(owners - {index})

    def _apply(self, rule, text):
        """td3_mask.Masker._apply (same matches, same notes) with the v2 adjacency absorption."""
        count = 0
        sentinel = self._sentinel(rule.index)
        out, last = [], 0
        for m in rule.regex.finditer(text):
            if m.start() < last:
                continue
            verdict = rule.accepted(m)
            if not verdict:
                continue
            if verdict == 'drop':
                self._note(rule.index, 'dropped_hq_town', rule.alias, m)
                out.append(text[last:m.start()] + DROP)
                last = m.end()
                continue
            count += 1
            if rule.group:
                self._note(rule.index, rule.kind, rule.alias, m, start=m.end(1))
                out.append(text[last:m.start()] + m.group(1) + sentinel)
                last = m.end()
                continue
            a2, b2, what = self._adjacent(rule, m, m.start(), m.end())
            if a2 < last:
                a2 = m.start()
            if what:
                self._note(rule.index, rule.kind, rule.alias, m, start=a2, end=b2, v2_absorbed=what)
            else:
                self._note(rule.index, rule.kind, rule.alias, m)
            out.append(text[last:a2] + sentinel)
            last = b2
        out.append(text[last:])
        return ''.join(out), count

    # ------------------------------------------------------------------------------------------- E embedded names
    def _token_hit(self, word):
        hit = super()._token_hit(word)
        if hit is not None:
            return hit
        if len(word) < 4 or word == word.lower() or _is_english(word, self.english):
            return None
        up = word.upper()
        for index, v2 in enumerate(self.v2):
            for t in v2.get('camel_terms', ()):
                k = len(t)
                if len(up) > k + 1 and up.startswith(t):
                    nxt = word[k]
                    mixed = any(c.islower() for c in word)
                    if mixed and (nxt.isupper() or nxt.isdigit()) and word[k - 1].islower():
                        return index, 'v2_camel', t
                    if not mixed and word[k:].isalpha() and len(word) - k >= 2 and not common_word(word) and \
                            t in v2.get('camel_caps_terms', ()):
                        return index, 'v2_camel_caps', t
            for t in v2.get('lead_terms', ()):
                k = len(t)
                if len(up) > k and up.startswith(t) and word[k:].isalpha():
                    if (word[:1].isupper() and word[1:].islower()) or word.isupper():
                        return index, 'v2_lead', t
            for t in v2.get('suffix_terms', ()):
                k = len(t)
                if len(up) >= k + 3 and up.endswith(t) and word[:-k].isalpha() and \
                        _is_english(word[:-k], self.english):
                    return index, 'v2_suffix', t
            if index in self.own_idx:
                for t in v2.get('own_lead', ()):
                    k = len(t)
                    if len(up) > k and up.startswith(t) and word[k:].isalpha() and not common_word(word):
                        if (word[:1].isupper() and word[1:].islower()) or word.isupper():
                            return index, 'v2_own_lead', t
        return None

    def _acronym_pass(self, text, counts):
        """TD3 acronym pass, then an active acronym right after a hyphen ('BRIEF-<ACR>'), which TD3's lookbehind
        leaves readable (M4 A4)."""
        text = super()._acronym_pass(text, counts)
        for index, acronyms in sorted(self.active.items()):
            for a in sorted(acronyms, key=lambda x: (-len(x), x)):
                rx = _rx_hyphen_acronym(a)

                def ra(m, index=index, a=a):
                    counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                    self._note(index, 'v2_hyphen_acronym', a, m)
                    return self._sentinel(index)
                text = rx.sub(ra, text)
        return text

    def _unit_pass(self, text, counts):
        """TD3 unit pass, then the v2 unit words and '<prefix> <Word> <legal suffix>' (non-English prefixes)."""
        text = super()._unit_pass(text, counts)
        for index, rx in enumerate(self.unit_rx2):
            if rx is None:
                continue

            def ru(m, index=index):
                if not M.unit_ok(m):
                    return m.group(0)
                counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                self._note(index, 'v2_unit', m.group('pre').upper() + ' ' + m.group('unit'), m)
                return self._sentinel(index)
            text = rx.sub(ru, text)
        return text

    def _token_pass(self, text, counts):
        """td3_mask.Masker._token_pass without its early return (the v2 embedded terms count too)."""
        def repl(m):
            hit = self._token_hit(m.group(1))
            if hit is None:
                return m.group(0)
            index, kind, term = hit
            if M.context_guard(term, m) == 'keep':
                return m.group(0)
            counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
            self._note(index, kind, term, m)
            return self._sentinel(index)
        return M._WORD_TOKEN.sub(repl, text)

    # ------------------------------------------------------------------------------------- U after-token passes
    def _after_token(self, text, counts):
        """Hyphenated own name words, RIC exchange suffix, aircraft model number and a unit-noun legal tail right after
        a token are absorbed into the token (M1 B13, M3 P8, M2, M4 A4, M1 B8)."""
        for index in range(len(self.tokens)):
            s = re.escape(self._sentinel(index))
            hx = self.hyphen_rx[index]
            if hx is not None:
                def rh(m, index=index):
                    w = m.group('w') or m.group('w2')
                    if w == w.lower():
                        return m.group(0)
                    counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                    self._note(index, 'v2_hyphen_own_word', w, m)
                    return self._sentinel(index)
                text = hx.sub(rh, text)
            for kind, rx in (('v2_ric_suffix', _compile(s + r'\.(?:[A-Z]{1,3}|[A-Z]{1,2}[a-z])(?![A-Za-z0-9])')),
                             ('v2_model_after_token', _compile(s + MODEL_AFTER)),
                             ('v2_unit_legal_tail', _compile(s + r'(?:[ \t]+' + _UNIT_NOUN + r'){1,4}[ \t]+' +
                                                             LEGAL_TAIL + r'(?![A-Za-z])'))):
                def rr(m, index=index, kind=kind):
                    counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                    self._note(index, kind, m.group(0)[1:], m)
                    return self._sentinel(index)
                text = rx.sub(rr, text)
        return text

    def _template_pass(self, text, counts):
        """S: the subject of an earnings-summary or preview template in an article chosen for exactly one firm."""
        if len(self.own_idx) != 1:
            return text
        index = next(iter(self.own_idx))
        for rx in TEMPLATE_RX:
            def rt(m, index=index):
                subj = m.group('s')
                counts[self.tokens[index]] = counts.get(self.tokens[index], 0) + 1
                self._note(index, 'v2_template_subject', subj, m)
                return m.group(0)[:m.start('s') - m.start()] + self._sentinel(index) + m.group(0)[m.end('s') - m.start():]
            text = rx.sub(rt, text)
        return text

    def _base_active(self):
        out = {}
        for index, v2 in enumerate(self.v2):
            acr = set(v2.get('always_acronyms', ()))
            if index in self.own_idx:
                acr |= set(v2.get('own_acronyms', ()))
            if acr:
                out[index] = acr
        return out

    def mask_article(self, headline, body):
        base = self._base_active()
        self.active = {k: set(v) for k, v in base.items()}
        h, c1 = self.mask(headline)
        l1 = self.log
        b, c2 = self.mask(body)
        l2 = self.log
        active = {k: set(v) for k, v in base.items()}
        index_of = {t: i for i, t in enumerate(self.tokens)}
        for e in l1 + l2:
            if e['kind'] == 'product_absorb' and e.get('acronyms'):
                active.setdefault(index_of[e['token']], set()).update(e['acronyms'])
        if active != base:
            self.active = active
            h, c1 = self.mask(headline)
            l1 = self.log
            b, c2 = self.mask(body)
            l2 = self.log
        return h, b, c1, c2, l1, l2

    def mask(self, text):
        """td3_mask.Masker.mask with the v2 passes (scrub_v2, template subjects, after-token absorption, v2 dates)."""
        self.log = []
        text = scrub_v2(text or '')
        counts = {}
        text = self._template_pass(text, counts)
        text = self._code_pass(text, counts)
        for rule in self.rules:
            text, k = self._apply(rule, text)
            if k:
                counts[self.tokens[rule.index]] = counts.get(self.tokens[rule.index], 0) + k
        text = self._token_pass(text, counts)
        text = self._unit_pass(text, counts)
        for _ in range(4):
            before = text
            text = self._joint_pass(text, counts)
            text = self._prefix_pass(text, counts)
            text = self._product_pass(self._name_absorb(text), counts)
            text = self._after_token(text, counts)
            if text == before:
                break
        text = self._acronym_pass(text, counts)
        text = self._ric_after_token(text, counts)
        text = self._glued_pass(text, counts)
        text = re.sub(DROP + r'[ \t]*,?[ \t]*', '', text)
        text = M.mask_agency(text)
        text = mask_dates_v2(text)
        for index, token in enumerate(self.tokens):
            text = text.replace(self._sentinel(index), token)
        if M._SENT_OPEN in text or M._SENT_CLOSE in text or DROP in text or re.search(_SENT, text):
            raise ValueError('Unresolved masking sentinel')
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text, counts

    def residual(self, text):
        """TD3 strict residual (every rule, incl. the v2 rules, re-applied) plus the v2 after-token checks."""
        hits = super().residual(text)
        neutral = M._TOKEN.sub(M._NEUTRAL, text)
        for index, token in enumerate(self.tokens):
            anchor = re.escape(token)
            ux = self.unit_rx2[index]
            if ux is not None:
                for mm in ux.finditer(neutral):
                    if M.unit_ok(mm):
                        hits.append({'token': token, 'kind': 'v2_unit', 'alias': mm.group('pre').upper()})
            words = self.v2[index].get('hyphen_words', ())
            if words:
                alt = '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True))
                for mm in re.finditer(anchor + r'(?:-[A-Za-z])?-(' + alt + r')(?![A-Za-z0-9])|(?<![A-Za-z0-9])(' + alt +
                                      r')-(?:[A-Za-z]-)?' + anchor, text, re.I):
                    w = mm.group(1) or mm.group(2)
                    if w != w.lower():
                        hits.append({'token': token, 'kind': 'v2_hyphen_own_word', 'alias': w.upper()})
            if re.search(anchor + r'\.(?:[A-Z]{1,3}|[A-Z]{1,2}[a-z])(?![A-Za-z0-9])', text):
                hits.append({'token': token, 'kind': 'v2_ric_suffix', 'alias': 'suffix'})
            if re.search(anchor + MODEL_AFTER, text):
                hits.append({'token': token, 'kind': 'v2_model_after_token', 'alias': 'model'})
        for index, acronyms in self.active.items():
            for a in acronyms:
                if _rx_hyphen_acronym(a).search(neutral):
                    hits.append({'token': self.tokens[index], 'kind': 'v2_hyphen_acronym', 'alias': a})
        return hits


@functools.lru_cache(maxsize=3000)
def _rx_compound(cps):
    return _compile(_B + '(?:' + '|'.join(re.escape(w) for w in cps) + r")(?:['’]s)?" + _E, re.I)


@functools.lru_cache(maxsize=5000)
def _rx_hyphen_acronym(a):
    return _compile(r'(?<=[A-Za-z][\-–])' + re.escape(a) + r"(?:['’]s)?(?![A-Za-z0-9])")


@functools.lru_cache(maxsize=20000)
def _rx_split(w):
    """A camel-case or English-word split of a one-word name: the parts with a space or hyphen between them."""
    return _compile(_B + r'[ \t\-]+'.join(re.escape(x) for x in w.split()) + r"(?:['’]s)?" + _E, re.I)


@functools.lru_cache(maxsize=20000)
def _rx_core(w):
    forms = sorted({w, w[:1] + w[1:].lower()}, key=len, reverse=True)
    return _compile(_B + '(?:' + '|'.join(re.escape(f) for f in forms) + r")(?:['’]s)?" + _E)


@functools.lru_cache(maxsize=20000)
def _rx_init2(w):
    return _compile(r'(?<![A-Za-z0-9&.])' + re.escape(w) + r"(?:['’]s)?(?![A-Za-z0-9&]|\.[A-Za-z0-9])")


@functools.lru_cache(maxsize=20000)
def _rx_own_word(w):
    forms = {w}
    if w.isalpha() and len(w) >= 3 and w.lower() not in PLACE_WORDS and w.lower() not in GENERIC:
        forms.add(w[:1] + w[1:].lower())          # Title case too, except for place and generic words
    alt = '|'.join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return _compile(r'(?<![A-Za-z0-9&.])(?:' + alt + r")(?:['’][sS])?(?![A-Za-z0-9&]|\.[A-Za-z0-9]|-\d)")


@functools.lru_cache(maxsize=20000)
def _rx_own_title(title):
    return _compile(r'(?<![A-Za-z0-9.$])' + re.escape(title) + r"(?:['’]s)?(?![A-Za-z0-9])")


@functools.lru_cache(maxsize=2000)
def _rx_owner_alias(a):
    rx = (_B + re.escape(a) + r"(?:['’]s)?" + _E) if len(a.replace(' ', '')) <= 3 else         (_B + M._JOIN.join(re.escape(x) for x in a.split()) + r"(?:['’]s)?" + _E)
    return _compile(rx, 0 if a[:1].isupper() else re.I)


@functools.lru_cache(maxsize=5000)
def _rx_ctx_ticker(tk):
    return _compile(r'(?<![A-Za-z0-9&.])' + re.escape(tk) + r"(?:['’][sS])?(?![A-Za-z0-9&]|\.[A-Za-z0-9]|-\d)")


@functools.lru_cache(maxsize=3000)
def _rx_code(cp):
    return _compile(M.code_regex(cp), re.I), _compile(M.project_regex(cp), re.I)


@functools.lru_cache(maxsize=3000)
def _rx_unit2(up):
    return _compile(unit_regex_v2(up), re.I)


@functools.lru_cache(maxsize=3000)
def _rx_hyphen(index, words):
    sen = re.escape(M.Masker._sentinel(index))
    alt = '|'.join(re.escape(w) for w in words)
    return _compile(r'(?P<a>' + sen + r')(?:-[A-Za-z])?-(?P<w>' + alt + r")(?:['’]s)?(?![A-Za-z0-9])|"
                    r'(?<![A-Za-z0-9])(?P<w2>' + alt + r')-(?:[A-Za-z]-)?(?P<b>' + sen + ')', re.I)


def unit_regex_v2(prefixes):
    """td3_mask.unit_regex with the extended unit words, and '<prefix> <Word> <legal suffix>' (M1 B8)."""
    alts = '|'.join(re.escape(p) for p in sorted(prefixes, key=lambda x: (-len(x), x)))
    units = '|'.join(re.escape(u) for u in sorted(set(UNIT_WORDS_V2), key=lambda x: (-len(x), x)))
    return (r'(?<![A-Za-z0-9])(?P<pre>' + alts + r')[ \t]+(?P<unit>(?:' + units + r")(?:['’]s)?|[A-Z][A-Za-z]{2,20}"
            r'(?:[ \t]+[A-Z][A-Za-z]{2,20})?,?[ \t]+' + LEGAL_TAIL + r')(?![A-Za-z0-9])')


def mistag(headline, body):
    """A: True for a mis-tag withdrawal notice."""
    return bool(MISTAG.search((headline or '') + '\n' + (body or '')))
