"""Versioned, outcome-free fact ledger and typed forecast submissions for debugging."""
from copy import deepcopy
from datetime import datetime, timedelta
import math
import re
import unicodedata
from typing import Literal

from dateutil import parser as date_parser
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

from news_agent_batched_runtime_2025 import digest, read, write
from news_network_50_2025.config import OUT
from network50_archive_tools import load
from network50_enhanced_toolkit import EnhancedToolkit


VERSION = 'grounded-v11'

DRIVER_NEWS_MAX_AGE_DAYS = 14
DIRECT_OPERATIONAL_SHOCK = re.compile(
    r'\b(?:factory|plant|facility|production|shipment|delivery|operations?)\b.{0,90}'
    r'\b(?:shut(?:down)?|closed?|halt(?:ed|s)?|stopp(?:ed|age)|disrupt(?:ed|ion)?|'
    r'outage|fire|explosion|strike|recall)\b|'
    r'\b(?:bankrupt(?:cy)?|insolvenc\w*|strike|recall)\b', re.I)

BROKER_HEADLINE = re.compile(
    r'\b(?:BUZZ[- :].*\b(?:PT|price target|target price|analyst|broker|upgrade|downgrade|rating)\b|'
    r'(?:RAISES?|LIFTS?|CUTS?|LOWERS?)\s+(?:.{0,50}\s+)?(?:PT|PRICE TARGET)\b)', re.I)

# Interpretations are hypotheses, not a second place to transcribe source figures.
NUMERIC_RECAP = re.compile(
    r'\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
    r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|'
    r'forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|'
    r'first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|percent|per cent)\b', re.I)
UNPRICED_REACTION = re.compile(
    r'\b(?:market reaction (?:occurs?|will occur|begins?|will begin)|'
    r'(?:not|isn.t|hasn.t|cannot have been) (?:yet )?(?:priced|reflected|absorbed)|'
    r'(?:will|must) (?:first )?be (?:priced|reflected|absorbed))\b', re.I)
UNSOURCED_PATTERN = re.compile(
    r'\b(?:documented|observed|established|proven)\b.{0,100}'
    r'\b(?:co[- ]?mov\w*|correlat\w*|historical? (?:reaction|response|pattern)|'
    r'prior (?:reaction|swing|pattern))\b|\bexplicitly (?:names?|says?|states?|confirms?)\b', re.I)
CERTAIN_WINDOW = re.compile(
    r'\b(?:falls?|occurs?|expected|scheduled|set|due|releases?|takes? place)\b.{0,85}'
    r'\b(?:in|within|during|inside) (?:the )?(?:forecast|same|coming|next) (?:week|window)\b|'
    r'\b(?:in|within|during) (?:the )?forecast window\b|'
    r'\blater in (?:the )?(?:same|coming|forecast) week\b', re.I)


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


class ContractError(ValueError):
    def __init__(self, code, field, detail):
        self.payload = {'code': code, 'field': field, 'detail': detail}
        super().__init__(detail)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SourceFact(StrictModel):
    source_id: StrictStr
    page: StrictInt = Field(ge=0)
    quote: StrictStr = Field(min_length=1)
    category: Literal['reported_fact', 'financing', 'consideration', 'timing', 'exposure', 'broker_commentary'] = 'reported_fact'


class Driver(StrictModel):
    event_key: StrictStr = Field(min_length=1)
    fact_ids: list[StrictStr] = Field(min_length=1)
    interpretation: StrictStr = Field(min_length=1)
    limitations: StrictStr = Field(min_length=1)
    timing_id: StrictStr | None = None
    relation_ids: list[StrictStr] = Field(default_factory=list)
    exposure_fact_ids: list[StrictStr] = Field(default_factory=list)


class OwnSubmission(StrictModel):
    score: float = Field(strict=True, allow_inf_nan=False)
    drivers: list[Driver]
    background_fact_ids: list[StrictStr] = Field(default_factory=list)
    transaction_facts: dict[Literal['announcement', 'financing', 'consideration', 'expected_close'], StrictStr] = Field(default_factory=dict)
    unknowns: list[StrictStr] = Field(default_factory=list)


class NetworkSubmission(StrictModel):
    total_score: float = Field(strict=True, allow_inf_nan=False)
    new_events: list[Driver]
    corroborating_fact_ids: list[StrictStr] = Field(default_factory=list)
    rejected_fact_ids: list[StrictStr] = Field(default_factory=list)
    unknowns: list[StrictStr] = Field(default_factory=list)


WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'


def calendar_resolution(phrase, reference, sessions):
    """Resolve a restricted source phrase, never impute precision to approximate text."""
    text = normalized(phrase).lower().strip(' .,')
    result = {'source_phrase': phrase, 'reference_date': reference.isoformat(),
              'date': None, 'precision': 'unknown', 'in_forecast_window': None}
    if re.search(r'\b(about|around|approximately|middle|mid|later|early|late|expected by)\b', text):
        return {**result, 'precision': 'approximate', 'note': 'No exact date or window membership inferred.'}
    target = None
    weekday = next((w for w in WEEKDAYS if re.search(r'\b'+w+r'\b', text)), None)
    span = re.search(r'\b('+MONTH+r')\.?\s+(\d{1,2})\s*(?:and|to|through|[-–])\s*(\d{1,2})\b', text, re.I)
    if span:
        first = date_parser.parse(f'{span.group(1)} {span.group(2)}',
                                  default=datetime(reference.year, 1, 1)).date()
        last = date_parser.parse(f'{span.group(1)} {span.group(3)}',
                                 default=datetime(reference.year, 1, 1)).date()
        if last < first:
            raise ContractError('invalid_date_range', 'phrase', 'End of same-month date range precedes start.')
        if weekday and first.weekday() != WEEKDAYS.index(weekday):
            raise ContractError('weekday_date_conflict', 'phrase', 'Weekday disagrees with first date in range.')
        start_window, end_window = sessions[0], sessions[-1]
        in_window = (True if start_window <= first.isoformat() and last.isoformat() <= end_window
                     else False if last.isoformat() < start_window or first.isoformat() > end_window
                     else None)
        return {**result, 'start_date': first.isoformat(), 'end_date': last.isoformat(),
                'precision': 'range', 'in_forecast_window': in_window,
                'note': 'The source describes a range; no single event day or decision day inferred.'}
    explicit = re.search(r'\b\d{4}-\d{2}-\d{2}\b', text)
    month_day = re.search(r'\b'+MONTH+r'\.?\s+\d{1,2}(?:,?\s+\d{4})?\b', text, re.I)
    if explicit or month_day:
        literal = (explicit or month_day).group()
        target = date_parser.parse(literal, default=datetime(reference.year, 1, 1)).date()
        if weekday and target.weekday() != WEEKDAYS.index(weekday):
            raise ContractError('weekday_date_conflict', 'phrase', f'{literal} is {target.strftime("%A")}, not {weekday}.')
    elif weekday and len(re.findall(r'\b(?:'+ '|'.join(WEEKDAYS) + r')\b', text)) == 1:
        offset = WEEKDAYS.index(weekday) - reference.weekday()
        if 'next week' in text:
            offset += 7
        elif 'last week' in text or re.search(r'\blast\s+'+weekday+r'\b', text):
            offset = offset if offset < 0 else offset-7
        elif re.search(r'\bnext\s+'+weekday+r'\b', text):
            # "Next Wednesday" can mean this coming or the following Wednesday.
            return {**result, 'note': 'Ambiguous next-weekday expression; retain unknown date.'}
        elif 'this week' not in text and not re.search(r'\bthis\s+'+weekday+r'\b', text):
            if offset < 0:
                return {**result, 'note': 'Bare weekday precedes source dateline; past or next week is ambiguous.'}
            offset %= 7
        target = reference + timedelta(days=offset)
        result['note'] = 'Weekday interpreted relative to the source dateline, not the forecast origin.'
    elif text in {'today', 'tomorrow', 'yesterday'}:
        target = reference + timedelta(days={'today':0, 'tomorrow':1, 'yesterday':-1}[text])
    if target:
        result.update(date=target.isoformat(), precision='day',
                      in_forecast_window=sessions[0] <= target.isoformat() <= sessions[-1],
                      is_forecast_trading_session=target.isoformat() in sessions)
    return result


class GroundedToolkit(EnhancedToolkit):
    def __init__(self, capability, archive, anchor, trace_path=None):
        super().__init__(capability, archive, anchor, trace_path)
        self.facts, self.dates, self.relations = {}, {}, {}
        self.own = deepcopy(capability.get('grounded_own'))
        if capability['network'] and not self.own:
            raise ValueError('NET requires a frozen grounded OWN submission')
        cutoff = datetime.fromisoformat(capability['case']['asof'])
        for event in capability.get('calendar', {}).get('events', []):
            if (event.get('ticker') != capability['case']['ticker'] or
                    event.get('date_status') != 'confirmed_in_window' or
                    event.get('in_window') is not True or
                    not event.get('receipt_ids') or
                    datetime.fromisoformat(event['available_at']) >= cutoff):
                continue
            fact = {'source_id': 'CALENDAR:'+event['receipt_ids'][0], 'page': 0,
                    'quote': event['source_text'], 'category': 'timing',
                    'headline': 'Source-backed Reuters earnings calendar',
                    'available_at': event['available_at'],
                    'story_id': 'CALENDAR:'+event['receipt_ids'][0], 'scope': 'calendar',
                    'minutes_before_decision': round((cutoff-datetime.fromisoformat(
                        event['available_at'])).total_seconds()/60, 1),
                    'reaction_status': 'scheduled event; market surprise remains unknown'}
            fid = digest(fact)
            self.facts[fid] = fact
            receipt = {'source': 'verified_calendar', 'fact_id': fid,
                       'source_text': event['source_text'], 'date': event['event_date'],
                       'time_text': event.get('time_text'), 'available_at': event['available_at'],
                       'precision': 'day', 'in_forecast_window': True,
                       'calendar_receipt_ids': event['receipt_ids'],
                       'ticker': event['ticker']}
            self.dates[digest(receipt)] = receipt

    def inventory(self, group: Literal['all','own_common','direct_economic','comention_without_direct','higher_hop_only','unlinked'] = 'all',
                  offset: int = 0, limit: int = 10) -> dict:
        """List shortlisted source IDs and graph leads. Choose own_common for OWN; archive search covers omitted sources."""
        return super().inventory(group, offset, limit)

    def search_archive(self, query: str, scope: Literal['own','common','network'] = 'own', limit: int = 8) -> dict:
        """Search the full eligible headline catalogue. Use focused conjunctive terms; numerical limit is 1..20. Prefer full UPDATE articles for missing transaction terms."""
        return super().search_archive(query, scope, limit)

    def inspect_archive(self, source_id: str, page: int = 0) -> dict:
        """Read a verified archive page; page is a zero-based integer, not a string. Read the returned page_count before requesting later pages."""
        return super().inspect_archive(source_id, page)

    def inspect_event(self, source_id: str, page: int = 0) -> dict:
        """Read a short-list source or any eligible archive result with the same source ID and page arguments."""
        if source_id in self.docs:
            return super().inspect_event(source_id, page)
        return self.inspect_archive(source_id, page)

    def market_state(self):
        result = super().market_state()
        result['calendar_timing_receipts'] = [
            {'timing_id': did, **receipt} for did, receipt in self.dates.items()
            if receipt.get('source') == 'verified_calendar']
        result['interpretation_limits'] = (
            'The anchor contains statistical controls, not the semantic content of particular articles. '
            'News counts do not establish that a specific news mechanism is priced or resolved. '
            'Unknown earnings coverage is not a confirmed absence of earnings. '
            'A recent M&A-news control is not evidence of cash consideration or a pending closing; '
            'source-bound transaction fields remain separate from the frozen anchor.')
        return result

    def _source(self, sid):
        row = self.archive.records.get(sid, self.docs.get(sid))
        if row is None:
            raise ContractError('unknown_source', 'source_id', sid)
        return row

    def _fact(self, fid):
        if fid in self.facts:
            return self.facts[fid]
        if self.own and fid in self.own['facts']:
            return self.own['facts'][fid]
        raise ContractError('unknown_fact', 'fact_id', f'{fid}: call record_facts on an inspected quote first.')

    def _mentions_target(self, fact):
        sid = fact['source_id']
        source = self._source(sid)
        passage = source['headline']+' '+fact['quote']
        company = re.sub(r'(?:\s+(?:INC|CORP|PLC|CO|COM|NEW))+\s*$', '',
                         self.data['case']['company'], flags=re.I).strip()
        ticker = self.data['case']['ticker']
        return bool((company and re.search(r'\b'+re.escape(company)+r'\b', passage, re.I)) or
                    re.search(r'<'+re.escape(ticker)+r'\.[A-Z]+>', passage) or
                    (len(ticker) >= 3 and re.search(r'\b'+re.escape(ticker)+r'\b', passage)))

    def record_facts(self, facts: list[dict]) -> dict:
        """Bind exact quotes to inspected source/page pairs. Returns fact IDs; this checks provenance, not truth or economic materiality."""
        results = []
        for index, item in enumerate(facts):
            try:
                fact = SourceFact.model_validate(item).model_dump()
                sid, page = fact['source_id'], str(fact['page'])
                passage = self.reads.get(sid, {}).get(page)
                if passage is None:
                    raise ContractError('unread_source', f'facts[{index}]', f'{sid} page {page}: inspect this page before binding a quote.')
                if normalized(fact['quote']) not in normalized(passage):
                    raise ContractError('quote_not_on_page', f'facts[{index}].quote', f'Quote not found verbatim in {sid} page {page}; cite the correct page, not a paraphrase.')
                row = self._source(sid)
                broker = bool(BROKER_HEADLINE.search(row['headline']) or
                              re.search(r'\b(?:target price|price target|PT)\b', fact['quote'], re.I))
                if broker:
                    fact['category'] = 'broker_commentary'
                fact.update(headline=row['headline'], available_at=row['available_at'],
                            story_id=row.get('story_id', sid),
                            scope=row.get('scope', 'network' if self.data['network'] else
                                          'own' if self.data['case']['permno'] in row.get('ids', []) else 'common'))
                fact['minutes_before_decision'] = round(
                    (datetime.fromisoformat(self.data['case']['asof']) -
                     datetime.fromisoformat(row['available_at'])).total_seconds()/60, 1)
                fact['reaction_status'] = 'unknown; pre-decision publication may already be priced at the decision close'
                fid = digest(fact)
                self.facts[fid] = fact
                results.append({'fact_id': fid, **fact, 'provenance_checked': True,
                                'semantic_entailment_verified': False})
            except (ContractError, ValidationError) as exc:
                results.append({'index': index, 'error': self._error(exc)})
        return {'facts': results}

    def resolve_event_date(self, fact_id: str, phrase: str) -> dict:
        """Compute date/window membership from an exact phrase in a bound fact. Approximate/ambiguous wording stays uncertain."""
        fact = self._fact(fact_id)
        if normalized(phrase).lower() not in normalized(fact['quote']).lower():
            raise ContractError('phrase_not_in_fact', 'phrase', 'Copy the timing phrase from the bound quote; do not supply an invented date.')
        ref = datetime.fromisoformat(fact['available_at']).date()
        page_zero = self.reads.get(fact['source_id'], {}).get('0', '')
        dateline = re.search(r'\b('+MONTH+r'\.?\s+\d{1,2})\s+\(Reuters\)', page_zero, re.I)
        basis = 'source availability date; no explicit Reuters dateline found'
        if dateline:
            parsed = date_parser.parse(dateline.group(1), default=datetime(ref.year, 1, 1)).date()
            if parsed > ref and ref.month == 1 and parsed.month == 12:
                parsed = parsed.replace(year=ref.year-1)
            ref, basis = parsed, 'Reuters dateline on inspected first page'
        result = calendar_resolution(phrase, ref, self.data['case']['next_sessions'])
        result.update(fact_id=fact_id, reference_basis=basis)
        did = digest(result)
        self.dates[did] = result
        return {'timing_id': did, **result}

    def exposure_paths(self, hops: int = 1, via: str = '', query: str = '', offset: int = 0) -> dict:
        """Look up typed paths with stable relation IDs. A relation is a retrieval lead, not verified transmission."""
        result = super().exposure_paths(hops, via, query, offset)
        paths = []
        for path in result['paths']:
            clean = deepcopy(path)
            for step in clean.get('steps', []):
                step.pop('revenue_pct', None)
            rid = digest(clean)
            self.relations[rid] = clean
            paths.append({'relation_id': rid, **clean})
        return {**result, 'paths': paths}

    def network_leads(self, query: str = '', relation_id: str = '', offset: int = 0, limit: int = 12) -> dict:
        """Use empty query first for balanced event discovery; optional query matches any term, ranked by overlap. Graph leads are not proof of transmission."""
        if not self.data['network']:
            raise ContractError('wrong_arm', 'network_leads', 'Only NET can search related-company news.')
        if relation_id and relation_id not in self.relations:
            raise ContractError('unknown_relation', 'relation_id', 'Obtain this ID from exposure_paths first.')
        permnos = self.relations[relation_id]['permnos'] if relation_id else None
        return self.archive.prioritized(query, permnos, offset, limit)

    def fact_ledger(self) -> dict:
        """Read bound facts, computed event dates, and frozen OWN event keys. OWN facts cannot be rewritten by NET."""
        return {'facts': self.facts, 'dates': self.dates,
                'frozen_own': self.own, 'relations': self.relations,
                'note': 'Quotes are observations; interpretations are model hypotheses, not verified facts.'}

    def risk_check(self, score: float) -> dict:
        """Compute authoritative risk units and horizon. Final summaries are host-rendered, not model-written."""
        values = super().risk_check(score)
        values['geometric_rms_multiple_vs_anchor'] = math.exp(score/2)
        values['summary'] = (
            f"Forecast through {values['forecast_end']}: {values['variance_multiple_vs_anchor']:.4f} times anchor variance; "
            f"{values['geometric_rms_multiple_vs_anchor']:.4f} times anchor geometric daily RMS "
            f"({values['forecast_daily_geometric_rms_percent']:.4f} percent).")
        return values

    def _transaction(self, fields):
        bound = {name:self._fact(fid) for name, fid in fields.items()}
        quote = normalized(bound.get('consideration', {}).get('quote', '')).lower()
        explicit = bool(re.search(r'\ball[- ]cash\b|\bper share in cash\b|\bcash (?:consideration|payment)\b|\bin cash (?:for|per) (?:each )?share\b', quote))
        uncertain = bool(re.search(r'\b(?:not|no|whether|may|might|could|reportedly|rumou?r|considering)\b', quote))
        return {'fields': {k:{'fact_id':fields[k], **v} for k,v in bound.items()},
                'cash_consideration': 'explicit_source_statement' if explicit and not uncertain else 'unknown',
                'note': 'Cash-on-hand funding alone does not establish shareholder payment type. Fields are quoted, not a reconstructed deal database.'}

    def _drivers(self, items, transaction, network):
        output = []
        own_sources = {f['source_id'] for f in self.own['facts'].values()} if self.own else set()
        own_stories = {f['story_id'] for f in self.own['facts'].values()} if self.own else set()
        own_keys = {d['event_key'] for d in self.own['drivers']} if self.own else set()
        seen = set()
        for index, driver in enumerate(items):
            field = f'{"new_events" if network else "drivers"}[{index}]'
            facts = [self._fact(fid) for fid in driver['fact_ids']]
            if driver['event_key'] in seen:
                raise ContractError('duplicate_event', field+'.event_key', 'Group the same event once with multiple supporting facts.')
            seen.add(driver['event_key'])
            if all(f['category'] == 'broker_commentary' for f in facts):
                raise ContractError('broker_background', field+'.fact_ids', 'Only broker commentary supports this driver; move it to background/rejected facts.')
            did = driver['timing_id']
            timing = None
            if did is not None and did not in self.dates and did not in (self.own or {}).get('dates', {}):
                raise ContractError('unknown_timing', field+'.timing_id', 'Use a timing_id returned by resolve_event_date, or null for unknown timing.')
            if did is not None:
                timing = self.dates.get(did, (self.own or {}).get('dates', {}).get(did))
                if timing.get('source') == 'verified_calendar' and not re.search(
                        r'\b(?:earnings|results|quarter(?:ly)?|reporting|release)\b',
                        driver['event_key']+' '+driver['interpretation'], re.I):
                    raise ContractError('calendar_receipt_event_mismatch', field+'.timing_id',
                                        'A target earnings-calendar receipt cannot certify a different event.')
                cited = set(driver['fact_ids'] + driver['exposure_fact_ids'])
                cited.update(v.get('fact_id') for v in transaction.get('fields', {}).values())
                if timing['fact_id'] not in cited:
                    raise ContractError('unlinked_timing', field+'.timing_id',
                                        'The computed date must come from a cited driver or transaction fact.')
            elif re.search(r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next week|this week)\b',
                           driver['interpretation'], re.I):
                raise ContractError('timing_receipt_required', field+'.timing_id',
                                    'A specific weekday/week claim needs resolve_event_date on a cited fact.')
            if UNPRICED_REACTION.search(driver['interpretation']):
                raise ContractError('unverified_market_reaction', field+'.interpretation',
                                    'Pre-decision articles may already be priced at the decision close. Do not assert that initial market reaction must occur in the future window; discuss only uncertain persistence or new information.')
            if CERTAIN_WINDOW.search(driver['interpretation']) and (
                    timing is None or timing['in_forecast_window'] is not True):
                raise ContractError('unverified_forecast_window', field+'.timing_id',
                                    'A definite scheduled event in the forecast window needs a cited timing receipt whose in_forecast_window is true. Approximate or unresolved timing permits only a conditional possibility.')
            if transaction['cash_consideration'] == 'unknown' and re.search(
                    r'\ball[- ]cash\b|\bin cash\b|\bcash (?:deal|offer|acquisition|consideration)\b', driver['interpretation'], re.I):
                raise ContractError('unsupported_cash_interpretation', field+'.interpretation',
                                    'Payment type is unknown. Analyze the documented acquisition without asserting cash consideration.')
            if any(NUMERIC_RECAP.search(driver[name]) for name in ('interpretation', 'limitations')):
                raise ContractError('numerical_recap_not_allowed', field+'.interpretation',
                                    'Keep interpretation and limitations qualitative, with no digit or spelled-out number. Exact figures belong in cited quotes; dates and risk units belong in computed tool fields.')
            if UNSOURCED_PATTERN.search(driver['interpretation']):
                raise ContractError('unsourced_attribution_or_pattern', field+'.interpretation',
                                    'Do not attribute reporter narration to a company or assert an empirical price pattern absent source evidence. Keep the exact source quote in fact_ids and state only a conditional mechanism here.')
            if re.search(r'\b(?:eliminat\w*|remov\w*|no|without)\b.{0,80}\bfinanc\w*\s+(?:contingen\w*|risk|uncertainty|condition|dependen\w*)',
                         driver['interpretation'], re.I):
                cited = facts + [v for v in transaction.get('fields', {}).values()]
                if not any(re.search(r'\b(?:no financing condition|not subject to (?:any )?financing|no financing contingency)\b',
                                     f['quote'], re.I) for f in cited):
                    raise ContractError('unsupported_financing_guarantee', field+'.interpretation',
                                        'Cash-on-hand funding alone does not establish absence of financing conditions.')
            cutoff = datetime.fromisoformat(self.data['case']['asof'])
            fresh = any((cutoff-datetime.fromisoformat(f['available_at'])) <=
                        timedelta(days=DRIVER_NEWS_MAX_AGE_DAYS) for f in facts)
            scheduled = timing is not None and timing.get('in_forecast_window') is True
            same_day_earnings = (not network and timing is not None and
                                 timing.get('date') == cutoff.date().isoformat() and
                                 timing.get('precision') == 'day' and
                                 re.search(r'\b(?:earnings|results|quarter(?:ly)?)\b',
                                           driver['event_key']+' '+driver['interpretation'], re.I) and
                                 any(f['category'] == 'timing' for f in facts))
            if same_day_earnings and not re.search(
                    r'\b(?:release time|publication time|before or after (?:the )?(?:close|cutoff)|'
                    r'timing (?:is |remains )?(?:unknown|unconfirmed)|after.hours status)\b',
                    driver['limitations'], re.I):
                raise ContractError('same_day_release_time_unknown', field+'.limitations',
                                    'A dated release on the decision day has unknown pre/post-close timing. State that uncertainty explicitly; do not claim a new future-window event.')
            announcement = transaction.get('fields', {}).get('announcement', {})
            live_deal = (not network and self.data.get('controls', {}).get('recent_ma_news_31d') == 1 and
                         any(f['source_id'] == announcement.get('source_id') for f in facts))
            if not (fresh or scheduled or same_day_earnings or live_deal):
                raise ContractError('stale_event', field+'.fact_ids',
                                    'An old report alone cannot be a new risk driver. Move it to background, or cite a recent update, a computed event inside the forecast window, or the documented recent deal state.')
            if not network and any(f['scope'] == 'own' for f in facts) and not any(
                    f['scope'] == 'own' and self._mentions_target(f) for f in facts) and not any(
                    f['scope'] == 'common' for f in facts):
                raise ContractError('unverified_own_tag', field+'.fact_ids',
                                    'A Reuters firm tag alone does not establish target exposure. Cite an inspected source actually naming the target, or move the broad story to background.')
            if network:
                duplicate = next((f for f in facts if f['source_id'] in own_sources or f['story_id'] in own_stories), None)
                if duplicate or driver['event_key'] in own_keys:
                    raise ContractError('inherited_event_not_new', field, f"{duplicate['source_id'] if duplicate else driver['event_key']}: use corroborating_fact_ids; do not copy OWN drivers into new_events.")
                if not any(f['scope'] == 'network' for f in facts):
                    raise ContractError('not_network_evidence', field, 'A new network event needs a newly inspected related-company fact.')
                if not any(f['scope'] == 'network' and f['category'] != 'broker_commentary' for f in facts):
                    raise ContractError('broker_network_evidence', field+'.fact_ids',
                                        'A network risk driver needs non-broker reporting about the related company.')
                paths = []
                for rid in driver['relation_ids']:
                    if rid not in self.relations:
                        raise ContractError('unknown_relation', field+'.relation_ids', f'{rid}: use exposure_paths to obtain relation IDs.')
                    paths.append(self.relations[rid])
                if not paths:
                    raise ContractError('missing_relation', field+'.relation_ids', 'A new network driver needs a graph path and a qualified mechanism.')
                ids = {p for f in facts for p in self._source(f['source_id']).get('ids', [])}
                if not any(ids & set(p['permnos']) for p in paths):
                    raise ContractError('path_source_mismatch', field, 'The cited graph paths do not reach an entity in the cited news.')
                if all(p.get('type') == 'comention' for p in paths) and not driver['exposure_fact_ids']:
                    raise ContractError('comention_only', field, 'Co-mention alone is not economic exposure; cite exposure facts or keep this lead as background.')
                direct_supply = any(p['hops'] == 1 and any(
                    s.get('neighbor_role') in {'supplier', 'customer'} for s in p.get('steps', [])) for p in paths)
                weak_relation = any(any(s.get('neighbor_role') == 'competitor' for s in p.get('steps', [])) or
                                    p.get('type') == 'comention' for p in paths)
                if weak_relation and not direct_supply:
                    exposure = [self._fact(fid) for fid in driver['exposure_fact_ids']]
                    target = self.data['case']['company']
                    ticker = self.data['case']['ticker']
                    if not any(f['category'] != 'broker_commentary' and
                               (target.lower() in f['quote'].lower() or
                                re.search(r'\b'+re.escape(ticker)+r'\b', f['quote'], re.I)) for f in exposure):
                        raise ContractError('unsupported_competitor_path', field+'.exposure_fact_ids',
                                            'Competitor/co-mention paths need a non-broker cited fact explicitly connecting the target; otherwise keep as a lead.')
                if all(p['hops'] > 1 for p in paths) and not any(
                        self._mentions_target(self._fact(fid)) for fid in driver['exposure_fact_ids']):
                    raise ContractError('unverified_two_hop', field+'.exposure_fact_ids',
                                        'A two-hop topology alone does not establish transmission through the intermediate firm. Keep it as a lead unless a cited source explicitly connects the target.')
                if direct_supply and not any(self._mentions_target(f) for f in facts) and not any(
                        DIRECT_OPERATIONAL_SHOCK.search(f['quote']+' '+f['headline']) for f in facts):
                    raise ContractError('unverified_partner_transmission', field+'.fact_ids',
                                        'A direct graph edge alone does not make generic partner earnings or tariff anecdotes a target-specific shock. Cite a source naming the target or a concrete operational disruption; otherwise move the lead to background.')
            for fid in driver['exposure_fact_ids']:
                self._fact(fid)
            output.append({**driver, 'mechanism_status': 'model_inference_not_independently_verified'})
        return output

    def _finish(self, score, drivers, extra):
        required = {'market_state', 'inventory', 'search_archive'}
        used = {c['tool'] for c in self.trace if c['ok']}
        if not required <= used:
            raise ContractError('missing_tools', 'submission', f'Use {sorted(required-used)} before submitting.')
        if self.data['network'] and not any(c['tool']=='network_leads' and c['ok'] and
                                            not c['args'].get('query') for c in self.trace):
            raise ContractError('missing_broad_network_discovery', 'submission',
                                'Call network_leads with an empty query first for balanced direct-partner discovery; a zero-hit narrow search alone does not establish no relevant network news.')
        if not self.data['network'] and not any(c['tool']=='search_archive' and c['ok'] and c['args'].get('scope')=='common' for c in self.trace):
            raise ContractError('missing_common_search', 'submission', 'Review common macro/sector sources as well as own news.')
        computed = self.risk_check(score)
        forecast = {'version': VERSION, 'case_id': self.data['case']['case_id'], 'score':score,
                    'drivers':drivers, 'facts':deepcopy(self.facts), 'dates':deepcopy(self.dates),
                    'computed':computed, 'summary':computed['summary'], **extra}
        receipt = digest(forecast)
        self.submissions.append({'receipt':receipt, 'forecast':forecast})
        return {'accepted':True, 'receipt':receipt, 'summary':computed['summary'],
                'instruction':'This host-rendered forecast is authoritative. Return the receipt only.'}

    def submit_own(self, forecast: dict) -> dict:
        """OWN only: score, drivers, background_fact_ids, transaction_facts, unknowns. All factual fields reference bound fact IDs; no free-form factual recap or date/units overrides."""
        if self.data['network']:
            raise ContractError('wrong_arm', 'submission', 'NET must use submit_network; frozen OWN is read-only.')
        value = OwnSubmission.model_validate(forecast).model_dump()
        transaction = self._transaction(value['transaction_facts'])
        drivers = self._drivers(value['drivers'], transaction, False)
        if abs(value['score']) > math.log(2) and not any(
                any(self._fact(fid)['scope'] == 'own' and
                    self._mentions_target(self._fact(fid)) for fid in driver['fact_ids']) and
                (driver['timing_id'] is None or
                 self.dates.get(driver['timing_id'], {}).get('in_forecast_window') is True)
                for driver in drivers):
            raise ContractError('large_score_without_target_catalyst', 'score',
                                'A more-than-twofold variance adjustment needs a fresh, cited target-specific driver or a confirmed in-window target event. Common macro stories, background, unknowns, and same-day earnings with unresolved release time do not qualify.')
        for fid in value['background_fact_ids']:
            self._fact(fid)
        return self._finish(value['score'], drivers, {'arm':'OWN', 'transaction':transaction,
                            'background_fact_ids':value['background_fact_ids'], 'unknowns':value['unknowns']})

    def submit_network(self, supplement: dict) -> dict:
        """NET only: total_score, new_events, corroborating_fact_ids, rejected_fact_ids, unknowns. Never copy OWN drivers. Empty new_events requires the identical OWN score; host preserves OWN facts and terms."""
        if not self.data['network']:
            raise ContractError('wrong_arm', 'submission', 'OWN has no network supplement.')
        value = NetworkSubmission.model_validate(supplement).model_dump()
        drivers = self._drivers(value['new_events'], self.own['transaction'], True)
        for fid in value['corroborating_fact_ids'] + value['rejected_fact_ids']:
            self._fact(fid)
        if not drivers and value['total_score'] != self.own['score']:
            raise ContractError('empty_network_score_change', 'total_score', f"No new drivers: use exactly {self.own['score']}; no redraw of OWN.")
        return self._finish(value['total_score'], drivers, {'arm':'NET', 'own_receipt':digest(self.own),
                            'frozen_own':deepcopy(self.own), 'transaction':deepcopy(self.own['transaction']),
                            'corroborating_fact_ids':value['corroborating_fact_ids'],
                            'rejected_fact_ids':value['rejected_fact_ids'], 'unknowns':value['unknowns']})

    @staticmethod
    def _error(exc):
        if isinstance(exc, ContractError):
            return exc.payload
        if isinstance(exc, ValidationError):
            return {'code':'schema_error', 'fields':exc.errors(include_url=False, include_input=False, include_context=False)}
        return {'code':'invalid_request', 'detail':str(exc)}

    def call(self, tool, **args):
        if tool not in {'record_facts', 'resolve_event_date', 'fact_ledger', 'network_leads', 'submit_own', 'submit_network'}:
            return super().call(tool, **args)
        try:
            result = getattr(self, tool)(**args)
            ok = True
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            result, ok = {'error':self._error(exc)}, False
        self.trace.append({'tool':tool, 'args':args, 'ok':ok, 'result':result})
        self.save_trace()
        return result

    def save_trace(self):
        if self.trace_path:
            write(self.trace_path, {'version':VERSION, 'capability_hash':digest(self.data), 'calls':self.trace,
                'reads':self.reads, 'inspected_entities':sorted(self.inspected_entities),
                'facts':self.facts, 'dates':self.dates, 'relations':self.relations,
                'submissions':self.submissions, 'labels_accessible':False})


COMMON_TOOLS = ['inventory', 'market_state', 'search_events', 'inspect_event', 'inspect_events',
                'event_logbook', 'search_archive', 'inspect_archive', 'risk_check',
                'record_facts', 'resolve_event_date', 'fact_ledger']


def serve(capfile, tracefile):
    from functools import wraps
    from mcp.server.mcpserver import MCPServer
    cap = read(capfile)
    archive, _ = load(cap['case']['case_id'], cap['network'])
    kit = GroundedToolkit(cap, archive, read(OUT/'anchor.json')['anchors'][cap['case']['case_id']], tracefile)
    server = MCPServer('news')
    names = COMMON_TOOLS + (['exposure_paths', 'network_leads', 'submit_network'] if cap['network'] else ['submit_own'])
    def register(name):
        method = getattr(kit, name)
        @wraps(method)
        def wrapped(*args, **kwargs):
            if args:
                import inspect
                kwargs = inspect.signature(method).bind(*args, **kwargs).arguments
            return kit.call(name, **kwargs)
        server.tool(name=name)(wrapped)
    for name in names:
        register(name)
    try:
        server.run(transport='stdio')
    finally:
        archive.close()


if __name__ == '__main__':
    import sys
    serve(sys.argv[1], sys.argv[2])
