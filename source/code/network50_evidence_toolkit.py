"""Evidence checks for the paused-run failure cases, separate from old tool versions."""
from copy import deepcopy
from datetime import datetime, date, timedelta
from decimal import Decimal
from functools import wraps
import math
import re
from typing import Literal
from zoneinfo import ZoneInfo

from news_agent_batched_runtime_2025 import digest, read, write
from network50_archive_tools import load
from network50_grounded_toolkit import GroundedToolkit, ContractError, COMMON_TOOLS, normalized


VERSION = 'evidence-v1'
EMPIRICAL_REACTION = re.compile(
    r'\b(?:historically|typically|usually|routinely|tends? to|known for|history of)\b'
    r'.{0,160}\b(?:reactions?|moves?|returns?|swings?|volatility|jump\w*|plung\w*|rall\w*)\b|'
    r'\b(?:historical|typical|usual)\b.{0,100}\b(?:reactions?|responses?|swings?)\b', re.I)
PRICE_PINNING = re.compile(
    r'\b(?:price|stock)\b.{0,65}\b(?:anchor\w*|pin\w*|fixed|capped)\b|'
    r'\bcompress\w*\b.{0,40}\bvolatility\b', re.I)
LOW_DEAL_RISK = re.compile(
    r'\b(?:deal[- ]break|break|termination|regulatory)\s+risk\b.{0,65}'
    r'\b(?:is low|low[- ]probability|negligible|minimal)\b|'
    r'\blow[- ]probability\b.{0,45}\bdeal[- ]break\b', re.I)
CHANNEL_SHOCK = {
    'supplier_input': re.compile(
        r'\b(?:shut\w*|halt\w*|outage|fire|strike|recall|disrupt\w*|shortage|'
        r'stopp\w*|suspend\w*|bankrupt\w*)\b', re.I),
    'customer_output': re.compile(
        r'\b(?:production|deliveries|shipments|orders|output|demand)\b.{0,80}'
        r'\b(?:halt\w*|cut\w*|reduc\w*|delay\w*|declin\w*|fall\w*|stop\w*)\b|'
        r'\b(?:halt\w*|cut\w*|reduc\w*|delay\w*|stop\w*)\b.{0,80}'
        r'\b(?:production|deliveries|shipments|orders|output)\b', re.I),
    'competition': re.compile(
        r'\b(?:demand|orders?|guidance|sales|revenue|market share|launch\w*|'
        r'exports?|licen[cs]\w*|substitut\w*|compet\w*)\b', re.I),
}
CHANNEL_ROLES = {'supplier_input': 'supplier', 'customer_output': 'customer',
                 'competition': 'competitor'}
MITIGATION = re.compile(
    r'\b(?:fully resolved|unaffected|not affected|no impact|returned to normal|'
    r'production (?:has )?resumed)\b', re.I)
GUIDANCE = '''
The tools and prompt must share forecast_anchor exactly. Call financial_history for
observed returns and prices. That tool has no verified earnings-event labels, so its
ordinary price history cannot establish typical earnings reactions. An empirical
reaction assertion needs the exact assertion in an inspected source; otherwise use
a conditional mechanism and keep the historical attribution unknown.
Call deal_terms before assuming a fixed offer anchors a stock price. Cash-on-hand
funding is not shareholder cash consideration. An observed discount to a quoted offer
does not establish a low probability of deal failure or future volatility compression.
Before a NET addition, call check_network_channel with a typed relation, fresh event
facts, a separate target-exposure fact and a specific affected item quoted in both.
The disruption article need not name the target. Prior target-product evidence can
supply that link. A graph edge alone cannot identify which product is affected.
Insufficient or mitigated channels remain retrieval leads and cannot change the score.
A conditional channel receipt checks cited prerequisites, not economic materiality
or the direction/magnitude of future market risk. Preserve those uncertainties.
'''


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def financial_snapshot(case):
    history = case.get('stock_history', {})
    dates, returns, prices = (history.get(k, []) for k in
                              ('dates', 'total_return_decimal', 'price_usd'))
    if not dates:
        return {'status': 'unavailable', 'price_asof_usd': None,
                'earnings_reaction_status': 'unavailable_without_verified_event_labels'}
    if history.get('asof') != case['asof'] or len(returns) != len(dates) or len(prices) != len(dates):
        raise ValueError('Financial-history cutoff or array lengths disagree with the capability')
    origin = datetime.fromisoformat(case['asof']).astimezone(ZoneInfo('America/New_York')).date()
    days = [date.fromisoformat(day) for day in dates]
    if days != sorted(set(days)) or any(day > origin for day in days):
        raise ValueError('Financial history contains duplicate, unsorted or future sessions')
    if any(not _finite(ret) for ret in returns):
        raise ValueError('Returns must be finite decimal total returns')
    if any(price is not None and (not _finite(price) or price <= 0) for price in prices):
        raise ValueError('Prices must be positive finite dollars or unknown')
    largest = max(range(len(returns)), key=lambda i: abs(returns[i]))
    return {'status': 'available', 'asof': case['asof'], 'sessions': len(days),
            'start_date': dates[0], 'end_date': dates[-1],
            'price_asof_usd': prices[-1] if days[-1] == origin else None,
            'daily_rms_percent': 100*math.sqrt(sum(ret*ret for ret in returns)/len(returns)),
            'largest_absolute_return_percent': 100*abs(returns[largest]),
            'largest_return_date': dates[largest],
            'earnings_reaction_status': 'unavailable_without_verified_event_labels',
            'note': 'These are observed price/return statistics, not a study of earnings reactions.'}


def offer_price(quote):
    candidates = re.findall(r'(?:US\$|(?<![A-Za-z])\$)\s*(\d[\d,]*(?:\.\d+)?)\s*'
                            r'(?:in cash\s+)?per\s+(?:common\s+)?share\b', normalized(quote), re.I)
    values = {Decimal(value.replace(',', '')) for value in candidates}
    return float(next(iter(values))) if len(values) == 1 and next(iter(values)) > 0 else None


class EvidenceToolkit(GroundedToolkit):
    def __init__(self, capability, archive, anchor=None, trace_path=None):
        supplied = capability.get('forecast_anchor')
        if not _finite(supplied):
            raise ValueError('Supply the run-specific forecast_anchor; global anchor files are not used')
        if ((anchor is not None and anchor != supplied) or
                capability['case'].get('anchor', supplied) != supplied):
            raise ValueError('Prompt/case/tool OLS anchors disagree')
        super().__init__(capability, archive, supplied, trace_path)
        self.evidence_receipts = {}

    def _receipt(self, value):
        value = {'case_id': self.data['case']['case_id'], 'asof': self.data['case']['asof'], **value}
        rid = digest(value)
        self.evidence_receipts[rid] = deepcopy(value)
        return {'receipt_id': rid, **value}

    def financial_history(self) -> dict:
        """Compute observed pre-cutoff price/return statistics. No earnings attribution or future outcomes."""
        return self._receipt({'kind': 'financial_history', **financial_snapshot(self.data['case'])})

    def _transaction(self, fields):
        if not set(fields) <= {'announcement', 'financing', 'consideration', 'expected_close'}:
            raise ContractError('invalid_transaction_field', 'transaction_facts', 'Use the documented transaction fields only.')
        result = super()._transaction(fields)
        consideration = result['fields'].get('consideration', {})
        quote = consideration.get('quote', '')
        price = offer_price(quote)
        clauses = re.split(r'(?<=[.!?])\s+', normalized(quote))
        explicit_cash = any(re.search(r'\b(?:all[- ]cash|per share in cash|'
                                      r'in cash per (?:common )?share|cash consideration|cash payment)\b', clause, re.I) and
                            not re.search(r'\b(?:whether|may|might|could|reportedly|rumou?r|considering)\b|'
                                          r'\b(?:not|no)\s+(?:an?\s+)?(?:all[- ]cash|cash|payment|consideration)\b', clause, re.I)
                            for clause in clauses)
        result['cash_consideration'] = 'explicit_source_statement' if explicit_cash else 'unknown'
        mixed = bool(re.search(r'\b(?:cash and stock|cash and shares|stock and cash|'
                               r'cash plus|cash as well as|combination|exchange ratio|shares of)\b', quote, re.I))
        fixed = (result['cash_consideration'] == 'explicit_source_statement' and price is not None and
                 not mixed and self._mentions_target(consideration))
        observed = financial_snapshot(self.data['case'])['price_asof_usd']
        result.update(offer_per_share_usd=price, price_asof_usd=observed,
                      fixed_cash_price_basis='supported_by_quote' if fixed else 'unknown',
                      observed_discount_to_quoted_offer_percent=
                          100*(1-observed/price) if observed is not None and price is not None else None,
                      deal_break_probability='unavailable',
                      future_volatility_compression='model_hypothesis_not_established')
        return result

    def deal_terms(self, transaction_facts: dict) -> dict:
        """Check bound shareholder consideration separately from funding; compute observed offer discount."""
        return self._receipt({'kind': 'deal_terms', **self._transaction(transaction_facts),
                              'next_step_if_unknown': 'Inspect and bind explicit shareholder payment terms; do not infer them from funding.'})

    def check_network_channel(self, event_key: str, relation_id: str, event_fact_ids: list[str],
                              exposure_fact_ids: list[str], affected_item: str,
                              channel: Literal['supplier_input', 'customer_output', 'competition']) -> dict:
        """Check a product/input channel across bound event and target-exposure quotes. Unknown is a lead."""
        if not self.data['network']:
            raise ContractError('wrong_arm', 'check_network_channel', 'Only NET has graph-channel tools.')
        if relation_id not in self.relations or channel not in CHANNEL_ROLES:
            raise ContractError('invalid_channel', 'relation_id/channel',
                                'Obtain a relation ID; choose supplier_input, customer_output, or competition.')
        if not event_fact_ids or not event_key or not isinstance(affected_item, str):
            raise ContractError('invalid_channel', 'event_fact_ids/affected_item', 'Provide bound event facts and a specific item.')
        events = [self._fact(fid) for fid in event_fact_ids]
        exposure = [self._fact(fid) for fid in exposure_fact_ids]
        path = self.relations[relation_id]
        item = normalized(affected_item).casefold()
        reasons = []
        if len(item) < 3 or item in {'company', 'companies', 'operations', 'production', 'demand', 'business'}:
            reasons.append('affected_item_not_specific')
        endpoint_names = {normalized(str(path.get(k, ''))).casefold() for k in ('name', 'entity')}
        endpoint_names.update({self.data['case']['ticker'].casefold(),
                               self.data['case']['company'].casefold()})
        entity_words = {word for name in endpoint_names for word in re.findall(r'[a-z0-9]+', name)}
        item_words = set(re.findall(r'[a-z0-9]+', item))
        if not item_words or item_words <= entity_words:
            reasons.append('entity_name_is_not_a_product')
        if path.get('hops') != 1:
            reasons.append('intermediate_product_continuity_not_verified')
        if not any(step.get('neighbor_role') == CHANNEL_ROLES[channel] for step in path.get('steps', [])):
            reasons.append('channel_does_not_match_relation_direction')
        event_matches = [f for f in events if f['scope'] == 'network' and
                         f['category'] != 'broker_commentary' and
                         re.search(r'(?<!\w)'+re.escape(item)+r'(?!\w)', normalized(f['quote']).casefold())]
        if not event_matches:
            reasons.append('affected_item_missing_from_event_quote')
        endpoint_ids = set(path.get('permnos', []))
        if not any(endpoint_ids & set(self._source(f['source_id']).get('ids', [])) for f in event_matches):
            reasons.append('event_source_not_on_path')
        target = self.data['case']['company']
        ticker = self.data['case']['ticker']
        exposure_matches = [f for f in exposure if f['category'] != 'broker_commentary' and
                            re.search(r'(?<!\w)'+re.escape(item)+r'(?!\w)', normalized(f['quote']).casefold()) and
                            (target.casefold() in f['quote'].casefold() or
                             re.search(r'\b'+re.escape(ticker)+r'\b', f['quote'], re.I))]
        if not exposure_matches:
            reasons.append('target_product_exposure_missing')
        if not any(CHANNEL_SHOCK[channel].search(f['quote']) for f in event_matches):
            reasons.append('affected_quantity_or_input_shock_missing')
        mitigated = any(MITIGATION.search(f['quote']) for f in event_matches)
        if mitigated:
            reasons.append('source_reports_mitigation_without_supported_remaining_effect')
        cutoff = datetime.fromisoformat(self.data['case']['asof'])
        if not any(timedelta(0) < cutoff-datetime.fromisoformat(f['available_at']) <= timedelta(days=14)
                   for f in event_matches):
            reasons.append('no_fresh_product_event')
        return self._receipt({'kind': 'network_channel', 'event_key': event_key,
                              'relation_id': relation_id, 'event_fact_ids': sorted(set(event_fact_ids)),
                              'exposure_fact_ids': sorted(set(exposure_fact_ids)), 'affected_item': affected_item,
                              'channel': channel, 'status': 'insufficient_evidence' if reasons else 'conditional_channel',
                              'missing_or_conflicting_evidence': reasons,
                              'materiality': 'unknown', 'transmission_effect': 'model_inference_not_verified',
                              'next_step': 'Search and inspect target-product evidence or retain the lead without changing OWN.' if reasons else
                                           'Reason conditionally; this receipt does not establish magnitude, pricing, or realized risk.'})

    def _drivers(self, items, transaction, network):
        output = super()._drivers(items, transaction, network)
        for driver in output:
            facts = [self._fact(fid) for fid in driver['fact_ids']]
            for sentence in re.split(r'(?<=[.!?])\s+', driver['interpretation']):
                if EMPIRICAL_REACTION.search(sentence) and not any(
                        normalized(sentence).rstrip('.!?').casefold() in normalized(f['quote']).casefold()
                        for f in facts if f['category'] != 'broker_commentary'):
                    raise ContractError('unsupported_historical_reaction', 'interpretation',
                                        'Ordinary price history does not verify typical earnings reactions. Cite the exact empirical source assertion or remove the historical claim.')
            if PRICE_PINNING.search(driver['interpretation']) and transaction.get('fixed_cash_price_basis') != 'supported_by_quote':
                raise ContractError('unsupported_fixed_price_mechanism', 'interpretation',
                                    'Fixed cash shareholder consideration is unverified. Call deal_terms; observed offer proximity alone does not establish price pinning or future volatility compression.')
            if PRICE_PINNING.search(driver['interpretation']) and not re.search(
                    r'\b(?:could|may|might|if|conditional)\b', driver['interpretation'], re.I):
                raise ContractError('price_mechanism_must_be_conditional', 'interpretation',
                                    'Verified fixed cash terms do not guarantee completion or future price stability. State the compression mechanism conditionally.')
            if any(LOW_DEAL_RISK.search(driver[k]) for k in ('interpretation', 'limitations')):
                raise ContractError('unestimated_deal_break_probability', 'interpretation/limitations',
                                    'The toolkit has no calibrated deal-break probability. State conditions and uncertainty without asserting low break risk.')
            if network:
                matches = [(rid, r) for rid, r in self.evidence_receipts.items() if
                           r['kind'] == 'network_channel' and r['event_key'] == driver['event_key'] and
                           r['status'] == 'conditional_channel' and r['relation_id'] in driver['relation_ids'] and
                           set(r['event_fact_ids']) <= set(driver['fact_ids']) and
                           set(r['exposure_fact_ids']) <= set(driver['exposure_fact_ids'])]
                if not matches:
                    raise ContractError('network_product_channel_unchecked', 'new_events',
                                        'Call check_network_channel for these exact facts and path. An eligible target-product channel is required; a broad graph edge is insufficient.')
                driver['channel_receipt_id'] = matches[-1][0]
        return output

    def _finish(self, score, drivers, extra):
        result = super()._finish(score, drivers, extra)
        forecast = self.submissions[-1]['forecast']
        forecast.update(version=VERSION, evidence_receipts=deepcopy(self.evidence_receipts))
        receipt = digest(forecast)
        self.submissions[-1]['receipt'] = receipt
        return {**result, 'receipt': receipt}

    def call(self, tool, **args):
        if tool not in {'financial_history', 'deal_terms', 'check_network_channel'}:
            return super().call(tool, **args)
        try:
            result, ok = getattr(self, tool)(**args), True
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            result, ok = {'error': self._error(exc)}, False
        self.trace.append({'tool': tool, 'args': args, 'ok': ok, 'result': result})
        self.save_trace()
        return result

    def save_trace(self):
        if self.trace_path:
            write(self.trace_path, {'version': VERSION, 'capability_hash': digest(self.data),
                  'calls': self.trace, 'reads': self.reads, 'inspected_entities': sorted(self.inspected_entities),
                  'facts': self.facts, 'dates': self.dates, 'relations': self.relations,
                  'evidence_receipts': self.evidence_receipts,
                  'submissions': self.submissions, 'labels_accessible': False})


def serve(capfile, tracefile):
    from mcp.server.mcpserver import MCPServer
    cap = read(capfile)
    archive, _ = load(cap['case']['case_id'], cap['network'])
    kit = EvidenceToolkit(cap, archive, trace_path=tracefile)
    server = MCPServer('news')
    names = COMMON_TOOLS + ['financial_history', 'deal_terms'] + (
        ['exposure_paths', 'network_leads', 'check_network_channel', 'submit_network'] if cap['network'] else ['submit_own'])
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('capability')
    parser.add_argument('trace')
    args = parser.parse_args()
    serve(args.capability, args.trace)
