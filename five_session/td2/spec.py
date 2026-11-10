"""TD2 exploratory definitions; no licensed text or private observations."""
VERSION='TD2-five-session-v1'
NAMES=['size','cash','cash_flow','leverage','inventory']
K_GRID=[0,1,2,3,5,'all']
RIDGE_GRID=[0.01,0.1,1.,10.,100.,1000.]
SEEDS=[72101,72103,72109]
PARTNER_CAP=5
LOOKBACK_DAYS=7
MENU=[
 dict(name='size',firm_column='size',economic_meaning='Log dollar market equity',
      risk_channel='Scale interacted with partner activity and disruption news',
      asof_source='TD1 CRSP size; origin session 20:00 New York < cutoff'),
 dict(name='cash',firm_column='cash',economic_meaning='Cash divided by total assets',
      risk_channel='Liquidity buffer against partner payment or operating disruption',
      asof_source='TD1 cheq/atq; rdq end of day else fiscal end +120 days; age <=550 days'),
 dict(name='cash_flow',firm_column='cash_flow',economic_meaning='Operating cash flow divided by assets',
      risk_channel='Internal funding capacity interacted with partner operating and demand news',
      asof_source='TD1 JKP ocf_at; month end +6 calendar months; age <=550 days'),
 dict(name='leverage',firm_column='leverage',economic_meaning='Short and long term debt divided by assets',
      risk_channel='Financing sensitivity to partner credit and payment stress',
      asof_source='TD1 (dlcq+dlttq)/atq; rdq end of day else fiscal end +120 days; age <=550 days'),
 dict(name='inventory',firm_column='inventory',economic_meaning='Inventory divided by total assets',
      risk_channel='Inventory exposure to partner shortage, production and demand changes',
      asof_source='TD1 invtq/atq; rdq end of day else fiscal end +120 days; age <=550 days')]
LEXICONS={
 'size':['expansion','capacity','production','disruption','shutdown','demand'],
 'cash':['liquidity','payment','payments','shortage','disruption','shutdown'],
 'cash_flow':['sales','orders','demand','production','cash','operating'],
 'leverage':['debt','credit','default','bankruptcy','financing','payment'],
 'inventory':['inventory','inventories','shortage','supply','production','shipment']}
RULES={
 'target':'log(mean of next five squared decimal daily total returns)); origin excluded; label end <=2024-12-31',
 'clock':'All availability strictly before TD1 UTC cutoff; day-only news and edges available next New York midnight; origin day excluded',
 'graph':'FSB3 sealed October direct CUSTOMER/SUPPLIER rows only; latest start strictly before origin per relationship ID; active end > origin; undirected traversal with original reporting roles retained',
 'hop':'Exact shortest graph distance 1,2,3; no self or repeated node; deterministic lexicographic shortest witness',
 'quarantine':'All 91869 FSB3 classes=added IDs excluded; FSB5 candidate entity/PERMNO links excluded in identity sensitivity only',
 'news':'Reuters article-firm tags; latest-extract global one-to-one RIC/PERMNO restricted to <=2022 for development and <=2024 for evaluation; mapping first month must precede origin month',
 'article':'Prior 7 calendar days excluding origin day; earliest day per alt_id (same-day ties deterministic first physical row); headline first 200 + body_short first 1000 characters; lowercase exact ASCII words; score=distinct matching dictionary words/dictionary size; max across unique partner stories per item',
 'aggregation':'Up to five distinct news-bearing partners per exact hop; SHA256(cutoff,partner) priority; same chosen partners for all items; sum partner maximum scores /5, zero padding; main E=(E_H1+E_H2+E_H3)/3',
 'characteristic':'Uncentered raw TD1 values; missing C gives Z=0, recorded as unavailable; no indicator enters risk head; latest extracts with availability lags are NOT certified revision-free PIT',
 'ridge':'No intercept; standardized by training root mean square, no centering; minimize sum residual squares + lambda norm(scaled beta)^2; all eligible coordinates q=1; lambda by selection MSE; preprocessing fixed on 2015-2020',
 'baseline':'Use TD1 selection_mlp (fit 2015-2020) for ridge training and selection; TD1 refitted b only at final refit and evaluation',
 'selection':'K and none fallback vs all fallback selected on 2021-2022 only; malformed provider rows logged without backfill; Noul threshold grid .25,.5,.75 requires selection scores before freezing',
 'ablations':'Same frozen five coefficients and identical firm-week rows for main, exact H1,H2,H3; same cap 5 per exact-hop arm; no ablation refit',
 'evidence':'Graph witnesses with original relationship and node locators; no product evidence supplied by this source; physical_channel_verified=false',
 'claims':'Exploratory historical reproduction; no confirmed causal channel, total gain, or Jev advantage asserted'}
SLOT_SCHEMAS={
 'deepseek_order':{'schema':'TD2_deepseek_order_v1','model_version':'REQUIRED','question_version':'TD2-order-v1','records':[]},
 'jev_choice':{'schema':'TD2_jev_choice_v1','model_version':'REQUIRED','question_version':'TD2-choice-v1','records':[]},
 'jev_noul':{'schema':'TD2_jev_noul_v1','model_version':'REQUIRED','question_version':'TD2-noul-v1','records':[]}}
