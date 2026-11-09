"""Active TD2 v2 entry point. Memory/offline/write limits precede scientific imports.

Earlier entry points (run_td2.py, run_td2_v2.py, run_td2_final.py, td2.py, batch_remaining.py) are
retained unchanged and inactive. Every command except qualify/evaluate/status requires a passing
QUALIFICATION_v2 receipt whose code hashes equal the current inventory.
"""
import argparse
import json
import sys
from runtime_v3 import limit, OUT, CODE, sha
limit()
import decoder_v3
sys.modules['decoder_v2'] = decoder_v3
sys.modules['decoder'] = decoder_v3

def require_qualified():
    current = {str(p): sha(p) for p in sorted(CODE.glob('*.py'))}
    for p in sorted(OUT.glob('QUALIFICATION_v2_*.json'), key=lambda q: q.stat().st_mtime, reverse=True):
        r = json.loads(p.read_text(encoding='utf-8'))
        if (r.get('tests', 0) > 0 and r.get('passed') == r.get('tests') and r.get('failures') == 0
                and r.get('errors') == 0 and r.get('code_sha256') == current):
            return p
    raise PermissionError('A passing QUALIFICATION_v2 receipt for the current code inventory is required')

def status():
    import io_v2 as io
    for y in range(2015, 2025):
        print(y, 'features', (io.FEATURES / f'year_{y}.parquet').exists(), 'evidence',
              (io.EVIDENCE / f'year_{y}.parquet').exists(), 'coverage', (io.COVERAGE / f'year_{y}.json').exists())
    for name in ['NEWS_EVAL_v2.json', 'MODEL_v2.json', 'SELECTION_v2.json', 'TOKEN_COST_ESTIMATE_v2.json',
                 'FREEZE_v2.json', 'CLEARANCE_REQUEST_v2.json', 'EVALUATION_STARTED_v2.json', 'EVALUATION_RESULT_v2.json']:
        print(name, (OUT / name).exists())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['qualify', 'news-eval', 'build', 'fit', 'cost', 'freeze', 'evaluate', 'status'])
    p.add_argument('--year', type=int)
    a = p.parse_args()
    if a.command == 'qualify':
        import qualify_v2
        sys.exit(qualify_v2.main())
    if a.command == 'evaluate':
        import evaluation_v3
        evaluation_v3.run()  # refuses unless clearance/TD2_CLEARANCE_v1.json matches FREEZE_v2.json
        return
    if a.command == 'status':
        status(); return
    q = require_qualified(); print('QUALIFIED_BY', str(q), flush=True)
    if a.command == 'news-eval':
        import io_v2
        print('NEWS_EVAL_V2', io_v2.build_eval_news(), flush=True)
    elif a.command == 'build':
        if a.year is None:
            p.error('--year required')
        import build_v2
        build_v2.build_year(a.year)
    elif a.command == 'fit':
        import head_v2
        head_v2.fit()
    elif a.command == 'cost':
        import cost_v2
        cost_v2.run()
    elif a.command == 'freeze':
        import freeze_v2
        freeze_v2.build(q)

if __name__ == '__main__':
    main()
