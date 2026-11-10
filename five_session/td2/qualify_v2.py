"""Full synthetic qualification: every v1 suite plus the v2 suite, bound to the code inventory.

Never issues a clearance. Writes QUALIFICATION_v2_<uuid>.json under the private root.
"""
import sys
import uuid
import runtime_v3
from runtime_v3 import CODE, OUT, new_json, peak, sha
if runtime_v3._job is None:
    runtime_v3.limit()
import decoder_v3
sys.modules['decoder_v2'] = decoder_v3
sys.modules['decoder'] = decoder_v3
import delivery  # v1 mapping (evaluation -> evaluation_v2) required by the v1 boundary suite
import fixture_tests
from boundary_tests import run_boundary
from extended_tests import run_extended
from fast_tests import run_fast
from provider_tests import run_provider
from tests_v2 import run_v2
from spec_v2 import VERSION

SUITE_NAMES = ['v1_fixture', 'v1_boundary', 'v1_extended', 'v1_fast_parity', 'v1_provider', 'v2_synthetic']

def main():
    results = [fixture_tests.run(), run_boundary(), run_extended(), run_fast(), run_provider(), run_v2()]
    report = {k: sum(r[k] for r in results) for k in ['tests', 'passed', 'errors', 'failures']}
    report.update(version=VERSION, suites=dict(zip(SUITE_NAMES, results)), synthetic_only=True,
                  code_sha256={str(p): sha(p) for p in sorted(CODE.glob('*.py'))}, **peak())
    path = OUT / f'QUALIFICATION_v2_{uuid.uuid4().hex}.json'
    new_json(path, report)
    print('TEST_COUNTS', {k: report[k] for k in ['tests', 'passed', 'errors', 'failures']}, flush=True)
    print('SUITES', {k: (v['passed'], v['tests']) for k, v in report['suites'].items()}, flush=True)
    print('QUALIFICATION_REPORT', str(path), flush=True)
    return 1 if report['errors'] or report['failures'] else 0

if __name__ == '__main__':
    sys.exit(main())
