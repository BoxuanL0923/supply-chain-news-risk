"""Complete synthetic qualification and code inventory binding; never issues clearance."""
import sys
import uuid
from runtime_v3 import limit,CODE,OUT,new_json,peak,sha
limit()
import decoder_v3
sys.modules['decoder_v2']=decoder_v3
sys.modules['decoder']=decoder_v3
import delivery
import fixture_tests
from boundary_tests import run_boundary
from extended_tests import run_extended
from fast_tests import run_fast
from provider_tests import run_provider

if __name__=='__main__':
    results=[fixture_tests.run(),run_boundary(),run_extended(),run_fast(),run_provider()]
    report={key:sum(r[key] for r in results) for key in ['tests','passed','errors','failures']}
    report.update(suites=results,synthetic_only=True,boundary_tests=results[1],
        code_sha256={str(p):sha(p) for p in sorted(CODE.glob('*.py'))},**peak())
    path=OUT/f'QUALIFICATION_{uuid.uuid4().hex}.json'; new_json(path,report)
    print('TEST_COUNTS',{k:report[k] for k in ['tests','passed','errors','failures']})
    print('QUALIFICATION_REPORT',str(path))
    if report['errors'] or report['failures']: sys.exit(1)
