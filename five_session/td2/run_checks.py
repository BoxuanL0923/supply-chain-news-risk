"""Complete synthetic suite, with append-only report and enforced resource ceiling."""
import sys
import uuid
from runtime_v3 import limit,OUT,new_json,peak
limit()
import decoder_v3
sys.modules['decoder_v2']=decoder_v3
sys.modules['decoder']=decoder_v3
import delivery
import fixture_tests
from boundary_tests import run_boundary
from extended_tests import run_extended

if __name__=='__main__':
    a=fixture_tests.run(); b=run_boundary(); c=run_extended()
    result={'tests':a['tests']+b['tests']+c['tests'],'passed':a['passed']+b['passed']+c['passed'],
        'errors':a['errors']+b['errors']+c['errors'],'failures':a['failures']+b['failures']+c['failures'],
        'boundary_tests':b,'adapter_tests':c,'synthetic_only':True,**peak()}
    path=OUT/f'SYNTHETIC_TESTS_{uuid.uuid4().hex}.json'; new_json(path,result)
    print('TEST_COUNTS',result); print('TEST_REPORT',str(path))
    if result['errors'] or result['failures']: sys.exit(1)
