from runtime_v3 import limit,OUT,new_json,peak
limit()
import sys
import uuid
import decoder_v3
sys.modules['decoder_v2']=decoder_v3
sys.modules['decoder']=decoder_v3
import delivery
from fast_tests import run_fast
if __name__=='__main__':
    result=run_fast(); result.update(peak()); path=OUT/f'FAST_PARITY_{uuid.uuid4().hex}.json'
    new_json(path,result); print('TEST_COUNTS',result); print('TEST_REPORT',str(path))
    if result['failures'] or result['errors']: sys.exit(1)
