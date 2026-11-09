"""Current TD2 entry point. Earlier entry points are retained, not supported."""
import sys
from runtime_v3 import limit
limit()
import decoder_v3
sys.modules['decoder_v2']=decoder_v3
sys.modules['decoder']=decoder_v3
import delivery
import run_td2
if __name__=='__main__': run_td2.main()
