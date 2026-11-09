"""Canonical TD2 CLI with bounded sparse traversal, offline and append-only rules."""
import sys
from runtime_v3 import limit
limit()
import decoder_v3
sys.modules['decoder_v2']=decoder_v3
sys.modules['decoder']=decoder_v3
import delivery
import fast_builder
fast_builder.install(delivery.pipeline,decoder_v3)
import run_td2
if __name__=='__main__': run_td2.main()
