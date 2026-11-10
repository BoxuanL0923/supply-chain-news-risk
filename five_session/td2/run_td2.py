"""Only supported command entry point; memory/offline limits precede scientific imports."""
import argparse
import sys
from runtime_v3 import limit,OUT,new_json,peak
limit()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['test','prepare','build','fit','freeze','cost','evaluate'])
    p.add_argument('--year',type=int)
    p.add_argument('--input-usd-per-million',type=float)
    p.add_argument('--output-usd-per-million',type=float)
    args=p.parse_args()
    if args.command=='test':
        import fixture_tests
        from boundary_tests import run_boundary
        result=fixture_tests.run(); bounds=run_boundary()
        result['boundary_tests']=bounds; result.update(peak())
        import uuid
        path=OUT/f'SYNTHETIC_TESTS_{uuid.uuid4().hex}.json'; new_json(path,result)
        print(str(path)); print('TEST_COUNTS',result)
        if result['failures'] or result['errors'] or bounds['failures'] or bounds['errors']: sys.exit(1)
    elif args.command=='evaluate':
        from evaluation import run
        run()
    else:
        import pipeline
        if args.command=='build':
            if args.year is None: p.error('--year required')
            pipeline.build_year(args.year)
        elif args.command=='cost':
            pipeline.cost(args.input_usd_per_million,args.output_usd_per_million)
        else: getattr(pipeline,args.command)()

if __name__=='__main__': main()
