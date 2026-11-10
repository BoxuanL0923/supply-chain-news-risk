"""TD1 entry point. Never imports a model/provider/network client."""
from runtime import limit
limit()
import argparse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('step',choices=['snapshots','manifest','fit','verify','evaluate','tests','repair','correction-tests','traps'])
    a=p.parse_args()
    if a.step!='evaluate':
        import runtime_safety_checked
        runtime_safety_checked.seal_labels()
    if a.step=='repair':
        from repair import run
        return run()
    if a.step=='correction-tests':
        from correction_tests import run
        return run()
    if a.step=='traps':
        import lookahead_traps
        return
    if a.step in ['snapshots','manifest']:
        import build
        return build.snapshots() if a.step=='snapshots' else build.manifests()
    if a.step=='fit':
        from fit import run
        return run()
    if a.step=='verify':
        from verify import run
        return run()
    if a.step=='evaluate':
        from evaluate import run
        try:
            return run()
        except PermissionError as error:
            print(str(error),flush=True)
            raise SystemExit(2)
    if a.step=='tests':
        import tests
        return tests.run_and_record()

if __name__=='__main__': main()
