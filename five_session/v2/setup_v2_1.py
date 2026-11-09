"""Active source constructor; keep task-only seals out of later cleared evaluation."""
import setup_v2

create_original = setup_v2.create
def create(path, text):
    if path.name == 'run_td1.py':
        text=setup_v2.replace_one(text, 'import runtime_safety\nruntime_safety.seal_labels()\n', '')
        text=setup_v2.replace_one(text, '    a=p.parse_args()',
            "    a=p.parse_args()\n    if a.step!='evaluate':\n        import runtime_safety_checked\n        runtime_safety_checked.seal_labels()")
    create_original(path, text)

setup_v2.create=create
if __name__=='__main__': setup_v2.main()
