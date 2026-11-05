"""Verify existing fresh-fit output files; this command does not train models."""
import argparse
import json
from pathlib import Path
from repro.reporting import report, SPECS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    out=args.output.resolve()
    env=json.loads((out/'environment.json').read_text(encoding='utf-8'))
    suites=[name for name in SPECS if (out/name/'raw.csv').is_file()]
    if not suites: parser.error('No computed raw.csv files found')
    passed=report(out,suites,env['suite'],env['source_config_sha256'])
    print(json.dumps({'passed':passed,'suites':suites}))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
