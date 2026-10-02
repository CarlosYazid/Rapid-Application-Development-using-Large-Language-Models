"""Maintainer command: regenerate clean English exercises from the canonical graph.

This overwrites learner edits. Use switch_language.py to preserve student work.
"""
import argparse
import copy
import json
from pathlib import Path
from switch_language import _cell


def render(entry):
    cells = [_cell(c['cell_type'], c['content']['en'] if isinstance(c['content'], dict) else c['content'], c)
             for c in entry['cells']]
    return {'cells': cells, 'metadata': copy.deepcopy(entry['metadata']), 'nbformat': 4, 'nbformat_minor': 5}


def main(check=False):
    root = Path(__file__).resolve().parent.parent
    graph = json.loads((root / 'composer/translation_web.json').read_text())
    failures = []
    for name, entry in graph['notebooks'].items():
        target = root / name
        expected = render(entry)
        if check:
            if not target.is_file() or json.loads(target.read_text()) != expected:
                failures.append(name)
        else:
            target.write_text(json.dumps(expected, ensure_ascii=False, indent=1) + '\n')
    if failures:
        raise SystemExit('Generated notebook drift: ' + ', '.join(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
