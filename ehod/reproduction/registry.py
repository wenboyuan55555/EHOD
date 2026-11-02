"""Ordered paper rows and configurations owned by the corresponding directories."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REGULAR_ORDER = ('ha','xgboost','lstm','duet','stid','stidgcn','patchstg','stdn','lightst','strep','stan','cmod','bootcf','stpro','odmixer')
EVENT_ORDER = ('t3','fuse','concat','ss_moe','egaf','qa_stgacn','rast','cdgon','event_matrix')
def read_json(path):
    return json.loads(Path(path).read_text())
def regular_configs():
    return {n: read_json(ROOT/'baselines/regular'/n/'config.json') for n in REGULAR_ORDER}
def event_configs():
    return {n: read_json(ROOT/'baselines/event'/n/'config.json') for n in EVENT_ORDER}
def tables():
    return {p.parent.name: read_json(p) for p in sorted((ROOT/'experiments').glob('*/config.json'))}
def select_rows(table, methods='all', variants='all'):
    rows = table['rows']
    for key, values in (('method', methods), ('variant', variants)):
        if values != 'all':
            requested = values.split(',')
            allowed = {r[key] for r in rows}
            unknown = set(requested) - allowed
            if unknown:
                raise ValueError('Unknown %s: %s; available: %s' % (key, ', '.join(sorted(unknown)), ', '.join(sorted(allowed))))
            rows = [r for r in rows if r[key] in requested]
    if not rows:
        raise ValueError('No table rows match the requested methods and variants')
    return rows
