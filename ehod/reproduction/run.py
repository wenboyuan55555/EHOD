"""Reproduce a paper table or a named baseline/module combination."""
import argparse
import json
import os
from pathlib import Path
from ehod.reproduction.registry import select_rows, tables
from ehod.reproduction.pipeline import Runner, ROOT


def execute(runner, row, history, seed):
    op, method = row['operation'], row['method']
    selector = None
    if op == 'normal':
        result = runner.normal(row.get('backbone', method), history, seed)
    elif op in {'hcf', 'hcf_event'}:
        result = runner.hcf(history, seed)
        if op == 'hcf_event':
            selector = 'evaluation.event_direct_test'
    elif op == 'event_origin':
        result = runner.event_origin(method, seed)
        selector = 'event' if method == 'event_matrix' else 'evaluation.event_direct'
    elif op == 'event_hcf':
        result = runner.event_hcf(method, seed)
        selector = 'event'
    elif op in {'portability', 'portability_base'}:
        result = runner.portability(method, seed)
        selector = 'base' if op == 'portability_base' else 'event'
    elif op in {'ehod', 'eadt_ablation'}:
        key, threshold = runner.hcf_cache(seed)
        name = 'validated_v5_complete_grid' if op == 'ehod' else 'ablation_' + row['variant']
        result = runner.eadt('count_fusion', key, threshold, seed, name, row.get('arguments', ()))
        selector = 'event'
    elif op == 'hcf_ablation':
        mode = row['prediction_mode']
        if mode == 'direct':
            runner.run('baselines.regular._shared.train_discrete',
                       ['--model', 'stidgcn', '--variant', 'plain', '--history-hours', history,
                        '--seed', seed, '--device', runner.args.device],
                       runner.output / ('runs/stidgcn/h%d_plain_seed%d_formal/result.json' % (history, seed)))
        else:
            runner.normal('stidgcn', history, seed)
        result = runner.output / ('runs/continuous_fusion_od/h%d_seed%d_%s/result.json' % (history, seed, row['run_name']))
        runner.run('ehod.hcf.train', ['--history-hours', history, '--seed', seed,
                   '--device', runner.args.device, '--continuous-mode', 'hierarchical',
                   '--feature-ablation', row['feature_ablation'], '--magnitude-loss', 'decision',
                   '--prediction-mode', mode, '--run-name', row['run_name']], result)
    else:
        raise ValueError('Unimplemented operation: ' + op)
    return result, selector


def main(argv=None):
    available = tables()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', choices=tuple(available))
    p.add_argument('--city', choices=('nyc', 'chicago'))
    p.add_argument('--methods', default='all', help='Comma-separated method IDs')
    p.add_argument('--variants', default='all', help='Comma-separated variant IDs')
    p.add_argument('--histories', help='Defaults to histories in the selected table')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--list', action='store_true', help='List all table rows without loading models')
    args = p.parse_args(argv)
    if args.list:
        for name, table in available.items():
            if args.table and args.table != name:
                continue
            print('%s [%s; %s]' % (name, table['paper_label'], table['scope']))
            for row in table['rows']:
                print('  %-15s %-27s %s' % (row['method'], row['variant'], row['label']))
        return
    if not args.table or not args.city:
        p.error('--table and --city are required unless --list is given')
    table = available[args.table]
    try:
        rows = select_rows(table, args.methods, args.variants)
        seeds = [int(s) for s in args.seeds.split(',')]
        histories = [int(h) for h in args.histories.split(',')] if args.histories else table['histories']
        if not seeds or set(seeds) - set(table['seeds']):
            raise ValueError('Seeds must be drawn from 0,1,2')
        if not histories or set(histories) - set(table['histories']):
            raise ValueError('This table uses histories ' + str(table['histories']))
    except ValueError as exc:
        p.error(str(exc))
    runner = Runner(args)
    output = Path(os.environ.get('EHOD_OUTPUT_ROOT', str(ROOT / 'outputs'))).resolve()
    for history in histories:
        for row in rows:
            for seed in seeds:
                print('\n[%s / %s / %s / %s / H%d / seed%d]' %
                      (args.table, args.city, row['method'], row['variant'], history, seed), flush=True)
                result, selector = execute(runner, row, history, seed)
                if args.dry_run:
                    continue
                from ehod.reproduction.collect import select_metrics
                metrics = select_metrics(json.loads(result.read_text()), selector, table['scope'])
                if not all(key in metrics for key in ('mae', 'rmse', 'pcc')):
                    raise ValueError('Incomplete metric result: ' + str(result))
                record = {'table': args.table, 'paper_label': table['paper_label'], 'scope': table['scope'],
                          'city': args.city, 'history': history, 'method': row['method'], 'variant': row['variant'],
                          'label': row['label'], 'seed': seed, 'result': str(result.resolve().relative_to(output)),
                          'selector': selector}
                path = output / 'reproduction' / args.table / args.city / ('h%d' % history) / row['method'] / row['variant'] / ('seed%d.json' % seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
