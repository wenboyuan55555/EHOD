"""Shared CLI for the run.py entry point in each baseline directory."""
import argparse
from ehod.reproduction.run import main


def regular_main(method):
    p = argparse.ArgumentParser(description=method + ': standalone or frozen backbone + EADT')
    p.add_argument('--city', required=True, choices=('nyc', 'chicago'))
    p.add_argument('--variant', choices=('standalone', 'with_eadt'), default='standalone')
    p.add_argument('--histories', default='12')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    histories = args.histories.split(',')
    if any(h not in {'1', '3', '6', '12'} for h in histories):
        p.error('--histories must be drawn from 1,3,6,12')
    if args.variant == 'with_eadt' and histories != ['12']:
        p.error('The portability comparison uses history 12')
    for history in histories:
        table = 'backbone_portability' if args.variant == 'with_eadt' else ('normal_demand' if history == '12' else 'history_sensitivity')
        command = ['--table', table, '--city', args.city, '--methods', method,
                   '--variants', args.variant, '--histories', history,
                   '--seeds', args.seeds, '--device', args.device]
        main(command + (['--dry-run'] if args.dry_run else []))


def event_main(method):
    p = argparse.ArgumentParser(description=method + ': origin architecture or event mechanism on frozen HCF')
    p.add_argument('--city', required=True, choices=('nyc', 'chicago'))
    p.add_argument('--variant', choices=('origin', 'with_hcf', 'both'), default='both')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    command = ['--table', 'event_prediction', '--city', args.city, '--methods', method,
               '--variants', 'all' if args.variant == 'both' else args.variant,
               '--seeds', args.seeds, '--device', args.device]
    main(command + (['--dry-run'] if args.dry_run else []))
