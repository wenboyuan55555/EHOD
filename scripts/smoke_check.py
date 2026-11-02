#!/usr/bin/env python3
"""Check real-data HCF/OD forwards and the EADT state interface, without fitting."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--city', choices=('nyc', 'chicago'), default='chicago')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-root', default=os.environ.get('EHOD_OUTPUT_ROOT', str(ROOT / 'outputs/smoke')))
    args = p.parse_args()
    os.environ['EHOD_CITY'] = args.city
    os.environ['EHOD_OUTPUT_ROOT'] = str(Path(args.output_root).expanduser().resolve())
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    import numpy as np
    import torch
    from ehod.data.od import N_NODES, load_static, load_normalized, history_batch, calendar_features
    from ehod.hcf.features import ContinuousFeatureStore
    from ehod.hcf.training_utils import torch_static
    from ehod.hcf.train import continuous_inputs
    from ehod.hcf.model import ContinuousHybridOD, feature_dimension
    from ehod.eadt.model import TaxiAwareEventTransitionAdapter
    from baselines.regular._shared.factory import create_model, forward_model, static_tensors
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.manual_seed(0)
    n = N_NODES
    static_raw = load_static()
    static = torch_static(static_raw, device)
    store = ContinuousFeatureStore()
    starts = np.array([7728], dtype=np.int32)
    origin = np.arange(16, dtype=np.int64)[None, :] % n
    destination = (origin + 1) % n
    model = ContinuousHybridOD(nodes=n, feature_dim=feature_dimension('hierarchical')).to(device).eval()
    for history in (1, 3, 6, 12):
        features, o, d, calendar, target = continuous_inputs(
            store, static, starts, origin, destination, history, device,
            'hierarchical', 'no_statistical_priors')
        with torch.no_grad():
            values = model(features, o, d, calendar, torch.log1p(target))
        assert features.shape == (1, 16, 41)
        assert all(v.shape == (1, 16) and torch.isfinite(v).all() for v in values)
        print('PASS HCF real-data forward, H=%d' % history, flush=True)
    adapter = TaxiAwareEventTransitionAdapter(
        nodes=n, event_types=8 if n == 77 else 7, venues=10 if n == 77 else 8).to(device).eval()
    zero = torch.zeros(8, 2, device=device)
    kind = zero.long()
    with torch.no_grad():
        values = adapter(kind, kind, kind, zero + .7, zero - 1, zero - 3,
                         zero, zero, torch.arange(8, device=device), torch.arange(8, device=device),
                         torch.zeros(8, device=device), torch.ones(8, device=device))
    assert all(v.shape == (8,) and torch.isfinite(v).all() for v in values)
    print('PASS EADT state interface', flush=True)
    history = torch.from_numpy(history_batch(load_normalized(), starts, 12)).to(device)
    calendar = torch.from_numpy(calendar_features(starts)).to(device)
    tensors = static_tensors(static_raw, device)
    for name in ('stidgcn', 'stid', 'stpro', 'odmixer', 'stan', 'odpf'):
        model = create_model(name, 12, device=device).to(device).eval()
        with torch.no_grad():
            result = forward_model(name, model, history, calendar, tensors,
                                   target_starts=torch.from_numpy(starts).long().to(device))
        assert result.shape == (1, n, n) and torch.isfinite(result).all(), name
        print('PASS %s OD forward' % name, flush=True)
    print('All smoke checks passed for %s. No training was performed.' % args.city)


if __name__ == '__main__':
    main()
