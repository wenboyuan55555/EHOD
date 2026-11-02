"""Load pinned research snapshots without leaking generic module names."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from threading import RLock

_LOCK = RLock()


def load_snapshot(root: Path, entry: str, name: str, aliases=()):
    """Preserve upstream files while isolating names such as ``modules``."""
    with _LOCK:
        if name in sys.modules:
            return sys.modules[name]
        saved = {k: v for k, v in sys.modules.items()
                 if any(k == a or k.startswith(a + ".") for a in aliases)}
        original_path = sys.path[:]
        for key in saved:
            del sys.modules[key]
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location(name, root / entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        finally:
            for key in list(sys.modules):
                if any(key == a or key.startswith(a + ".") for a in aliases):
                    del sys.modules[key]
            sys.modules.update(saved)
            sys.path[:] = original_path
        return module
