#!/usr/bin/env python3
"""Repository entry point; see --list for paper tables and combinations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehod.reproduction.run import main
if __name__ == '__main__':
    main()
