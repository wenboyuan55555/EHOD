#!/usr/bin/env python3
"""Export a paper table in row order with mean and sample standard deviation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehod.reproduction.collect import main
if __name__ == '__main__':
    main()
