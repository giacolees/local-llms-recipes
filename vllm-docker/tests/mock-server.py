#!/usr/bin/env python3
"""Compatibility entry point for the relocated test harness."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/vllm-docker/tests/mock-server.py'), run_name='__main__')
