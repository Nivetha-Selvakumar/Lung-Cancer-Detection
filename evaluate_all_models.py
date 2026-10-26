#!/usr/bin/env python3
"""
Root workspace wrapper for evaluate_all_models.py
Allows execution directly from root: python evaluate_all_models.py
"""
import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Delegate directly to backend/evaluate_all_models.py
backend_script = os.path.join(backend_dir, "evaluate_all_models.py")
with open(backend_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), backend_script, "exec")
    exec(code, {"__name__": "__main__", "__file__": backend_script})
