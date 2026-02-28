"""Streamlit entrypoint.

Streamlit Cloud can be configured to run either:
- neo_pir_omr/ui/app_streamlit.py (recommended), or
- this root app.py (works as a wrapper).

This file keeps deployment simple when the platform expects app.py at the repo root.
"""

# Importing the Streamlit script executes it (it defines the UI at import time).
from neo_pir_omr.ui import app_streamlit  # noqa: F401
