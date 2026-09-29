"""
Tests for Streamlit App loading
"""

import py_compile
from pathlib import Path


def test_app_syntax_and_compilation():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists()
    # Ensure py_compile compiles app.py cleanly without syntax or indentation errors
    compiled = py_compile.compile(str(app_path), doraise=True)
    assert compiled is not None
