This projec uses uv for everything and is written in python and mojo.
Run tests with `uv run pytest`.
Check lints with `uv run ruff check`
Check types with `uv run pyright`
Format changes with `uv run ruff format`

You can limit scope of pytests with something like `uv run pytest -v tests/test_tic_tac_toe.py::test_game_coordination`
Please make sure to fix up, type checking, ruff lints, and formatting in code changes.
Always makes sure tests are correct when making changes.
