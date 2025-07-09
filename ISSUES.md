This is just documenting some hiccups I hit along the way with Modular's stack.

## Tooling Issues

### Mising LSP Server
When installing with uv, the `mojo-lsp-server` does not get installed.
And we have no easy way to install it...so just stuck without lsp.

### Formatting not running
`uv run mojo format` does not work in github ci, but works locally.
hits:
```
/home/runner/work/alpha-max-zero/alpha-max-zero/.venv/bin/mojo: error: unable to resolve Mojo formatter in PATH
```

### Formatting not working
Some reason, we don't seem able to format `out self`
```
error: cannot format src/alpha_max_zero/kernels/games/tic_tac_toe.mojo: Cannot parse: 20:20:     fn __init__(out self):

Oh no! 💥 💔 💥
3 files left unchanged, 1 file failed to reformat.
```

### Can't run mojo tests from uv
```
dyld[77231]: Library not loaded: @rpath/libMojoJupyter.dylib
  Referenced from: <4C4C44E2-5555-3144-A196-AE9E10B2AAAC> /Users/bren077s/Projects/alpha-max-zero/.venv/lib/python3.12/site-packages/max/lib/mojo-test-executor
  Reason: tried: '/libMojoJupyter.dylib' (no such file), '/Users/bren077s/.local/lib/libMojoJupyter.dylib' (no such file), '/Users/bren077s/Projects/alpha-max-zero/.venv/lib/python3.12/site-packages/max/lib/../lib/libMojoJupyter.dylib' (no such file), '/Users/bren077s/Projects/alpha-max-zero/.venv/lib/python3.12/site-packages/max/lib/../lib/libMojoJupyter.dylib' (no such file), '/usr/local/lib/libMojoJupyter.dylib' (no such file), '/usr/lib/libMojoJupyter.dylib' (no such file, not in dyld cache)
[77224:221345426:20250708,231049.308490:ERROR bootstrap.cc:65] bootstrap_look_up com.apple.ReportCrash: Unknown service name (1102)	
```

For now just gonna test everything through python

## Graph API Issues

### Pyright complains about max graph types
example errors:
```
/Users/bren077s/Projects/alpha-max-zero/src/alpha_max_zero/game.py:9:6 - error: Stub file not found for "max.graph" (reportMissingTypeStubs)
/Users/bren077s/Projects/alpha-max-zero/src/alpha_max_zero/kernels.py:5:6 - error: Stub file not found for "max.driver" (reportMissingTypeStubs)
Type of "custom" is "(name: str, device: DeviceRef, values: Sequence[Value[Unknown]], out_types: Sequence[Type[Unknown]], parameters: Mapping[str, bool | int | str | DType] | None = None) -> list[Value[Unknown]]" (reportUnknownMemberType)
/Users/bren077s/Projects/alpha-max-zero/tests/test_tic_tac_toe.py
```

### Opaque types and values are not public

For this project opaque types are needed.
They are still labeled as private, so pyright complains about them.
I feel like at this point they should be official even if we eventually want to deprecate them.
