# External storage contract

Large datasets, archives, extracted frames, caches, checkpoints, and benchmark
outputs must live outside the repository. GenFrames deliberately has no implicit
fallback directory: configure `GENFRAMES_STORAGE_ROOT` or pass an explicit root
to a command.

Example for the current Windows workstation:

```powershell
$env:GENFRAMES_STORAGE_ROOT = "<large-disk>:\GenFrames"
```

Code and committed manifests remain in the repository. Manifest paths are
relative to the configured dataset root so the same records can be mounted on a
different drive or operating system without rewriting provenance.
