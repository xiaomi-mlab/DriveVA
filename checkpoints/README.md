# Checkpoints

Put the released DriveVA and UNIVERSE full checkpoints here, for example:

```text
checkpoints/pdms90_9.safetensors
checkpoints/UNIVERSE.safetensors
```

Inference loads only the `--full_ckpt` file. EMA setup, optimizer state, warmup
configuration, and training-only noise or loss settings are not read by the
released inference entry points.
