_base_ = ["./detector-v4-pt-v3m2-ft-pid-fft.py"]
hooks_override = {"WandbNamer": {"extra": "scratch"}}
