_base_ = ["./tiny_semseg.py"]

epoch = 2

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=0),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=5),
    dict(type="SimulateCrash"),
]
