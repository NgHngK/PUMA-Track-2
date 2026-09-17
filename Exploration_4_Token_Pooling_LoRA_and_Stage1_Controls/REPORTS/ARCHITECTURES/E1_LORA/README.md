# Restricted LoRA experiment

Run `python run_train.py --output NEW_RUNS --weights PATH_TO_UNI2` with timm1.0.20. This reproduces the three fixed five-epoch seeds with local code, cached blocks0–19 and Q/V LoRA on blocks20–23. The checkpoint contains adapters and head; full frozen weights remain external by SHA256. Re-evaluate stored logits with `run_eval.py --predictions FILE --output FILE`. The run did not pass promotion. Full training details and individual epochs are preserved in RESULTS.
