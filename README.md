# CADD
Copyright © 2025 by Northwestern University. All Rights Reserved.

### 1. Install the environment
```bash
pip install -r requirments.txt
```

For downloading the Whisper model, please run `download_whisper.py`.

### 2. Change data directory
Change the data path at `paths.conf`.

### 3. Preprocessing

### 4. Run CADD
```bash
bash run_context.sh
```

Change the TYPE to `c`, `t`, `ct` in the bash file for using the context, transcripts and context+transcript information. For the experiments on the SYN dataset:
```bash
bash run_context_syn.sh
```
For the experiments on the In-The_wild dataset:
```bash
bash run_context_inthewild.sh
```
For our Boost setting:
```bash
bash run_bost.sh
```

### Configurations
You can modify the yaml file at `configs` folder in order to change the configurations of the models and training.
The trained model can be found at our huggingface page, [CADD](https://huggingface.co/collections/gcyzsl/cadd-models-67a0eb4e83c3565727a8f9d4).


### Baselines
```bash
bash run_baseline.sh
```

DATASET `jdd`, `jdd_synthetic` denotes the JDD and SYN datasets. For the experiments on the SYN dataset:
```bash
bash run_baseline_syn.sh
```
For the experiments on the In-The_wild dataset:
```bash
bash run_baseline_inthewild.sh
```

