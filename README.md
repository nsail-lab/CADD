# CADD
Copyright © 2025 by Northwestern University. All Rights Reserved.

### 1. Install the environment
```bash
pip install -r requirments.txt
```

For downloading the Whisper model, please run download_whisper.py.

### 2. Change data directory
Change the data path at paths.conf

### 3. Run baseline
```bash
bash run_baseline.sh
```

DATASET `jdd`, `jdd_synthetic` denotes the JDD and SYN datasets.

### 4.Run CADD
```bash
bash run_context.sh
```

Change the TYPE to `c`, `t`, `ct` in the bash file for using the context, transcripts
and context+transcript information.

