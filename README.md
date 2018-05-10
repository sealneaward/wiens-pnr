# wiens-pnr
Implementation of 2014 and 2016 Pick and Roll Sloan Papers

### Setup

1. Uncompress the SportVU data from `wiens/data/sportvu/data`
```bash
./uncompress.sh
```

2. Add the path to the data you extracted to the `wiens/data/constant.py` file.
```py
import os
if os.environ['HOME'] == '/home/neil':
    data_dir = '/home/neil/projects/pnr-labels/pnr'
    # insert data path as data_dir
else:
    raise Exception("Unspecified data_dir, unknown environment")
```

3. Install the packages required from the repo dir.
```
python setup.py build
python setup.py install
```

4. Unpack the preprocessed data in `wiens/data/rev3_1/0`
```bash
./uncompress.sh
```

5. Additional processing steps for preprocessed data using script in `wiens`.
```
python make_sequences_from_sportvu.py wiens.yaml
```

### Train

1. Train SVM classifier for binary prediction
```
python train_classifier.py wiens.yaml --binary
```

2. Train SVM classifier for multi labelled prediction
```
python train_classifier.py wiens.yaml --multi
```

### Process Data

This is if you want to sit through the 6-7 processing time of creating negative annotations, forming roles, etc.
Note, that you should delete the data in `wiens/data/rev3_1/0` for this to work.
Script ignores process if data already exists.

1. Run the script.
```
python make_sequences_from_sportvu.py wiens.yaml
```
