"""make_sequences_from_games.py

Usage:
    make_sequences_from_games.py <f_data_config>

Arguments:
    <f_data_config>  example ''wiens.yaml''

Example:
    python make_sequences_from_sportvu.py wiens.yaml
"""
from tqdm import tqdm
import os
from docopt import docopt
import yaml
import numpy as np
import signal
import pandas as pd
import cPickle as pkl
from contextlib import contextmanager

from wiens.data.dataset import BaseDataset
from wiens.data.extractor import BaseExtractor
from wiens.data.loader import BaseLoader
from wiens.data.constant import data_dir
import wiens.config as CONFIG

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")

f_data_config = '%s/%s'%(CONFIG.data.config.dir,arguments['<f_data_config>'])
data_config = yaml.load(open(f_data_config, 'rb'))

# make a new data directions
if ('<new_data_dir>' in arguments and arguments['<new_data_dir>'] != None):
    assert (arguments['<new_data_dir>'] == data_config['preproc_dir'])

new_root = os.path.join(data_dir, data_config['preproc_dir'])
if not os.path.exists(new_root):
    os.makedirs(new_root)

# save the configuartion
with open(os.path.join(new_root, 'config.yaml'), 'w') as outfile:
    yaml.dump(data_config, outfile)


# for fold_index in tqdm(xrange(data_config['data_config']['N_folds'])):
for fold_index in xrange(1): ## I have never actually used more than 1 fold...
    curr_folder = os.path.join(new_root, '%i' % fold_index)
    if not os.path.exists(curr_folder):
        os.makedirs(curr_folder)
    # Initialize dataset/loader
    dataset = BaseDataset(f_data_config, fold_index=fold_index, load_raw=False)
    extractor = BaseExtractor(f_data_config)
    games = dataset.game_ids
    data_config = yaml.load(open(f_data_config, 'rb'))

    for game in tqdm(games):
        if os.path.exists(os.path.join(curr_folder, '%s_t.npy' % game)):
            continue
        try:
            with time_limit(4500):
                # create dataset fo single game
                data_config['data_config']['game_ids'] = [game]
                dataset = BaseDataset(data_config, fold_index=fold_index, game=game)

                loader = BaseLoader(f_data_config, dataset, extractor, data_config['batch_size'])
                loaded = loader.load_valid(extract=False, positive_only=False)
                if loaded is None:
                    continue
                else:
                    val_x, val_t = loaded
                    if not len(val_x) > 0:
                        continue
                pkl.dump(val_x, open(os.path.join(curr_folder, '%s_val_x.pkl' % game), 'wb'))
                np.save(os.path.join(curr_folder, '%s_val_t' % game), val_t)
                del val_x, val_t

                x, t = loader.load_train(extract=False, positive_only=False)
                pkl.dump(x, open(os.path.join(curr_folder, '%s_x.pkl' % game), 'wb'))
                np.save(os.path.join(curr_folder, '%s_t' % game), t)
                del x

        except TimeoutException as e:
            print("Game sequencing too slow for %s - skipping" % (game))  # some
            continue
