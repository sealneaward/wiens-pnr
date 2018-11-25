"""make_conv_sequences_from_games.py

Usage:
    make_conv_sequences_from_games.py <f_data_config>

Arguments:
    <f_data_config>  example ''rev3_1-bmf-25x25.yaml''

Example:
    python make_conv_sequences_from_games.py rev3_1-bmf-25x25.yaml
"""
from wiens.data.dataset import BaseDataset
from wiens.data.extractor import BaseExtractor
from wiens.data.loader import BaseLoader
import wiens.config as CONFIG
from wiens.data.constant import data_dir
from wiens.data.utils import make_3teams_11players

from tqdm import tqdm
import os
from docopt import docopt
import yaml
import pandas as pd
import numpy as np
import signal
from contextlib import contextmanager
import cPickle as pkl
import time


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


def sequence_games():
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
            if os.path.exists(os.path.join(curr_folder, '%s_neg_t.npy' % game)):
                continue
            try:
                with time_limit(1500):
                    # create dataset fo single game
                    data_config['data_config']['game_ids'] = [game]
                    dataset = BaseDataset(data_config, fold_index=fold_index, game=game)

                    loader = BaseLoader(f_data_config, dataset, extractor, data_config['batch_size'])
                    loaded = loader.load_valid(extract=False, positive_only=False)
                    if loaded is None:
                        continue
                    else:
                        val_x, val_t, _ = loaded
                        if not len(val_x) > 0:
                            continue

                    pkl.dump(val_x, open(os.path.join(curr_folder, '%s_val_x.pkl' % game), 'wb'))
                    np.save(os.path.join(curr_folder, '%s_val_t' % game), val_t)
                    del val_x, val_t

                    x, t, _ = loader.load_train(extract=False, positive_only=True)
                    pkl.dump(x, open(os.path.join(curr_folder, '%s_pos_x.pkl' % game), 'wb'))
                    np.save(os.path.join(curr_folder, '%s_pos_t' % game), t)
                    del x

                    xs = []
                    ind = 0
                    while True:
                        print ('%i/%i' % (ind, len(dataset.train_hash)))
                        print (len(xs))
                        ind += 1
                        start = time.time()

                        loaded = loader.load_split_event('train', extract=False)
                        if loaded is not None:
                            if loaded == 0:
                                continue
                            batch_xs, labels, gameclocks, meta = loaded
                            ## filter out positive examples
                            new_batch_xs = []
                            for x in batch_xs:
                                e_gc = x.moments[len(x.moments) / 2].game_clock
                                ispositive = False
                                for label in labels:
                                    if np.abs(e_gc - label) < data_config['data_config']['t_negative']:
                                        ispositive = True
                                        break
                                if not ispositive:
                                    xs.append(x)

                            end = time.time()
                            print('\n Time executed: %s seconds' % (end - start))

                            if ind % 50 == 0:
                                print('Saving split')
                                # npy file was written from previous split
                                x = xs
                                xs = []
                                if os.path.exists(os.path.join(curr_folder, '%s_neg_x.pkl' % game)):
                                    x_arr = pd.read_pickle(os.path.join(curr_folder, '%s_neg_x.pkl' % game))
                                    x_arr.append(x)
                                    pkl.dump(x_arr, open(os.path.join(curr_folder, '%s_neg_x.pkl' % game), 'wb'))
                                else:
                                    pkl.dump(x, open(os.path.join(curr_folder, '%s_neg_x.pkl' % game), 'wb'))
                                del x
                        else:
                            print('Saving split')
                            # last split
                            x = xs
                            x_arr = pd.read_pickle(os.path.join(curr_folder, '%s_neg_x.pkl' % game))
                            x_arr.append(x)
                            pkl.dump(x_arr, open(os.path.join(curr_folder, '%s_neg_x.pkl' % game), 'wb'))
                            neg_t = np.array([[1, 0]]).repeat(len(x_arr), axis=0)
                            np.save(os.path.join(curr_folder, '%s_neg_t' % game), neg_t)
                            del x_arr, xs, x
                            break

            except TimeoutException as e:
                print("Game sequencing too slow for %s - skipping" % (game))  # some
                continue


if __name__ == '__main__':

    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
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

    sequence_games()