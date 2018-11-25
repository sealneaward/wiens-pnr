"""make_raw_from_untrained.py

Usage:
    make_raw_from_untrained.py <fold_index> <f_data_config> <f_model_config> <f_detect_config> <every_K_frame>

Arguments:
    <fold_index> fold index
    <f_data_config> from contains config of game_ids that have not been trained example: rev3_1-bmf-25x25.yaml
    <f_model_config> example: conv2d-3layers-25x25.yaml
    <f_detect_config> example: nms1.yaml
    <every_K_frame> something

Example:
     python make_raw_from_untrained.py 0 rev3_1-bmf-25x25.yaml conv2d-3layers-25x25.yaml nms.yaml 5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
optimize_loss = tf.contrib.layers.optimize_loss
import numpy as np
import os
import sys
from tqdm import tqdm
from docopt import docopt
import yaml
import gc
import cPickle as pkl
import matplotlib.pyplot as plt
import pandas as pd

from wiens.model.convnet2d import ConvNet2d
from wiens.data import constant
from wiens.detect.nms import NMS
from wiens.data.dataset import BaseDataset
from wiens.data.extractor import BaseExtractor
from wiens.data.loader import GameSequenceLoader, WiensSequenceLoader, BaseLoader
from wiens.annotation import annotation

import wiens.config as CONFIG
game_dir = constant.game_dir
pnr_dir = os.path.join(game_dir, 'pnr-annotations')


def make_raw_non_conv(data_config, model_config, exp_name, fold_index, plot_folder, every_K_frame=25):
    """
    Load trained model and create probabilities from raw data that is seperated
    from training and testing data.
    Parameters
    ----------
    data_config: dict
        config that contains games trained on
    model_config: dict
        config
    exp_name: str
        config
    fold_index: int
        config
    every_K_frame: int
        splitting events
    plot_folder: str
        location of pkl files
    Returns
    -------
    probs: ndarray
        probabilities
    meta: array
        information about probabilities
    """

    # test section, change to games that have not been trained on
    data_config['data_config']['game_ids'] = data_config['data_config']['detect_validation']
    data_config['data_config']['detect_validation'] = []
    dataset = BaseDataset(data_config, int(arguments['<fold_index>']), load_raw=True, no_anno=True)
    extractor = BaseExtractor(data_config)
    loader = WiensSequenceLoader(dataset=dataset, extractor=extractor, batch_size=data_config['batch_size'], raw=True)
    annotations = annotation.prepare_gt_file_from_raw_label_dir(pnr_dir, game_dir, game_id=data_config['data_config']['game_ids'][0])
    annotations = pd.DataFrame(annotations)
    annotations = annotations.loc[annotations.gameid.isin(data_config['data_config']['game_ids']), :]

    # load model_dict from saved states
    model_dict = pd.read_pickle('%s/wiens-models.pkl' % (CONFIG.model.dir))
    models = model_dict.keys()

    for model_name in models:
        model = model_dict[model_name]
        ind = 0
        for game_id in dataset.game_ids:
            game_events = dataset.games[game_id]['events']
            for event_id, event in enumerate(game_events):
                event = event['playbyplay']
                if event.empty:
                    continue

                print('Game: %s, Index: %i' % (game_id, event_id))
                loaded = loader.load_event(event_id=event_id, game_id=game_id, every_K_frame=every_K_frame)
                labels = annotations.loc[annotations.eid == event_id, 'gameclock'].values
                if loaded is not None:
                    if loaded == 0:
                        ind += 1
                        continue
                    batch_xs, gameclocks = loaded
                    meta = [event_id, game_id]
                    batch_xs, label = batch_xs[0][0], batch_xs[0][1]
                    batch_xs = batch_xs.reshape(1, -1)

                else:
                    print('Bye')
                    sys.exit(0)

                probs = model.predict_proba(batch_xs)
                probs = np.full([len(gameclocks), 2], probs)

                # save the raw predictions
                probs = probs[:, 1]

                if not os.path.exists('%s/pkl/' % plot_folder):
                    os.makedirs('%s/pkl/' % plot_folder)

                pkl.dump([gameclocks, probs, labels], open('%s/pkl/%s-raw-%i.pkl' % (plot_folder, model_name, ind), 'w'))
                pkl.dump(meta, open('%s/pkl/%s-raw-meta-%i.pkl' % (plot_folder, model_name, ind), 'w'))
                ind += 1



def label_in_cand(cand, labels):
    for l in labels:
        if l > cand[1] and l < cand[0]:
            return True
    return False


def detect_from_prob(data_config, model_config, detect_config, exp_name, fold_index, plot_folder):
    """
    From probability maps, use detection to identify pnr instances from
    unseen data.
    Parameters
    ----------
    data_config: dict
        config
    model_config: dict
        config
    detect_config: dict
        config
    exp_name: str
        config
    fold_index: int
        config
    plot_folder: str
        location of pkl files
    Returns
    -------
    """
    model_dict = pd.read_pickle('%s/wiens-models.pkl' % (CONFIG.model.dir))
    models = model_dict.keys()

    data_config['data_config']['game_ids'] = data_config['data_config']['detect_validation']
    data_config['data_config']['detect_validation'] = []

    for model_name in models:
        dataset = BaseDataset(data_config, int(arguments['<fold_index>']), load_raw=True, no_anno=True)
        detector = eval(detect_config['class'])(detect_config)
        all_pred_f = filter(lambda s:'%s-raw-' % model_name in s and '%s-raw-meta' % model_name not in s,os.listdir('%s/pkl'%(plot_folder)))

        annotations = []
        for _, f in tqdm(enumerate(all_pred_f)):
            ind = int(f.split('.')[0].split('-')[-1])
            model = f.split('.')[0].split('-')[0]
            gameclocks, pnr_probs, labels = pkl.load(open('%s/pkl/%s-raw-%i.pkl'%(plot_folder, model, ind), 'rb'))
            meta = pkl.load(open('%s/pkl/%s-raw-meta-%i.pkl' %(plot_folder, model, ind), 'rb'))
            cands, mp, frame_indices = detector.detect(pnr_probs, gameclocks, True)

            plt.plot(gameclocks, pnr_probs, '-')
            if mp is not None:
                plt.plot(gameclocks, mp, '-')
            # plt.plot(np.array(labels), np.ones((len(labels))), '.')
            for ind, cand in enumerate(cands):
                cand_x = np.arange(cand[1], cand[0], .1)
                plt.plot(cand_x, np.ones((len(cand_x))) * .95, '-' )
                anno = {
                    'gameid': meta[1],
                    'gameclock': gameclocks[frame_indices[ind]],
                    'eid':meta[0],
                    'quarter':dataset.games[meta[1]]['events'][meta[0]]['quarter']
                }
                annotations.append(anno)

            plt.ylim([0,1])
            plt.title('Game: %s, Event: %i'%(meta[1], meta[0]))
            plt.savefig('%s/%s-%s-raw-%i.png' %(plot_folder, model_name, detect_config['class'], ind))
            plt.clf()

        pkl.dump(annotations, open('%s/gt/%s-from-raw-examples.pkl'%(pnr_dir, model_name), 'wb'))
        annotations = pd.DataFrame(annotations)
        game_ids = annotations.loc[:,'gameid'].drop_duplicates(inplace=False).values
        for game_id in game_ids:
            print('Detecting game: %s' % (game_id))
            annotations_game = annotations.loc[annotations.gameid == game_id,:]
            annotations_game = annotations_game.sort_values(by=['eid','gameclock'], ascending=[True, False])
            annotations_game = annotations_game.drop('gameid', axis=1, inplace=False)
            annotations_game.to_csv('%s/format/%s-detect-%s-%s.csv' % (pnr_dir, model_name, game_id, detect_config['class']), index=False)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])
    f_model_config = '%s/%s' % (CONFIG.model.config.dir,arguments['<f_model_config>'])
    f_detect_config = '%s/%s' % (CONFIG.detect.config.dir,arguments['<f_detect_config>'])

    data_config = yaml.load(open(f_data_config, 'rb'))
    model_config = yaml.load(open(f_model_config, 'rb'))
    detect_config = yaml.load(open(f_detect_config, 'rb'))

    model_name = os.path.basename(f_model_config).split('.')[0]
    data_name = os.path.basename(f_data_config).split('.')[0]
    exp_name = '%s-X-%s' % (model_name, data_name)

    fold_index = int(arguments['<fold_index>'])
    every_K_frame = int(arguments['<every_K_frame>'])
    plot_folder = '%s/%s' % (CONFIG.plots.dir, exp_name)

    make_raw_non_conv(data_config, model_config, exp_name, fold_index, every_K_frame=every_K_frame, plot_folder=plot_folder)
    detect_from_prob(data_config, model_config, detect_config, exp_name, fold_index, plot_folder)
