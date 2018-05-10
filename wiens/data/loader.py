from __future__ import division
import cPickle as pickle
import yaml
import os
import numpy as np
import pandas as pd
import cPickle as pkl
import time
from tqdm import tqdm

from wiens.vis.Event import Event, EventException, FeatureException
from wiens.vis.Team import TeamNotFoundException
from wiens.data.extractor import ExtractorException, OneHotException
from wiens.data.utils import shuffle_2_array, make_3teams_11players
from wiens.data.constant import data_dir, game_dir
import wiens.config as CONFIG
from wiens.data.constant import game_dir


class BaseLoader:
    def __init__(self, data_config, dataset, extractor, batch_size, mode='sample', fraction_positive=.5):
        self.data_config = data_config
        self.dataset = dataset
        self.extractor = extractor
        self.batch_size = batch_size
        self.fraction_positive = fraction_positive
        self.mode = mode
        self.event_index = 0
        self.games = self.dataset.games

    def load_split(self, split='val', extract=True, positive_only=False):
        N_pos = 0
        ret_val = []
        ret_labels = []
        detail_labels = []
        self.extractor.augment = False
        istrain = split == 'train'
        while True:
            anno = self.dataset.propose_positive_Ta(jitter=False, train=istrain, loop=True)
            if anno == None:
                break
            try:
                start = time.time()
                e = Event(
                    self.dataset.games[anno['gameid']]['events'][int(anno['eid'])],
                    anno=anno,
                    gameid=anno['gameid'],
                    data_config=self.dataset.config['data_config']
                )
                e.sequence_around_t(anno, self.dataset.tfr, data_config=self.dataset.config)  # EventException
                e.build_features()
                end = time.time()
                print('\n Time executed: %s seconds' % (end - start))
                if extract:
                    # ExtractorException
                    ret_val.append(self.extractor.extract(e))
                else:
                    # just to make sure event not malformed (like
                    # missing player)
                    _ = self.extractor.extract_raw(e)
                    ret_val.append(e)
            except TeamNotFoundException as exc:
                pass
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            except FeatureException as exc:
                continue
            else:
                N_pos += 1
                ret_labels.append([0, 1])
                detail_labels.append(int(e.anno['label']))
        if not positive_only:
            for i in xrange(N_pos):
                while True:
                    try:
                        anno = self.dataset.propose_negative_Ta()
                        start = time.time()
                        e = Event(
                            self.dataset.games[anno['gameid']]['events'][int(anno['eid'])],
                            anno=anno,
                            gameid=anno['gameid'],
                            data_config=self.dataset.config
                        )
                        e.sequence_around_t(anno, self.dataset.tfr, data_config=self.dataset.config)  # EventException
                        e.build_features()
                        end = time.time()
                        print('\n Time executed: %s seconds' % (end - start))
                        if extract:
                            # ExtractorException
                            ret_val.append(self.extractor.extract(e))
                        else:
                            # just to make sure event not malformed (like
                            # missing player)
                            _ = self.extractor.extract_raw(e)
                            ret_val.append(e)
                    except EventException as exc:
                        pass
                    except ExtractorException as exc:
                        pass
                    except TeamNotFoundException as exc:
                        pass
                    else:
                        ret_labels.append([1, 0])
                        break
        self.extractor.augment = True
        return np.array(ret_val), np.array(ret_labels), np.array(detail_labels)

    def load_train(self, extract=True, positive_only=False):
        return self.load_split(split='train', extract=extract, positive_only=positive_only)

    def load_valid(self, extract=True, positive_only=False):
        return self.load_split(split='val', extract=extract, positive_only=positive_only)

    def reset(self):
        pass


class GameSequenceLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample'):
        """
        """
        self.config = dataset.config
        self.dataset = dataset  # not used
        self.root_dir = os.path.join(os.path.join(data_dir, self.dataset.config['preproc_dir']), str(self.dataset.fold_index))
        self.extractor = extractor
        self.batch_size = batch_size  # not used
        self.mode = mode
        self.batch_index = 0

        games = self.dataset.config['data_config']['game_ids']

        self.x = []
        self.t = []
        self.z = []
        self.val_x = []
        self.val_t = []
        self.val_z = []

        for game in tqdm(games):
            try:
                x = pd.read_pickle(os.path.join(self.root_dir, '%s_x.pkl' % game))
                t = np.load(os.path.join(self.root_dir, '%s_t.npy' % game))
                # z = np.load(os.path.join(self.root_dir, '%s_z.npy' % game))
                val_x = pd.read_pickle(os.path.join(self.root_dir, '%s_val_x.pkl' % game))
                val_t = np.load(os.path.join(self.root_dir, '%s_val_t.npy' % game))
                # val_z = np.load(os.path.join(self.root_dir, '%s_val_z.npy' % game))
            except IOError:
                continue

            self.x.extend(x)
            self.t.extend(t)
            # self.z.extend(z)
            self.val_x.extend(val_x)
            self.val_t.extend(val_t)
            # self.val_z.extend(val_z)

        self.val_x = np.array(self.val_x)
        self.val_t = np.array(self.val_t)
        # self.val_z = np.array(self.val_z)

        self.x = np.array(self.x)
        self.t = np.array(self.t)
        # self.z = np.array(self.z)

        self.ind = 0
        self.val_ind = 0
        self.N = int(batch_size)

    def _split(self, inds, fold_index=0):
        if self.config['data_config']['shuffle']:
            np.random.seed(self.config['randseed'])
            np.random.shuffle(inds)
        N = len(inds)
        val_start = np.round(fold_index/self.config['data_config']['N_folds'] * N).astype('int32')
        val_end = np.round((fold_index + 1)/self.config['data_config']['N_folds'] * N).astype('int32')
        val_inds= inds[val_start:val_end]
        train_inds = inds[:val_start] + inds[val_end:]
        return train_inds, val_inds

    def load_valid(self, decode=False, multi=False):
        x = self.val_x
        t = self.val_t
        # z = self.val_z

        if decode:
            labels = np.zeros((t.shape[0]))
            labels[t[:, 1] == 1] = 1
            t = labels

        if multi:
            z = []
            x = x[t == 1]
            for event in x:
                z.append(int(event.anno['label']))

            x = self.get_features(x)
            x = self.get_binarized_features(x)

            z = np.array(z)
            x = np.array(x)

            x, z = shuffle_2_array(x, z)
            return x, z
        else:
            x = self.get_features(x)
            x = self.get_binarized_features(x)

            t = np.array(t)
            x = np.array(x)

            x, t = shuffle_2_array(x, t)
            return x, t

    def load_train(self, decode=False, multi=False):
        x = self.x
        t = self.t
        # z = self.z
        if decode:
            labels = np.zeros((t.shape[0]))
            labels[t[:, 1] == 1] = 1
            t = labels
        if multi:
            z = []
            x = x[t == 1]
            for event in x:
                z.append(int(event.anno['label']))

            x = self.get_features(x)
            x = self.get_binarized_features(x)

            z = np.array(z)
            x = np.array(x)

            x, z = shuffle_2_array(x, z)
            return x, z
        else:
            x = self.get_features(x)
            x = self.get_binarized_features(x)

            t = np.array(t)
            x = np.array(x)

            x, t = shuffle_2_array(x, t)
            return x, t

    def reset(self):
        self.batch_index = 0
        
    def get_features(self, events):
        features = []
        for event in events:
            event_features = []
            event_features.append(event.min_dist_bh_bd)
            event_features.append(event.min_dist_bh_ss)
            event_features.append(event.min_dist_bd_ss)
            event_features.append(event.min_dist_bh_hp)
            event_features.append(event.min_dist_bd_hp)
            event_features.append(event.min_dist_ss_hp)
    
            event_features.append(event.diff_dist_bh_bd_ap)
            event_features.append(event.diff_dist_bh_ss_ap)
            event_features.append(event.diff_dist_bd_ss_ap)
            event_features.append(event.diff_dist_bh_hp_ap)
            event_features.append(event.diff_dist_bd_hp_ap)
            event_features.append(event.diff_dist_ss_hp_ap)
    
            event_features.append(event.diff_dist_bh_bd_ex)
            event_features.append(event.diff_dist_bh_ss_ex)
            event_features.append(event.diff_dist_bd_ss_ex)
            event_features.append(event.diff_dist_bh_hp_ex)
            event_features.append(event.diff_dist_bd_hp_ex)
            event_features.append(event.diff_dist_ss_hp_ex)
    
            event_features.append(event.ave_dist_bh_bd_ap)
            event_features.append(event.ave_dist_bh_ss_ap)
            event_features.append(event.ave_dist_bd_ss_ap)
            event_features.append(event.ave_dist_bh_hp_ap)
            event_features.append(event.ave_dist_bd_hp_ap)
            event_features.append(event.ave_dist_ss_hp_ap)
    
            event_features.append(event.ave_dist_bh_bd_ex)
            event_features.append(event.ave_dist_bh_ss_ex)
            event_features.append(event.ave_dist_bd_ss_ex)
            event_features.append(event.ave_dist_bh_hp_ex)
            event_features.append(event.ave_dist_bd_hp_ex)
            event_features.append(event.ave_dist_ss_hp_ex)

            features.append(event_features)

        features = np.array(features)
        return features

    def get_binarized_features(self, features, bins=5):
        binned_features = pd.DataFrame()
        for ind in range(features.shape[-1]):
            feature = features[:, ind]
            feature = pd.qcut(feature, bins, labels=False)
            feature = pd.get_dummies(feature)
            binned_features = pd.concat([binned_features, feature], axis=1)

        binned_features = binned_features.values
        return binned_features