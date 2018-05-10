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

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')

    def next_batch(self, extract=True, no_anno=False):
        N_pos = int(self.fraction_positive * self.batch_size)
        N_neg = self.batch_size - N_pos
        ret_val = []
        if not no_anno:
            func = [self.dataset.propose_positive_Ta,
                    self.dataset.propose_negative_Ta]
        else:
            func = [self.dataset.propose_Ta]
        Ns = [N_pos, N_neg]
        # anno = func[0]()
        for j in xrange(len(func)):
            for _ in xrange(Ns[j]):
                while True:
                    try:
                        anno = func[j]()
                        e = Event(self.dataset.games[anno['gameid']]['events'][anno['eid']], gameid=anno['gameid'], anno=anno)
                        e.sequence_around_t(anno, self.dataset.tfr)  # EventException
                        e.build_features()
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
                        break

        return (
                np.array(ret_val),
                np.vstack([np.array([[0, 1]]).repeat(N_pos, axis=0),
                np.array([[1, 0]]).repeat(N_neg, axis=0)])
            )

    def load_by_annotations(self, annotations, extract=True):
        """
        no labels returned
        """
        ret_val = []
        ret_labels = []
        self.extractor.augment = False
        for anno in annotations:
            try:
                e = Event(self.dataset.games[anno['gameid']]['events'][anno['eid']], gameid=anno['gameid'], anno=anno)
                e.sequence_around_t(anno, self.dataset.tfr)  # EventException
                e.build_features()
                if extract:
                    # ExtractorException
                    ret_val.append(self.extractor.extract(e))
                else:
                    # just to make sure event not malformed (like
                    # missing player)
                    _ = self.extractor.extract_raw(e)
                    ret_val.append(e)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            except TeamNotFoundException as exc:
                pass
        return ret_val

    def load_split(self, split='val', extract=True, positive_only=False):
        N_pos = 0
        ret_val = []
        ret_labels = []
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
        return np.array(ret_val), np.array(ret_labels)

    def load_train(self, extract=True, positive_only=False):
        return self.load_split(split='train', extract=extract, positive_only=positive_only)

    def load_valid(self, extract=True, positive_only=False):
        return self.load_split(split='val', extract=extract, positive_only=positive_only)

    def _load_event(self, anno, extract, every_K_frame, dont_resolve_basket=False):
        ret_val = []
        ret_gameclocks = []
        ret_frame_idx = []
        event_id = int(anno['eid'])
        game_id = anno['gameid']
        try:
            e = Event(
                self.dataset.games[game_id]['events'][event_id],
                gameid=game_id,
                anno=anno,
                data_config=self.dataset.config
            )
        except TeamNotFoundException:
            # malformed event
            return 0
        N_moments = len(e.moments)
        for i in xrange(0, N_moments, every_K_frame):
            try:
                e = Event(
                    self.dataset.games[game_id]['events'][event_id],
                    gameid=game_id,
                    anno=anno,
                    data_config=self.dataset.config
                )
                game_clock = e.moments[i].game_clock
                quarter = e.moments[i].quarter
                anno = self.dataset._make_annotation(game_id, quarter, game_clock, event_id, self.dataset.tfr)
                e.sequence_around_t(anno, self.dataset.tfr)  # EventException
                e.build_features()

                # just to make sure event not malformed (like
                # missing player)
                _ = self.extractor.extract_raw(e)
                ret_val.append(e)
                ret_gameclocks.append(game_clock)
                ret_frame_idx.append(i)
            except EventException as exc:
                continue
            except ExtractorException as exc:
                continue
            except TeamNotFoundException as exc:
                pass
        if len(ret_val) == 0:  # malformed Event
            ret = 0
        else:
            if extract:
                ret_val = self.extractor.extract_batch(ret_val,dont_resolve_basket=dont_resolve_basket)
            ret = [ret_val, ret_gameclocks, ret_frame_idx]
        return ret

    def load_split_event(self, split, extract, every_K_frame=4):
        if split == 'val':
            split_hash = self.dataset.val_hash
        elif split == 'train':
            split_hash = self.dataset.train_hash
        elif split == 'game':
            split_hash = self.dataset.game_hash
        else:
            raise NotImplementedError()
        if self.event_index == len(split_hash):
            self.event_index = 0
            return None
        vh = split_hash.values()[self.event_index]

        ret_labels = filter(lambda t: t != -1, [i['gameclock'] for i in vh])
        self.extractor.augment = False
        anno = vh[0]
        ret = self._load_event(anno, extract, every_K_frame)
        if ret == 0:  # malformed Event
            ret = 0
        else:
            ret_val, ret_gameclocks, ret_frame_idx = ret
            meta = [vh[0]['eid'], vh[0]['gameid']]
            ret = [ret_val, ret_labels, ret_gameclocks, meta]
        self.extractor.augment = True
        self.event_index += 1
        return ret

    def load_event(self, game_id, event_id, every_K_frame, player_id=None):
        o = self.extractor.augment
        self.extractor.augment = False
        anno = {"gameid": game_id, "eid": event_id}
        ret = self._load_event(anno, True, every_K_frame, dont_resolve_basket=True)
        self.extractor.augment = o
        return ret

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
        self.val_x = []
        self.val_t = []

        for game in tqdm(games):
            try:
                x = pd.read_pickle(os.path.join(self.root_dir, '%s_x.pkl' % game))
                t = np.load(os.path.join(self.root_dir, '%s_t.npy' % game))
                val_x = pd.read_pickle(os.path.join(self.root_dir, '%s_val_x.pkl' % game))
                val_t = np.load(os.path.join(self.root_dir, '%s_val_t.npy' % game))
            except IOError:
                continue

            x = self.get_features(x)
            val_x = self.get_features(val_x)

            self.x.extend(x)
            self.t.extend(t)
            self.val_x.extend(val_x)
            self.val_t.extend(val_t)

        self.val_x = np.array(self.val_x)
        self.val_t = np.array(self.val_t)
        self.x = np.array(self.x)
        self.t = np.array(self.t)

        self.x = self.get_binarized_features(self.x)
        self.val_x = self.get_binarized_features(self.val_x)

        self.x, self.t = shuffle_2_array(self.x, self.t)
        self.val_x, self.val_t = shuffle_2_array(self.val_x, self.val_t)

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

    def next(self):
        """
        """
        if self.mode == 'sample':
            return self.next_batch()
        elif self.mode == 'valid':
            return self.load_valid()
        else:
            raise Exception('unknown loader mode')

    def next_batch(self, extract=True):
        # if self.batch_index == self.dataset_size:
        #     return None

        if self.ind + self.N >= self.x.shape[0]:
            self.ind = 0
            self.x, self.t = shuffle_2_array(self.x, self.t)

        s = list(self.x.shape)
        s[0] = self.batch_size
        x = np.zeros(s)
        t = np.zeros((self.batch_size, 2))
        x[:self.N] = self.x[self.ind:self.ind + self.N]
        t[:self.N] = self.t[self.ind:self.ind + self.N]
        if extract:
            x = self.extractor.extract_batch(x, input_is_sequence=True)
        self.ind += self.N
        return x, t

    def load_valid(self, decode=False):
        x = self.val_x
        t = self.val_t
        if decode:
            labels = np.zeros((t.shape[0]))
            labels[t[:, 1] == 1] = 1
            t = labels

        return x, t

    def load_train(self, decode=False):
        x = self.x
        t = self.t
        if decode:
            labels = np.zeros((t.shape[0]))
            labels[t[:, 1] == 1] = 1
            t = labels

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