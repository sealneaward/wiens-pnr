from __future__ import division
import cPickle as pickle
import yaml
import os
import numpy as np
import pandas as pd
import cPickle as pkl
import time
from tqdm import tqdm

from wiens.vis.Event import Event, EventException, FeatureException, get_diff_distance, get_min_distance, get_average_distance, get_hoop_location
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

    def _load_event(self, anno, extract, every_K_frame, dont_resolve_basket=False):
        ret_val = []
        ret_gameclocks = []
        ret_frame_idx = []
        event_id = anno['eid']
        game_id = anno['gameid']
        try:
            e = Event(
                self.dataset.games[anno['gameid']]['events'][int(anno['eid'])],
                anno=anno,
                gameid=anno['gameid'],
                data_config=self.dataset.config
            )
        except TeamNotFoundException:
            # malformed event
            return 0
        N_moments = len(e.moments)
        ind = 0
        for i in xrange(0, N_moments, every_K_frame):
            try:
                print ('%i/%i' % (ind, int(N_moments / every_K_frame)))
                ind += 1

                e = Event(
                    self.dataset.games[anno['gameid']]['events'][int(anno['eid'])],
                    anno=anno,
                    gameid=anno['gameid'],
                    data_config=self.dataset.config
                )
                game_clock = e.moments[i].game_clock
                quarter = e.moments[i].quarter
                anno = self.dataset._make_annotation(game_id, quarter, game_clock, event_id, self.dataset.tfr)
                e.sequence_around_t(anno, self.dataset.tfr, data_config=self.dataset.config)  # EventException
                e.build_features()

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
            except FeatureException as exc:
                pass
        if len(ret_val) == 0:  # malformed Event
            ret = 0
        else:
            if extract:
                ret_val = self.extractor.extract_batch(ret_val, dont_resolve_basket=dont_resolve_basket)
            ret = [ret_val, ret_gameclocks, ret_frame_idx]
        return ret

    def load_split_event(self, split, extract, every_K_frame=25):
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
        detect_validation = self.dataset.config['data_config']['detect_validation']
        detect_testing = self.dataset.config['data_config']['detect_testing']

        self.x = []
        self.t = []
        self.z = []
        self.val_x = []
        self.val_t = []
        self.val_z = []

        for game in detect_testing:
            if game in games:
                games.remove(game)

        for game in detect_validation:
            if game in games:
                games.remove(game)

        for game in tqdm(games):
            try:
                x = pd.read_pickle(os.path.join(self.root_dir, '%s_x.pkl' % game))
                t = np.load(os.path.join(self.root_dir, '%s_t.npy' % game))
                val_x = pd.read_pickle(os.path.join(self.root_dir, '%s_val_x.pkl' % game))
                val_t = np.load(os.path.join(self.root_dir, '%s_val_t.npy' % game))
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

        return self.get_xy(x, t, decode=decode, multi=multi)

    def load_train(self, decode=False, multi=False):
        x = self.x
        t = self.t
        # z = self.z

        return self.get_xy(x, t, decode=decode, multi=multi)


class WiensSequenceLoader:
    def __init__(self, dataset, extractor, batch_size, mode='sample', raw=False):
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
        detect_validation = self.dataset.config['data_config']['detect_validation']
        detect_testing = self.dataset.config['data_config']['detect_testing']

        self.x = []
        self.t = []
        self.z = []
        self.val_x = []
        self.val_t = []
        self.val_z = []

        for game in detect_testing:
            if game in games:
                games.remove(game)

        for game in detect_validation:
            if game in games:
                games.remove(game)

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


        # self.z = np.array(self.z)

        if raw == True:
            self.x.extend(self.val_x)
            self.t.extend(self.val_t)

            self.val_x, self.val_t = [], []
            self.binned, _ = self.get_xy(self.x, self.t)

        else:
            self.val_x = np.array(self.val_x)
            self.val_t = np.array(self.val_t)
            # self.val_z = np.array(self.val_z)

            self.x = np.array(self.x)
            self.t = np.array(self.t)

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

        return self.get_xy(x, t, decode=decode, multi=multi)

    def load_train(self, decode=False, multi=False):
        x = self.x
        t = self.t
        # z = self.z

        return self.get_xy(x, t, decode=decode, multi=multi)

    def load_event(self, game_id, event_id, every_K_frame):
        ret_val = []
        ret_gameclocks = []

        for ind, event in enumerate(self.x):
            t = self.t[ind]
            binned_features = self.binned[ind]
            if event.anno['eid'] == event_id and event.anno['gameid'] == game_id:
                N_moments = len(event.moments)
                for i in xrange(0, N_moments, every_K_frame):
                    game_clock = event.moments[i].game_clock
                    ret_val.append([binned_features, t])
                    ret_gameclocks.append(game_clock)
        if len(ret_val) > 0:
            ret = [ret_val, ret_gameclocks]
        else:
            ret = 0

        return ret

    def get_xy(self, x, t,  decode=False, multi=False, shuffle=True):
        if decode:
            labels = np.zeros((t.shape[0]))
            labels[t[:, 1] == 1] = 1
            t = labels

        if multi:
            z = []
            x = x[t == 1]
            for event in x:
                z.append(int(event.anno['label']))

            x = self.get_features(x, multi=True)
            x = self.get_binarized_features(x)

            z = np.array(z)
            x = np.array(x)

            x, z = shuffle_2_array(x, z)
            return x, z
        else:
            try:
                x = self.get_features(x)
                x = self.get_binarized_features(x)
            except Exception:
                raise EventException

            t = np.array(t)
            x = np.array(x)

            if shuffle:
                x, t = shuffle_2_array(x, t)
            return x, t

    def reset(self):
        self.batch_index = 0
        
    def get_features(self, events, multi=False):
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

            if multi:
                event_features.append(get_min_distance(event.movement, event.anno, 'ball_handler', 'screen_defender'))
                event_features.append(get_min_distance(event.movement, event.anno, 'ball_defender', 'screen_defender'))
                event_features.append(get_min_distance(event.movement, event.anno, 'screen_setter', 'screen_defender'))
                event_features.append(get_min_distance(event.movement, event.anno, 'screen_defender', 'hoop'))

                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'ball_handler', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'ball_defender', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'screen_setter', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'ball_handler', 'ball_defender'))

                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'ball_handler', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'ball_defender', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'screen_setter', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'screen_defender', 'hoop'))

                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'ball_handler', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'screen_setter', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'screen_setter', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'approach', event.anno, 'ball_handler', 'ball_defender'))

                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'ball_handler', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'ball_defender', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'screen_setter', 'screen_defender'))
                event_features.append(get_diff_distance(event.movement, 'execution', event.anno, 'screen_defender', 'hoop'))

            features.append(event_features)

        features = np.array(features)
        return features

    def get_binarized_features(self, features, bins=5):
        binned_features = pd.DataFrame()
        for ind in range(features.shape[-1]):
            feature = features[:, ind]
            # feature = feature + np.random.normal(0, 1, len(feature))
            try:
                feature = pd.qcut(feature, bins, labels=False)
                feature = pd.get_dummies(feature)
                binned_features = pd.concat([binned_features, feature], axis=1)
            except Exception:
                raise EventException


        binned_features = binned_features.values
        return binned_features