"""annotation.py

Usage:
    annotation.py <f_data_config>

Arguments:
    <f_data_config>  example 'pnrs.yaml'

Example:
    python annotation.py pnrs.yaml
"""
import warnings
warnings.filterwarnings('ignore')

import math
import os
import cPickle as pkl
from docopt import docopt
import yaml
from tqdm import tqdm

import wiens.config as CONFIG
from wiens.annotation.roles import *
from wiens.annotation.actions import *

movement_headers = [
    "team_id",
    "player_id",
    "x_loc",
    "y_loc",
    "radius",
    "game_clock",
    "shot_clock",
    "quarter",
    "game_id",
    "event_id"
]


def gameclock_to_str(gameclock):
    """
    Float to minute:second
    """
    return '%d:%d%d' % (int(gameclock/60), int((gameclock % 60)/10), int((gameclock % 60) % 10))

def get_type(annotation):
    if annotation['over'] == 'x':
        return 0
    elif annotation['under'] == 'x':
        return 1
    elif annotation['switch'] == 'x':
        return 2
    elif annotation['trap'] == 'x':
        return 3
    else:
        return None


def read_annotations(fpath):
    game_annotations = pd.read_csv(fpath)
    game_annotations = game_annotations.groupby('eid')
    annotations = {}
    for eid, event_annotations in game_annotations:
        anno = []
        for ind, annotation in event_annotations.iterrows():
            anno.append(annotation)
        try:
            annotations[int(eid)] = anno
        except Exception:
            continue
    return annotations


def read_annotation_from_raw(fpath, game_id):
    data = pkl.load(open(fpath, 'rb'))
    annotations = {}
    data = pd.DataFrame(data)
    data['gameid'] = '00' + data['gameid'].astype(int).astype(str)
    event_ids = data.loc[:,'eid'].drop_duplicates(inplace=False).values
    data = data.loc[data.gameid == game_id, :]

    for event in event_ids:
        eid_annotations = data.loc[data.eid == event, :].to_dict(orient='records')
        if len(eid_annotations) > 0:
            annotations[int(event)] = eid_annotations

    return annotations


def prepare_gt_file_from_raw_label_dir(pnr_dir, game_dir, game_id=None):
    gt = []
    if game_id is None:
        all_raw_f = filter(lambda s:'raw-' in s,os.listdir(pnr_dir))
    else:
        all_raw_f = filter(lambda s: 'raw-%s' % game_id in s, os.listdir(pnr_dir))
    for pnr_anno_ind in xrange(len(all_raw_f)):
        game_anno_base = all_raw_f[pnr_anno_ind]
        if not os.path.isfile(os.path.join(pnr_dir,game_anno_base)):
            continue
        game_id = game_anno_base.split('.')[0].split('-')[1]
        raw_data = pd.read_pickle(os.path.join(game_dir, game_id+'.pkl'))
        fpath = os.path.join(pnr_dir, game_anno_base)
        anno = read_annotations(fpath)
        for k, v in anno.items():
            if len(v) == 0:
                continue
            q = raw_data['events'][k]['quarter']
            gt_entries = []
            annotations = v
            for annotation in annotations:
                anno_type = get_type(annotation)
                if anno_type is None:
                    continue
                gt_entries.append({
                    'gameid': game_id,
                    'quarter': q,
                    'gameclock': annotation['gameclock'],
                    'gameclock_str': annotation['gameclock_str'],
                    'eid': k,
                    'label': anno_type
                })
            gt += gt_entries
    return gt


def script_anno_rev0():
    gt = prepare_gt_file_from_raw_label_dir(pnr_dir, game_dir)
    pkl.dump(gt, open(os.path.join(pnr_dir,'gt/rev0.pkl'),'wb'))
    annotations = pd.DataFrame(gt)
    annotations.to_csv(os.path.join(pnr_dir, 'gt/annotations.csv'), index=False)


def get_annotation_movement():
    """
    Segment actions of players before and after the screen time as actions
    Get the movement of individual actions for post processing for trajectory2vec
    """
    annotations = pd.read_csv(os.path.join(pnr_dir, 'roles/annotations.csv'))
    game_ids = annotations.loc[:,'gameid'].drop_duplicates(inplace=False).values

    missed_count = 0
    trajectories = []
    for game_id in tqdm(game_ids):
        game = pd.read_pickle(os.path.join(game_dir, '00' + str(int(game_id)) + '.pkl'))
        game_annotations = annotations.loc[annotations.gameid == game_id, :]
        for ind, annotation in game_annotations.iterrows():
            moments = []
            movement_data = game['events'][int(annotation['eid'])]['moments']
            for moment in movement_data:
                for player in moment[5]:
                    player.extend([moment[2], moment[3], moment[0], game_id, annotation['eid']])
                    player = player[:len(movement_headers)]
                    moments.append(player)

            try:
                movement_data = pd.DataFrame(data=moments, columns=movement_headers)
                movement_data = movement_data.drop_duplicates(subset=['game_clock', 'shot_clock', 'quarter', 'player_id'], inplace=False)
                annotation_trajectories = get_actions(annotation, movement_data, data_config)
                if annotation_trajectories is None:
                    missed_count += 1
                    continue
                else:
                    trajectories.append(annotation_trajectories)
            except Exception as err:
                continue
    print('Missed Count: %s annotations' % (str(missed_count)))
    pkl.dump(trajectories, open(os.path.join(pnr_dir, 'roles/trajectories.pkl'), 'wb'))


def annotate_roles():
    """
    Use underlying logic to determine:
    - ball-handler
    - ball-defender
    - screen-setter
    - screen-defender
    """
    annotations_with_roles = pd.DataFrame()

    if not os.path.exists('%s/roles/' % pnr_dir):
        os.makedirs('%s/roles/' % pnr_dir)

    missed_count = 0
    annotations = pd.read_csv(os.path.join(pnr_dir, 'gt/annotations.csv'))
    game_ids = annotations.loc[:,'gameid'].drop_duplicates(inplace=False).values
    for game_id in tqdm(game_ids):
        count = 0
        print('\n')

        game = pd.read_pickle(os.path.join(game_dir, '00' + str(game_id) + '.pkl'))
        game_annotations = annotations.loc[annotations.gameid == game_id, :]
        for ind, annotation in game_annotations.iterrows():
            count += 1
            print('Game Annotations: %s/%s' % (count, len(game_annotations)))
            moments = []
            movement_data = game['events'][annotation['eid']]['moments']
            for moment in movement_data:
                for player in moment[5]:
                    player.extend([moment[2], moment[3], moment[0], game_id, annotation['eid']])
                    player = player[:len(movement_headers)]
                    moments.append(player)

            try:
                movement_data = pd.DataFrame(data=moments, columns=movement_headers)
                annotation = get_roles(annotation, movement_data, data_config)
                if annotation is None:
                    missed_count += 1
                    continue
                annotations_with_roles = annotations_with_roles.append(annotation)
            except Exception as err:
                continue
    print('Missed %s annotations' % str(missed_count))
    annotations_with_roles.to_csv(os.path.join(pnr_dir, 'roles/annotations.csv'), index=False)
    gt = annotations_with_roles.to_dict(orient='records')
    pkl.dump(gt, open(os.path.join(pnr_dir, 'roles/rev0.pkl'), 'wb'))


def raw_to_gt_format():
    """
    Read raw annotation files to convert to output similar to the output from make_raw_from_untrained
    Use for future comparison purposes
    """
    if not os.path.exists('%s/raw/' % pnr_dir):
        os.makedirs('%s/raw/' % pnr_dir)

    annotations = pd.read_csv(os.path.join(pnr_dir, 'gt/annotations.csv'))
    game_ids = annotations.loc[:,'gameid'].drop_duplicates(inplace=False).values
    for game_id in game_ids:
        game_annotations = annotations.loc[annotations.gameid == game_id, ['eid', 'gameclock', 'gameclock_str', 'quarter']]
        game_annotations.to_csv('%s/raw/raw-00%s.csv' % (pnr_dir, game_id),index=False)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    from wiens.data.constant import game_dir
    pnr_dir = os.path.join(game_dir, 'pnr-annotations')
    # script_anno_rev0()
    # annotate_roles()
    get_annotation_movement()
    # raw_to_gt_format()
