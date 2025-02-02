from __future__ import division
from Constant import Constant
from Moment import Moment, MomentException, PNRException
from Team import TeamNotFoundException

plt = None
import numpy as np
import pandas as pd
import seaborn as sns

from wiens.vis.utils import draw_full_court, draw_half_court
import wiens.config as CONFIG
from wiens.annotation.roles import get_roles
from wiens.annotation.features import *

class EventException(Exception):
    pass

class OneHotException(Exception):
    pass

class FeatureException(Exception):
    pass


def format_pbp(pbp):
    event_str = "Play-By-Play Annotations\n"
    g = pbp.iterrows()
    for eind, pp in pbp.iterrows():
      event_str += '------------Event: %i ---------------\n' % eind
      event_str += str(pp['HOMEDESCRIPTION'])+ " , " +\
              str(pp['VISITORDESCRIPTION'])+ " , "+\
              str(pp['PCTIMESTRING'])+ '\n'
    return event_str

def format_anno(anno):
    event_str = "PnR Annotation \n"
    event_str += '------------Event: %i ---------------\n' % anno['eid']
    return event_str

class Event:
    """A class for handling and showing events"""

    def __init__(self, event, gameid, data_config, anno):
        self.gameid = gameid
        self.home_team_id = event['home']['teamid']
        self.away_team_id = event['visitor']['teamid']
        self.moments = []
        self.pbp = event['playbyplay']
        self.data_config = data_config
        self.anno = anno
        self.start_time = 0
        self.end_time = 999

        for ind, moment in enumerate(event['moments']):
            try:
                moment = Moment(moment, anno=self.anno)
                self.moments.append(moment)
                if moment.game_clock < self.end_time:
                    self.end_time = moment.game_clock
                if moment.game_clock > self.start_time:
                    self.start_time = moment.game_clock
            except MomentException:
                continue
            except PNRException:
                continue
            except TeamNotFoundException:
                raise TeamNotFoundException

        start_moment = self.moments[0]
        start_moment_ids = [player.id for player in start_moment.players]
        home_players = pd.DataFrame(event['home']['players'])
        guest_players = pd.DataFrame(event['visitor']['players'])

        self.home_players = home_players.loc[home_players.playerid.isin(start_moment_ids), :].T.to_dict().values()
        self.guest_players = guest_players.loc[guest_players.playerid.isin(start_moment_ids), :].T.to_dict().values()
        self.players = self.home_players + self.guest_players

        player_ids = [player['playerid'] for player in self.players]
        player_names = ['%s %s' % (player['firstname'], player['lastname']) for player in self.players]
        player_jerseys = [player['jersey'] for player in self.players]
        values = list(zip(player_names, player_jerseys))

        self.player_ids_dict = dict(zip(player_ids, values))
        self._resolve_home_basket()

    def _resolve_home_basket(self):
        """
        hardcoded for the 3 games labelled
        '0021500357' q1 home: 0
        '0021500150' q1 home: 1
        '0021500278' q1 home: 0
        """
        hard_code = {
            '0021500357': 0,
            '0021500150': 1,
            '0021500278': 0,
            '0021500408': 1,
            '0021500009': 1,
            '0021500066': 0,
            '0021500024': 0,
            '0021500196': 0,
            '0021500383': 0,
            '0021500096': 0,
            '0021500075': 0,
            '0021500477': 1,
            '0021500057': 0,
            '0021500188': 0
        }
        self.home_basket = (hard_code[self.gameid] + (self.moments[0].quarter > 2)) % 2

    def is_home_possession(self, moment):
        ball_basket = int(moment.ball.x > 50)
        if ball_basket == self.home_basket: # HOME possession
          return True
        else: # VISITOR possession
          return False

    def truncate_by_following_event(self, event2):
        """
        use the given event to truncate the current  (i.e. do not include the
        trailing frames shown in a later event)
        """
        # trunctate
        end_time_from_e2 = event2['moments'][0][2]
        last_idx = -1
        for idx, moment in enumerate(self.moments):
          if moment.game_clock < end_time_from_e2:
            last_idx = idx
            break
        if last_idx != -1:
          self.moments = self.moments[:last_idx]

    def sequence_around_t(self, anno, tfr, data_config):
        """
        segment [T_a - tfr, T_a + tfr]
        note: when seek_last = True, seek for the last T_a
              (this detail becomes important when game-clock stops within one Event)
        """
        T_a = anno['gameclock']

        T_a_index = -1
        for idx, moment in enumerate(self.moments):
          if moment.game_clock < T_a:
            T_a_index = idx
            break

        if T_a_index == -1:
          # print ('EventException')
          raise EventException('bad T_a, or bad event')

        start_ind = np.max([0, T_a_index-tfr])
        end_ind = np.min([len(self.moments)-1, T_a_index + tfr])

        if end_ind - start_ind != 2*tfr:
          raise EventException('incorrect length')
        self.moments = self.moments[start_ind:end_ind]

        # check if roles have been formed
        self.movement = self.get_movement(limit=False)
        if 'ball_handler' not in anno.keys():
            self.anno = get_roles(annotation=anno, movement=self.movement, data_config=data_config)
            if self.anno is None:
                raise EventException

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info, lines, pred_lines):
        line = lines[0]
        ret = [player_circles, ball_circle, line]
        if i in self.futures[0]:
          frame_ind = self.futures[0].index(i)
          for sample_idx, l in enumerate(pred_lines):
            l.set_ydata(self.futures[2][frame_ind, sample_idx,:,1])
            l.set_xdata(self.futures[2][frame_ind, sample_idx,:,0])
            ret.append(l)
          line.set_ydata(self.futures[1][frame_ind, :, 1])
          line.set_xdata(self.futures[1][frame_ind, :, 0])

        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            try:
              circle.center = moment.players[j].x, moment.players[j].y
            except:
              raise EventException()

            annotations[j].set_position(circle.center)
            clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                         moment.quarter,
                         int(moment.game_clock) % 3600 // 60,
                         int(moment.game_clock) % 60,
                         moment.shot_clock)
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        x = np.arange(Constant.X_MIN, Constant.X_MAX, 1)
        court_center_x = Constant.X_MAX /2
        court_center_y = Constant.Y_MAX /2
        player_of_interest = moment.players[7]

        return ret

    def get_movement(self, limit=False, limit_set=None):
        """
        Get dataframe of event movement
        """
        movement = pd.DataFrame(columns=['player_id', 'team_id', 'x_loc', 'y_loc', 'game_clock', 'color'])
        for moment in self.moments:
            for player in moment.players:
                if (limit) and (player.id in limit_set):
                    movement = movement.append({
                        'player_id': player.id,
                        'team_id': player.team.id,
                        'x_loc': player.x,
                        'y_loc': player.y,
                        'game_clock': moment.game_clock,
                        'color': player.color
                    }, ignore_index=True)
                elif not limit:
                    movement = movement.append({
                        'player_id': player.id,
                        'team_id': player.team.id,
                        'x_loc': player.x,
                        'y_loc': player.y,
                        'game_clock': moment.game_clock,
                        'color': player.color
                    }, ignore_index=True)
            movement = movement.append({
                'player_id': -1,
                'team_id': -1,
                'x_loc': moment.ball.x,
                'y_loc': moment.ball.y,
                'game_clock': moment.game_clock,
                'color': moment.ball.color
            }, ignore_index=True)

        return movement

    def update_movement(self, i, player_circles, ball_circle, annotations, clock_info):
        ret = [player_circles, ball_circle]

        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            try:
                circle.center = moment.players[j].x, moment.players[j].y
                annotations[j].set_position(circle.center)
                clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                    moment.quarter,
                    int(moment.game_clock) % 3600 // 60,
                    int(moment.game_clock) % 60,
                    moment.shot_clock)
                clock_info.set_text(clock_test)

            except Exception as err:
                raise EventException()
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        x = np.arange(Constant.X_MIN, Constant.X_MAX, 1)
        court_center_x = Constant.X_MAX / 2
        court_center_y = Constant.Y_MAX / 2
        player_of_interest = moment.players[7]

        return ret

    def show(self, save_path='', anno=None):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,Constant.X_MAX), ylim=(Constant.Y_MIN,Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        try:
          start_moment = self.moments[0]
        except IndexError as e:
          raise EventException()

        player_dict = self.player_ids_dict

        clock_info = ax.annotate(
            '',
            xy=[Constant.X_CENTER, Constant.Y_CENTER],
            color='black',
            horizontalalignment='center',
            verticalalignment='center'
        )

        annotations = [
            ax.annotate(
                self.player_ids_dict[player['playerid']][1],
                xy=[0, 0],
                color='w',
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='bold'
            )
            for player in self.players
        ]
        player_circles = [
            plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
            for player in start_moment.players
        ]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig,
            self.update_movement,
            fargs=(player_circles, ball_circle, annotations, clock_info),
            frames=len(self.moments),
            interval=Constant.INTERVAL
        )

        court = plt.imread('%s/court.png' % (CONFIG.vis.dir))
        plt.imshow(
            court,
            zorder=0,
            extent=[
                Constant.X_MIN,
                Constant.X_MAX - Constant.DIFF,
                Constant.Y_MAX, Constant.Y_MIN
            ]
        )

        plt.title(format_anno(self.anno))
        if save_path == '':
          plt.show()
        else:
          plt.ioff()
          Writer = animation.writers['ffmpeg']
          writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
          anim.save(save_path, writer)
        plt.clf()

    def show_static(self, save_path='', anno=None, plot_type='half'):
        import matplotlib.pyplot as plt
        if plot_type == 'full':
            fig = plt.figure(figsize=(15, 7.5))
            ax = plt.gca()
            ax = draw_full_court(ax=ax)
            ax.grid(False)  # Remove grid
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim([0, 94])
            ax.set_ylim([-50, 0])
        elif plot_type == 'half':
            fig = plt.figure(figsize=(12, 11))
            ax = plt.gca()
            ax = draw_half_court(ax=ax)
            ax.grid(False)  # Remove grid
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim([-250, 250])
            ax.set_ylim([422.5, -47.5])

        movement = self.get_movement(limit=True, limit_set=[anno['screen_defender']])

        if plot_type == 'half':
            movement.loc[movement.x_loc > 47, 'y_loc'] = movement.loc[movement.x_loc > 47, 'y_loc'].apply(lambda y: 50 - y)
            movement.loc[movement.x_loc > 47, 'x_loc'] = movement.loc[movement.x_loc > 47, 'x_loc'].apply(lambda x: 94 - x)
            movement['x_loc_copy'] = movement['x_loc']
            movement['y_loc_copy'] = movement['y_loc']
            movement['x_loc'] = movement['y_loc_copy'].apply(lambda y: 250 * (1 - (y - 0) / (50 - 0)) + -250 * ((y - 0) / (50 - 0)))
            movement['y_loc'] = movement['x_loc_copy'].apply(lambda x: -47.5 * (1 - (x - 0) / (47 - 0)) + 422.5 * ((x - 0) / (47 - 0)))
            movement = movement.drop('x_loc_copy', axis=1, inplace=False)
            movement = movement.drop('y_loc_copy', axis=1, inplace=False)

        players = movement['player_id'].drop_duplicates(inplace=False).values
        for player in players:
            if player == -1:
                continue
            player_movement = movement.loc[movement.player_id == player, :]
            player_color = player_movement['color'].values[0]
            cm = sns.light_palette(player_color, as_cmap=True)
            if plot_type == 'full':
                plt.scatter(
                    player_movement.x_loc,
                    -player_movement.y_loc,
                    c=-player_movement.game_clock,
                    cmap=cm,
                    s=100,
                    zorder=1,
                    alpha=1,
                    edgecolors='none'
                )
            elif plot_type == 'half':
                plt.scatter(
                    player_movement.x_loc,
                    player_movement.y_loc,
                    c=-player_movement.game_clock,
                    cmap=cm,
                    s=200,
                    zorder=1,
                    alpha=1,
                    edgecolors='none'
                )
        fig.show()
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def build_features(self):
        """
        Use defined pairwise distances from Wiens paper,
         to create feature set from movement of roles in pnr.

        Features:
            - min distance from player a to player b over entire pnr window
            - difference in distance from a to b from start to end of approach window
            - average distance from a to b over approach window
            - difference in distance from a to b from start to end of execution window
            - average distance from a to b over execution window

        These 5 features are calculated for relationships between:
            - ball handler and ball defender
            - ball handler and screen setter
            - ball defender and screen setter
            - ball handler and hoop
            - ball defender and hoop
            - screen setter and hoop

        Once the 30 features are extracted, each feature is binned into 5 bins, making 150 binary features.
        This is done in a later step, as these features are computed annotation by annotation.
        """
        try:
            self.movement = get_hoop_location(self.movement, self.anno)

            self.min_dist_bh_bd = get_min_distance(self.movement, self.anno, 'ball_handler', 'ball_defender')
            self.min_dist_bh_ss = get_min_distance(self.movement, self.anno, 'ball_handler', 'screen_setter')
            self.min_dist_bd_ss = get_min_distance(self.movement, self.anno, 'ball_defender', 'screen_setter')
            self.min_dist_bh_hp = get_min_distance(self.movement, self.anno, 'ball_handler', 'hoop')
            self.min_dist_bd_hp = get_min_distance(self.movement, self.anno, 'ball_defender', 'hoop')
            self.min_dist_ss_hp = get_min_distance(self.movement, self.anno, 'screen_setter', 'hoop')

            self.diff_dist_bh_bd_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_handler', 'ball_defender')
            self.diff_dist_bh_ss_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_handler', 'screen_setter')
            self.diff_dist_bd_ss_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_defender', 'screen_setter')
            self.diff_dist_bh_hp_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_handler', 'hoop')
            self.diff_dist_bd_hp_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_defender', 'hoop')
            self.diff_dist_ss_hp_ap = get_diff_distance(self.movement, 'approach', self.anno, 'screen_setter', 'hoop')

            self.diff_dist_bh_bd_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_handler', 'ball_defender')
            self.diff_dist_bh_ss_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_handler', 'screen_setter')
            self.diff_dist_bd_ss_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_defender', 'screen_setter')
            self.diff_dist_bh_hp_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_handler', 'hoop')
            self.diff_dist_bd_hp_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_defender', 'hoop')
            self.diff_dist_ss_hp_ex = get_diff_distance(self.movement, 'execution', self.anno, 'screen_setter', 'hoop')

            self.ave_dist_bh_bd_ap = get_average_distance(self.movement, 'approach', self.anno, 'ball_handler', 'ball_defender')
            self.ave_dist_bh_ss_ap = get_average_distance(self.movement, 'approach', self.anno, 'ball_handler', 'screen_setter')
            self.ave_dist_bd_ss_ap = get_average_distance(self.movement, 'approach', self.anno, 'ball_defender', 'screen_setter')
            self.ave_dist_bh_hp_ap = get_average_distance(self.movement, 'approach', self.anno, 'ball_handler', 'hoop')
            self.ave_dist_bd_hp_ap = get_average_distance(self.movement, 'approach', self.anno, 'ball_defender', 'hoop')
            self.ave_dist_ss_hp_ap = get_average_distance(self.movement, 'approach', self.anno, 'screen_setter', 'hoop')

            self.ave_dist_bh_bd_ex = get_average_distance(self.movement, 'execution', self.anno, 'ball_handler', 'ball_defender')
            self.ave_dist_bh_ss_ex = get_average_distance(self.movement, 'execution', self.anno, 'ball_handler', 'screen_setter')
            self.ave_dist_bd_ss_ex = get_average_distance(self.movement, 'execution', self.anno, 'ball_defender', 'screen_setter')
            self.ave_dist_bh_hp_ex = get_average_distance(self.movement, 'execution', self.anno, 'ball_handler', 'hoop')
            self.ave_dist_bd_hp_ex = get_average_distance(self.movement, 'execution', self.anno, 'ball_defender', 'hoop')
            self.ave_dist_ss_hp_ex = get_average_distance(self.movement, 'execution', self.anno, 'screen_setter', 'hoop')

            self.min_dist_bh_sd = get_min_distance(self.movement, self.anno, 'ball_handler', 'screen_defender')
            self.min_dist_bd_sd = get_min_distance(self.movement, self.anno, 'ball_defender', 'screen_defender')
            self.min_dist_ss_sd = get_min_distance(self.movement, self.anno, 'screen_setter', 'screen_defender')
            self.min_dist_sd_hp = get_min_distance(self.movement, self.anno, 'screen_defender', 'hoop')

            self.diff_dist_bh_sd_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_handler', 'screen_defender')
            self.diff_dist_bd_sd_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_defender', 'screen_defender')
            self.diff_dist_ss_sd_ap = get_diff_distance(self.movement, 'approach', self.anno, 'screen_setter', 'screen_defender')
            self.diff_dist_sd_hp_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_handler', 'ball_defender')

            self.diff_dist_bh_sd_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_handler', 'screen_defender')
            self.diff_dist_bd_sd_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_defender', 'screen_defender')
            self.diff_dist_ss_sd_ex = get_diff_distance(self.movement, 'execution', self.anno, 'screen_setter', 'screen_defender')
            self.diff_dist_sd_hp_ex = get_diff_distance(self.movement, 'execution', self.anno, 'screen_defender', 'hoop')

            self.ave_dist_bh_sd_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_handler', 'screen_defender')
            self.ave_dist_bd_sd_ap = get_diff_distance(self.movement, 'approach', self.anno, 'screen_setter', 'screen_defender')
            self.ave_dist_ss_sd_ap = get_diff_distance(self.movement, 'approach', self.anno, 'screen_setter', 'screen_defender')
            self.ave_dist_sd_hp_ap = get_diff_distance(self.movement, 'approach', self.anno, 'ball_handler', 'ball_defender')

            self.ave_dist_bh_sd_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_handler', 'screen_defender')
            self.ave_dist_bd_sd_ex = get_diff_distance(self.movement, 'execution', self.anno, 'ball_defender', 'screen_defender')
            self.ave_dist_ss_sd_ex = get_diff_distance(self.movement, 'execution', self.anno, 'screen_setter', 'screen_defender')
            self.ave_dist_sd_hp_ex = get_diff_distance(self.movement, 'execution', self.anno, 'screen_defender', 'hoop')

        except Exception:
            raise FeatureException('Bad movement')


def convert_time(time):
    return '%s:%s' % (int(time/60), int(time % 60))
