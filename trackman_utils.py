import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def agg_statcast_pitchers(pitcher, min_batters_faced):
    """ 
    For a given pitcher with statcast data from Baseball Savant, compute stats. Function is meant to be applied on
    the dataframe with all events from 2020 which is grouped by pitcher.
    
    pitcher [Pandas dataframe]: Dataframe of events for a single pitcher.
    min_batters_faced [int]: only pitchers who have faced >= min_batters_faced will have non-NAN results.
    
    Returns: pandas.Series object with putaway, whiff, k, bb, and woba.
    """
    appearances = len(pitcher[(pitcher['description'].isin(['hit_by_pitch','hit_into_play','hit_into_play_no_out',
     'hit_into_play_score'])) | (pitcher['events'].isin(['strikeout', 'walk']))])
    if appearances < min_batters_faced:
        return pd.Series({'putaway_rate': float('NaN'), 'whiff_rate': float('NaN'), 'k_rate': float('NaN'), 
                         'bb_rate': float('NaN'), 'woba': float('NaN')})
    pitcher_2str = pitcher[pitcher['strikes']==2]
    putaways = len(pitcher_2str[pitcher_2str['description'].isin(['swinging_strike','swinging_strike_blocked', 
                                                                     'called_strike'])])
    putaway_rate = putaways/len(pitcher_2str)
    whiffs = len(pitcher[pitcher['description'].isin(['swinging_strike','swinging_strike_blocked'])])
    swings = len(pitcher[pitcher['description'].isin(['swinging_strike','swinging_strike_blocked',
                    'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_tip', 'foul', 'foul_bunt'])])
    whiff_rate = whiffs/swings
    strikeouts = len(pitcher[pitcher['events']=='strikeout'])
    k_rate = strikeouts/appearances
    walks = len(pitcher[pitcher['events']=='walk'])
    bb_rate = walks/appearances
    woba = pitcher['woba_value'].sum()/pitcher['woba_denom'].sum()
    return pd.Series({'putaway_rate': putaway_rate, 'whiff_rate': whiff_rate, 'k_rate': k_rate, 'bb_rate': bb_rate, 
                      'woba': woba})

def percentile(val, stats_by_pitcher, stat):
    """ Calculates percentile of val for stat across all qualified pitchers.
    
    Returns: Float representing percentile. """
    sorted_rates = stats_by_pitcher.sort_values(stat, ascending=True)
    i=0
    while val > sorted_rates.iloc[i][stat]:
        i += 1
    return i/len(sorted_rates)


def compute_putaway(df, calls_by_pitch):
    """ Given a dataframe df representing a pitcher's events and a groupby (by pitch), compute putaway rate. 
    
    Returns: Float representing putaway rate. """
    total = 0
    total_k = 0
    for pitch in set(df['tagged_pitch_type']):
        if pitch in calls_by_pitch:
            for row in calls_by_pitch[pitch].index:
                if row == 'StrikeSwinging' or row == 'StrikeCalled':
                    total_k += calls_by_pitch[pitch][row]
                total += calls_by_pitch[pitch][row]
    putaway_rate = total_k/total
    return putaway_rate


def parse_woba(row, weights):
    """ Given a row representing a single event, and wOBA weights, return the wOBA value for row. 
    
    Returns: Float representing wOBA. """
    if row.play_result != 'Undefined':
        return weights[row.play_result]
    else:
        if row.k_or_bb != 'Undefined':
            return weights[row.k_or_bb]
        else:
            return float('NaN')
     
    
def parse_woba_denom(row, denoms):
    """ Given a row representing a single event, and wOBA denoms, return the wOBA denominator for row. 
    
    Returns: Float representing wOBA denominator. """
    if row.play_result != 'Undefined':
        return denoms[row.play_result]
    else:
        if row.k_or_bb != 'Undefined':
            return denoms[row.k_or_bb]
        else:
            return 0

        
def parse_whiff(row, calls_by_pitch):
    """ Given a row representing a pitch, and a groupby (by pitch), return the whiff rate for the pitch. 
    
    Returns: Float representing whiff rate. """
    swings = 0
    whiffs = 0
    if row.name in calls_by_pitch:
        if 'StrikeSwinging' in calls_by_pitch[row.name]:
            whiffs = calls_by_pitch[row.name]['StrikeSwinging']
            swings += whiffs
        if 'FoulBall' in calls_by_pitch[row.name]:
            swings += calls_by_pitch[row.name]['FoulBall']
        if 'InPlay' in calls_by_pitch[row.name]:
            swings += calls_by_pitch[row.name]['InPlay']
        if swings > 0:
            return whiffs/swings
        else:
            return float('NaN')
    else:
        return float('NaN')
    
    
def parse_whiff_statcast(row, all_stats):
    """ Same as parse_whiff() but for statcast data. """
    all_stats_pitch = all_stats[all_stats['pitch_type']==row.name]
    whiffs = len(all_stats_pitch[all_stats_pitch['description'].isin(['swinging_strike','swinging_strike_blocked'])])
    swings = len(all_stats_pitch[all_stats_pitch['description'].isin(['swinging_strike','swinging_strike_blocked',
                    'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_tip', 'foul', 'foul_bunt'])])
    if swings==0:
        return float('NaN')
    whiff_rate = whiffs/swings
    return whiff_rate
    
    
def parse_putaway(row, calls_by_pitch, statcast=False):
    """ Parase putaway using same style as above. """
    total_k = 0
    total = 0
    if row.name in calls_by_pitch:
        for outcome in calls_by_pitch[row.name].index:
            if not statcast:
                if outcome == 'StrikeSwinging' or outcome == 'StrikeCalled':
                    total_k += calls_by_pitch[row.name][outcome]
            else:
                if outcome in ['swinging_strike', 'swinging_strike_blocked', 'called_strike']:
                    total_k += calls_by_pitch[row.name][outcome]
            total += calls_by_pitch[row.name][outcome] 
        return total_k/total
    else:
        return float('NaN')
    
    
def parse_prop(row, df, statcast=False):
    """ Parse pitch frequency. """
    if not statcast:
        return len(df[df['tagged_pitch_type']==row.name])/len(df)
    else:
        return len(df[df['pitch_type']==row.name])/len(df)
    
    
def whiff_by_height(pitches, df, heights, putaway=False):
    """ Given pitches of interest and heights delineation and a df representing all pitches, compute stats
    per-pitch and per-height and plot. Universal strike zone docs described in notebook. """
    
    # "universal" strike zone params
    lowerb = 18.29/12
    upperb = lowerb+(25.79/12)
    leftb = -9.97/12
    rightb = leftb +(19.94/12)
    
    for pitch in pitches:
        freqs = {}
        for height in heights:
            print('\nINFO FOR ' + str(pitch) + ' at ' + str(height))
            df_pitch = df[(df['tagged_pitch_type']==pitch) & (df['plate_loc_height'] > height[0]) &
                         (df['plate_loc_height'] <= height[1])]
            whiffs = len(df_pitch[df_pitch['pitch_call']=='StrikeSwinging'])
            swings = len(df_pitch[df_pitch['pitch_call'].isin(['StrikeSwinging', 'InPlay', 'FoulBall'])])
            if swings > 0:
                whiff_rate = whiffs/swings
            else:
                whiff_rate = float('NaN')
            clean = len(df_pitch[df_pitch['pitch_call'].isin(['StrikeSwinging', 'StrikeCalled'])])
            chances = len(df_pitch)
            called_str = len(df_pitch[df_pitch['pitch_call']=='StrikeCalled'])
            strikes = len(df_pitch[(df_pitch['plate_loc_height'] >= lowerb) & (df_pitch['plate_loc_height'] <= upperb) &
                         (df_pitch['plate_loc_side'] >= leftb) & (df_pitch['plate_loc_side'] <= rightb)])
            print(pitch + ' clean rate: ' + str(clean/chances))
            print(pitch + ' whiff rate: ' + str(whiff_rate))
            print(pitch + ' whiffs: ' + str(whiffs))
            print(pitch + ' called strikes: ' + str(called_str))
            print(pitch + ' swings: ' + str(swings))
            print(pitch + ' strikes: ' + str(strikes))
            print(pitch + ' chances: ' + str(chances))
            freqs[height] = len(df_pitch)

            # plot
            df_pitch.plot(x='plate_loc_side', y='plate_loc_height', kind='scatter')
            ax = plt.gca()
            rect = patches.Rectangle((-9.97/12,18.29/12),19.94/12,25.79/12,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.xlim([-3, 3])
            plt.ylim([0, 6])
            plt.show()
        total = sum(freqs.values())
        for height in freqs:
            freqs[height] /= total
        print("frequencies for " + pitch + ' by height: ' + str(freqs))
        