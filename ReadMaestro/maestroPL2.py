import numpy as np
from numpy import linalg as la
from scipy import stats, signal
import matplotlib.pyplot as plt
import math
import copy
import operator
import ReadMaestro.utils.PL2_read
from SessionAnalysis import timeseries
import warnings
import pickle


""" Some non-specific helper functions """


""" This is to move an index in a numpy array or Python list search item to
    index the NEXT element in the array that holds the input "relation" with
    the input "value".  It will NOT output the value of current_index even if
    it matches value!  This search is done starting at "current_index" and
    outputs an absolute index adjusted for this starting point. The search
    will not wrap around from the end to the beginning, but instead will
    terminate and return None if the end is reached without finding the
    correct value. If a negative current_index is input, the function will
    attempt it to a positive index and output a positive index."""
def move_index_next(search_item, value, current_index=0, relation='='):
    next_index = None
    if len(search_item) < 2:
        # Only one item, cannot be a next index
        return next_index

    # Make dictionary to find functions for input relation operator
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}
    if relation not in ops.keys():
        raise ValueError("Input 'relation' must be '>', '<', '>=', '<=' or '='")

    # Adjust negative index to positive
    if current_index < 0:
        current_index = current_index + len(search_item)

    # Check for index of matching value in search_item
    for index, vals in zip(range(current_index + 1, len(search_item), 1), search_item[current_index + 1:]):
        if ops[relation](vals, value):
            next_index = index
            break

    return(next_index)


""" This is to move an index in a numpy array or Python list search item to
    index the PREVIOUS element in the array that holds the input "relation" with
    the input "value".  It will NOT output the value of current_index even if
    it matches value!  This search is done starting at "current_index" and
    outputs an absolute index adjusted for this starting point.  The search
    will not wrap around from the beginning to the end, but instead will
    terminate and return None if the beginning is reached without finding the
    correct value. If a negative current_index is input, the function will
    attempt it to a positive index and output a positive index."""
def move_index_previous(search_item, value, current_index=0, relation='='):
    previous_index = None
    if len(search_item) < 2:
        # Only one item, cannot be a next index
        return previous_index

    # Make dictionary to find functions for input relation operator
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}
    if relation not in ops.keys():
        raise ValueError("Input 'relation' must be '>', '<', '>=', '<=' or '='")

    # Adjust negative index to positive
    if current_index < 0:
        current_index = current_index + len(search_item)

    # Check for index of matching value in search_item
    for index, vals in zip(range(current_index - 1, -1, -1), search_item[current_index - 1::-1]):
        if ops[relation](vals, value):
            previous_index = index
            break

    return(previous_index)


""" This is to find the index of first occurence of some value in a numpy array
    or python list that satisfies the input relation with the input value.
    Returns None if value isn't found, else returns it's index. """
def find_first_value(search_array, value, relation='='):
    index_out = None

    # Make dictionary to find functions for input relation operator
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}
    if relation not in ops.keys():
        raise ValueError("Input 'relation' must be '>', '<', '>=', '<=' or '='")

    # Check if search_array is iterable, and if not assume it is scalar and check equality
    try:
        for index, vals in enumerate(search_array):
            if ops[relation](vals, value):
                index_out = index
                break
    except TypeError:
        if ops[relation](search_array, value):
            index_out = 0

    return(index_out)


def convert_to_maestroPL2(fname_maestro, fname_PL2, save_name=None):
    maestro_data = maestro_read.load_directory(fname_maestro)
    pl2_reader = PL2_read.PL2Reader(fname_PL2)
    maestro_PL2_data = make_maestro_PL2_data(maestro_data)
    maestro_PL2_data = assign_trial_events(maestro_PL2_data, pl2_reader)

    if save_name is not None:
        with open(save_name, 'wb') as fp:
            pickle.dump(maestro_PL2_data, fp)

    return maestro_PL2_data


class MaestroTarget(object):
    """ Stores target position and velocity as input from maestro_data.  It can
        then compute target velocity on the fly from position (which may not be
        the same as the input velocity, which is given as "commanded" velocity
        from the MaestroTargetCalculator in maestro_read.py) and can use this to
        find target event times aligned to the refresh rate. Data are COPIED
        from maestro_data so it can be deleted afterwards. """

    def __init__(self, maestro_data, target_num, input_commanded_velocity=True):
        self.n_time_points = maestro_data['header']['num_scans'] - 1
        self.position = np.zeros((2, maestro_data['header']['num_scans']))
        self.velocity = np.zeros((2, maestro_data['header']['num_scans']))
        self.commanded_velocity = input_commanded_velocity
        self.frame_refresh_time = 1000.0 / maestro_data['header']['display_framerate']
        self.position[0, :] = maestro_data['horizontal_target_position'][target_num, :]
        self.position[1, :] = maestro_data['vertical_target_position'][target_num, :]
        self.velocity[0, :] = maestro_data['horizontal_target_velocity'][target_num, :]
        self.velocity[1, :] = maestro_data['vertical_target_velocity'][target_num, :]

    def get_next_refresh(self, from_time, n_forward=1):
        n_found = 0
        from_time = np.ceil(from_time).astype('int')
        for t in range(from_time, self.n_time_points + 1, 1):
            if t % self.frame_refresh_time < 1:
                n_found += 1
                if n_found == n_forward:
                    return t
        return None

    def get_last_refresh(self, from_time, n_back=1):
        n_found = 0
        from_time = np.floor(from_time).astype('int')
        for t in range(from_time, -1, -1):
            if t % self.frame_refresh_time < 1:
                n_found += 1
                if n_found == n_back:
                    return t
        return None

    def get_next_change_time(self, time, axis=None):
        # This finds the first time FOLLOWING a nan.  Will skip times before a nan.
        # t_start = np.where(np.isnan(maestro_PL2_data[100]['targets'][0].position[1, :]))[0][-1] + 1
        # np.diff(maestro_PL2_data[100]['targets'][0].position[1, t_start:]) > 0
        pass

    def get_next_position_threshold(self, time, axis=None):
        pass

    def get_next_velocity_threshold(self, time, axis=None, velocity_data='commanded'):
        pass

    def get_next_acceleration_threshold(self, time, axis=None):
        pass

    def velocity_from_position(self, remove_transients=True):
        calculated_velocity = np.copy(self.position)
        transient_index = [[], []]
        last = 0
        next = self.get_next_refresh(1)
        while next is not None:
            calculated_velocity[0, last:next] = np.linspace(calculated_velocity[0, last], calculated_velocity[0, next], next - last, endpoint=False)
            calculated_velocity[1, last:next] = np.linspace(calculated_velocity[1, last], calculated_velocity[1, next], next - last, endpoint=False)

            if (np.all(np.isnan(self.velocity[:, next-1])) or np.all(self.velocity[:, next-1] == 0)) and np.any(self.velocity[:, next] != 0):
                transient_index[0].append(last)
                transient_index[1].append(next)

            if (np.all(np.isnan(self.velocity[:, next])) or np.all(self.velocity[:, next] == 0)) and np.any(self.velocity[:, next-1] != 0):
                transient_index[0].append(last)
                transient_index[1].append(next)

            last = next
            next = self.get_next_refresh(next + 1)

        if last < self.n_time_points:
            calculated_velocity[0, last+1:] = calculated_velocity[0, last]
            calculated_velocity[1, last+1:] = calculated_velocity[1, last]

        calculated_velocity[:, 0:-1] = np.diff(calculated_velocity, axis=1) * 1000
        calculated_velocity[:, -1] = calculated_velocity[:, -2]
        if remove_transients:
            for window in range(0, len(transient_index[0])):
                calculated_velocity[:, transient_index[0][window]:transient_index[1][window]] = self.velocity[:, transient_index[0][window]:transient_index[1][window]]
        calculated_velocity[np.isnan(calculated_velocity)] = 0

        return calculated_velocity

    def acceleration_from_velocity(self, velocity_data='commanded'):
        pass

    def readcxdata_velocity(self):
        pass


def make_maestro_PL2_data(maestro_data):
    """ This trims the fat off of the maestro_data file and combines it with the
        PL2 file data. Data from maestro_data is COPIED, and so maestro_data
        can be deleted afterwards. """

    maestro_PL2_data = [{} for x in range(len(maestro_data))]
    for trial in range(0, len(maestro_PL2_data)):
        maestro_PL2_data[trial]['filename'] = copy.copy(maestro_data[trial]['filename'])
        maestro_PL2_data[trial]['trial_name'] = copy.copy(maestro_data[trial]['header']['name'])
        if maestro_data[trial]['header']['version'] >= 21: # Maestro 4.0
            maestro_PL2_data[trial]['set_name'] = copy.copy(maestro_data[trial]['header']['set_name'])
            maestro_PL2_data[trial]['sub_set_name'] = copy.copy(maestro_data[trial]['header']['sub_set_name'])
            maestro_PL2_data[trial]['timestamp'] = copy.copy(maestro_data[trial]['header']['timestamp'])
        else:
            maestro_PL2_data[trial]['set_name'] = None
            maestro_PL2_data[trial]['sub_set_name'] = None
            maestro_PL2_data[trial]['timestamp'] = None

        maestro_PL2_data[trial]['duration_ms'] = copy.copy(maestro_data[trial]['header']['_num_saved_scans'])
        maestro_PL2_data[trial]['maestro_events'] = [[] for x in maestro_data[trial]['events']]
        maestro_PL2_data[trial]['time_series'] = timeseries.Timeseries(0, maestro_PL2_data[trial]['duration_ms'], 1)
        maestro_PL2_data[trial]['time_series_alignment'] = 0
        for event in range(0, len(maestro_data[trial]['events'])):
            maestro_PL2_data[trial]['maestro_events'][event] = [x * 1000 for x in maestro_data[trial]['events'][event]]
        maestro_PL2_data[trial]['neurons'] = {}
        maestro_PL2_data[trial]['spikes'] = {}
        maestro_PL2_data[trial]['spike_alignment'] = {}
        maestro_PL2_data[trial]['eye_position'] = np.stack([maestro_data[trial]['horizontal_eye_position'], maestro_data[trial]['vertical_eye_position']])
        maestro_PL2_data[trial]['eye_velocity'] = np.stack([maestro_data[trial]['horizontal_eye_velocity'], maestro_data[trial]['vertical_eye_velocity']])

        # Make a target object for each target in the trial
        maestro_PL2_data[trial]['targets'] = []
        for target in range(0, maestro_data[trial]['horizontal_target_position'].shape[0]):
            maestro_PL2_data[trial]['targets'].append(MaestroTarget(maestro_data[trial], target, input_commanded_velocity=True))

    return maestro_PL2_data


def remove_trials_less_than_event(maestro_PL2_data, event_number):

    event_number -= 1
    for trial in range(len(maestro_PL2_data) - 1, -1, -1):
        if not maestro_PL2_data[trial]['maestro_events'][event_number]:
            del maestro_PL2_data[trial]

    return maestro_PL2_data

""" Reads the Plexon strobe inputs by looking for the file saved code from Maestro.
    The number of file saved codes should match the number of trials that were saved.
    It then reads backward to find the last trial start code before the current
    saved code, then forward to find the next trial end code.  These two points
    mark the beginning and end of a Maestro trial.  The start and stop code times,
    and the trial name and trial file name are saved to a list of dictionaries for
    each trial and output in trial_strobe_info. """
# List of strobed attributes:
# 0x02 = trial start
# 0x03 = trial stop
# 14 = lost fixation code
# 15 = aborted due to any reason except lost fixation
# 6 data_saved code
# 0 = null code
def read_pl2_strobes(pl2_reader):

    PL2_strobe_data = pl2_reader.load_channel('strobed')
    strobe_nums = PL2_strobe_data['strobed'] - 32768

    # This function works on the assumption that the following values of Maestro strobes
    # sent to Plexon indicate: the file was saved, trial started, trial ended
    saved_code, start_code, stop_code = 6, 2, 3
    n_trials = 0
    # Track indices of saved trial marker and start - stop code pairs
    current_index = 0
    start_stop = [0, 0]
    word_start_stop = [0, len(strobe_nums)]
    trial_strobe_info = []
    current_index = find_first_value(strobe_nums, saved_code)
    while current_index is not None:
        n_trials += 1
        # print(n_trials)
        # Find next start and stop codes that surround current index and save their time
        strobe_info = {}
        start_stop[0] = move_index_previous(strobe_nums, start_code, current_index, '=')
        if start_stop[0] < start_stop[1] or start_stop[0] is None:
            print('Missed a start code?')
            print(start_stop)
            print(current_index)

        start_stop[1] = move_index_next(strobe_nums, stop_code, start_stop[0], '=')
        strobe_info['trial_start_time'] = PL2_strobe_data['timestamps'][start_stop[0]] * 1000
        strobe_info['trial_stop_time'] = PL2_strobe_data['timestamps'][start_stop[1]] * 1000

        # Read trial name and trial filename and save for output
        word_start_stop[0] = start_stop[0] + 1
        word_start_stop[1] = move_index_next(strobe_nums, 0, word_start_stop[0], '=')
        strobe_info['trial_name'] = "".join([chr(c) for c in strobe_nums[word_start_stop[0]:word_start_stop[1]]])
        word_start_stop[0] = word_start_stop[1] + 1
        word_start_stop[1] = move_index_next(strobe_nums, 0, word_start_stop[0], '=')
        strobe_info['trial_file'] = "".join([chr(c) for c in strobe_nums[word_start_stop[0]:word_start_stop[1]]])

        trial_strobe_info.append(strobe_info)

        # Find the index of the next saved trial, loop breaks if None is returned
        current_index = move_index_next(strobe_nums, saved_code, current_index, '=')

    return trial_strobe_info


""" Based on the strobe data read in from read_pl2_strobes, this function finds the
    DIO pulses from Maestro detected by Plexon and saves their times in an array
    matched with the channel they were detected on.  This assumes that Maestro outputs
    XS2 on Plexon event channel 02, and Maestro digital out pulse 01 is on Plexon
    channel 04, 02 is channel 05, and so on, i.e. maestro_pl2_chan_map = 3. """
def assign_trial_events(maestro_PL2_data, pl2_reader):
    maestro_pl2_chan_map = 3
    trial_strobe_info = read_pl2_strobes(pl2_reader)

    # Read in all the event channels that have data and concatenate their channel number and event times
    # into one big long numpy array.  Pad first column with -1.
    pl2_events_times = np.ones([2, 1]) * -1
    for event_key in pl2_reader.info['file_channel_indices']['event']:
        if event_key.startswith('EVT'):
            event_data = pl2_reader.load_channel(event_key)
            if event_data is not None:
                pl2_events_times = np.concatenate((pl2_events_times, np.stack([event_data['timestamps'] * 1000,
                                                                              float(event_key[3:]) * np.ones_like(event_data['timestamps'])], axis=0)), axis=1)

    # Sort the array of event channel numbers and event times by timestamps and set initial index to 0
    pl2_events_times = pl2_events_times[:, np.argsort(pl2_events_times[0])]
    pl2_event_index = 0
    remove_ind = []
    for trial in range(0, len(trial_strobe_info)):
        # Check if Plexon strobe and Maestro file names match
        if trial_strobe_info[trial]['trial_file'] in maestro_PL2_data[trial]['filename']:

            # First make numpy array of all event numbers and their times for this trial
            # and sort it according to time of events
            maestro_events_times = [[], []]
            for p in range(0, len(maestro_PL2_data[trial]['maestro_events'])):
                maestro_events_times[0].extend(maestro_PL2_data[trial]['maestro_events'][p])
                maestro_events_times[1].extend([p + 1] * len(maestro_PL2_data[trial]['maestro_events'][p]))
            maestro_events_times = np.array(maestro_events_times)
            maestro_events_times = maestro_events_times[:, np.argsort(maestro_events_times[0])]

            # Now build a matching event array with Plexon events.  Include 2 extra event
            # slots for the start and stop XS2 events, which are not in Maestro file.  This
            # array will be used for searching the Plexon data to find correct output
            pl2_trial_events = np.zeros([2, len(maestro_events_times[1]) + 2])

            # Strobed start precedes everything else so start here and find XS2 start and stop times
            pl2_event_index = move_index_next(pl2_events_times[0, :], trial_strobe_info[trial]['trial_start_time'], pl2_event_index, '>')
            pl2_event_index = move_index_next(pl2_events_times[1, :], 2, pl2_event_index - 1, '=')
            pl2_trial_events[:, 0] = pl2_events_times[:, pl2_event_index]
            pl2_trial_events[:, -1] = pl2_events_times[:, move_index_next(pl2_events_times[1, :], 2, pl2_event_index, '=')]

            # Save start and stop for output
            maestro_PL2_data[trial]['plexon_start_stop'] = (pl2_trial_events[0, 0], pl2_trial_events[0, -1])

            # Again, only include Plexon events that were observed in Maestro by looking
            # through all Maestro events and finding their counterparts in Plexon
            # according to the mapping defined in maestro_pl2_chan_map
            maestro_PL2_data[trial]['plexon_events'] = [[] for x in range(0, len(maestro_PL2_data[trial]['maestro_events']))]
            for event_ind, event_num in enumerate(maestro_events_times[1]):
                event_num = event_num.astype('int')
                pl2_event_index = move_index_next(pl2_events_times[1, :], event_num + maestro_pl2_chan_map, pl2_event_index, '=')
                pl2_trial_events[:, event_ind + 1] = pl2_events_times[:, pl2_event_index]

                # Compare Maestro and Plexon inter-event times
                if event_ind > 0:
                    aligment_difference = abs((maestro_events_times[0, event_ind] - maestro_events_times[0, event_ind-1]) -
                                              (pl2_trial_events[0, event_ind + 1] - pl2_trial_events[0, event_ind]))
                    if aligment_difference > 0.1:
                        remove_ind.append(trial)
                        break
                        # raise ValueError("Plexon and Maestro inter-event intervals do not match within 0.1 ms for trial {} and event number {}.".format(trial, event_num))

                # Re-stack plexon events for output by lists of channel number so they match Maestro data in maestro_PL2_data[trial]['events']
                if event_num > 0:
                    maestro_PL2_data[trial]['plexon_events'][event_num - 1].append(pl2_events_times[0, pl2_event_index])
                else:
                    # This is probably an error
                    print('FOUND A START STOP CODE IN TRIAL EVENTS!? PROBABLY AN ERROR')

        else:
            # This is an error that I am not sure what to do with
            print("Plexon filename {} and Maestro filename {} don't match!".format(trial_strobe_info[trial]['trial_file'], maestro_PL2_data[trial]['filename']))

    if len(maestro_PL2_data) != len(trial_strobe_info):
        # At this point it already went through all trial_strobe_info successfully, so assume any extra trials were dropped
        print("The {} extra trials on the end of maestro_PL2_data than Plexon strobes were removed by maestroPL2.assign_trial_events.".format(len(maestro_PL2_data) - len(trial_strobe_info)))
        del maestro_PL2_data[len(trial_strobe_info):]

    if len(remove_ind) > 0:
        remove_ind.reverse()
        for index in remove_ind:
            print("Trial {} did not have matching Plexon and Maestro events and was removed".format(index))
            del maestro_PL2_data[index]

    return maestro_PL2_data


def assign_trial_spikes(maestro_PL2_data, spike_times, spike_name=None):

    if 'plexon_events' not in maestro_PL2_data[0].keys():
        raise ValueError("Plexon events for maestro_PL2_data trial 0 not found! Must run assign_trial_events(maestro_PL2_data, pl2_reader) first.")

    if spike_name is None:
        for n_spike in range(0, 100):
            if n_spike not in maestro_PL2_data[0]['spikes'].keys():
                spike_name = n_spike
                break

    for trial in range(0, len(maestro_PL2_data)):
        # Since XS2 start pulse was screwey, use XS2 END pulse for beginning and end, find beginning by subtracting duration
        trial_index = np.all([spike_times >= (maestro_PL2_data[trial]['plexon_start_stop'][1] - maestro_PL2_data[trial]['duration_ms']),
                              spike_times <= maestro_PL2_data[trial]['plexon_start_stop'][1]], axis=0)

        maestro_PL2_data[trial]['spikes'][spike_name] = spike_times[trial_index]
        maestro_PL2_data[trial]['spike_alignment'][spike_name] = 0

    return maestro_PL2_data


def align_spikes_to_event(maestro_PL2_data, event_number, occurrence_n=1, trial_names=None, next_refresh=True, target_num=1):

    if event_number < 1:
        raise ValueError('Event number must be >= 1')
    # Subtract 1 from event number to match event indexing
    event_number -= 1
    occurrence_n -= 1

    for trial in range(0, len(maestro_PL2_data)):
        if event_number > len(maestro_PL2_data[trial]['maestro_events']):
            continue
        if occurrence_n > len(maestro_PL2_data[trial]['maestro_events'][event_number]) - 1:
            continue
        if trial_names is not None:
            if maestro_PL2_data[trial]['trial_name'] not in trial_names:
                continue

        if next_refresh:
            # Find the difference between next refresh time and Maestro event time add it to Plexon event time
            align_time = (maestro_PL2_data[trial]['plexon_events'][event_number][occurrence_n] +
                          (maestro_PL2_data[trial]['targets'][target_num].get_next_refresh(maestro_PL2_data[trial]['maestro_events'][event_number][occurrence_n]) -
                           maestro_PL2_data[trial]['maestro_events'][event_number][occurrence_n]))
        else:
            align_time = maestro_PL2_data[trial]['plexon_events'][event_number][occurrence_n]

        for spike_name in maestro_PL2_data[trial]['spikes'].keys():
            # Un-do old spike alignment, do new alignment and save it's time
            maestro_PL2_data[trial]['spikes'][spike_name] += maestro_PL2_data[trial]['spike_alignment'][spike_name]
            maestro_PL2_data[trial]['spikes'][spike_name] -= align_time
            maestro_PL2_data[trial]['spike_alignment'][spike_name] = align_time

    return maestro_PL2_data


def assign_retinal_velocity(maestro_PL2_data, target_num):

    for trial in maestro_PL2_data:
        calculated_velocity = trial['targets'][target_num].velocity_from_position(True)
        trial['retinal_velocity'] = calculated_velocity - trial['eye_velocity']


def align_timeseries_to_event(maestro_PL2_data, event_number, occurrence_n=1, trial_names=None, next_refresh=True, target_num=1):
    """ Timeseries are defined starting at time 0 for the beginning of each trial, this aligment subtracts the
        Plexon event time relative to trial time from each time series. """

    if event_number < 1:
        raise ValueError('Event number must be >= 1')
    # Subtract 1 from event number to match event indexing
    event_number -= 1
    occurrence_n -= 1

    for trial in range(0, len(maestro_PL2_data)):
        if event_number > len(maestro_PL2_data[trial]['maestro_events']):
            continue
        if occurrence_n > len(maestro_PL2_data[trial]['maestro_events'][event_number]) - 1:
            continue
        if trial_names is not None:
            if maestro_PL2_data[trial]['trial_name'] not in trial_names:
                continue

        if next_refresh:
            # Find the difference between next refresh time and Maestro event time add it to Plexon event time
            align_time = (maestro_PL2_data[trial]['plexon_events'][event_number][occurrence_n] +
                          (maestro_PL2_data[trial]['targets'][target_num].get_next_refresh(maestro_PL2_data[trial]['maestro_events'][event_number][occurrence_n]) -
                           maestro_PL2_data[trial]['maestro_events'][event_number][occurrence_n]))
        else:
            align_time = maestro_PL2_data[trial]['plexon_events'][event_number][occurrence_n]

        # Adjust align time to be relative to trial start (using plexon STOP CODE!) rather than Plexon file time
        align_time = maestro_PL2_data[trial]['duration_ms'] - (maestro_PL2_data[trial]['plexon_start_stop'][1] - align_time)

        # Un-do old time series alignment, do new alignment and save it's time
        maestro_PL2_data[trial]['time_series'] += maestro_PL2_data[trial]['time_series_alignment']
        maestro_PL2_data[trial]['time_series'] -= align_time
        maestro_PL2_data[trial]['time_series_alignment'] = align_time

    return maestro_PL2_data


def align_timeseries_to_maestro_event(maestro_PL2_data, event_number, occurrence_n=1, trial_names=None, next_refresh=True, target_num=1):
    """ Timeseries are defined starting at time 0 for the beginning of each trial, this aligment subtracts the
        Maestro event time relative to trial time from each time series. """

    if event_number < 0:
        raise ValueError('Event number must be >= 0')
    # Subtract 1 from occurence number to match event indexing
    occurrence_n -= 1

    for trial in range(0, len(maestro_PL2_data)):
        if event_number > len(maestro_PL2_data[trial]['maestro_events']):
            continue
        if occurrence_n > len(maestro_PL2_data[trial]['maestro_events'][event_number]) - 1:
            continue
        if trial_names is not None:
            if maestro_PL2_data[trial]['trial_name'] not in trial_names:
                continue

        if next_refresh:
            # Find the difference between next refresh time and Maestro event time
            maestro_PL2_data[trial]['maestro_events'][event_number][occurrence_n]
            align_time = maestro_PL2_data[trial]['targets'][target_num].get_next_refresh(np.floor(maestro_PL2_data[trial]['maestro_events'][event_number][occurrence_n]))
        else:
            align_time = maestro_PL2_data[trial]['maestro_events'][event_number][occurrence_n]

        # Un-do old time series alignment, do new alignment and save it's time
        maestro_PL2_data[trial]['time_series'] += maestro_PL2_data[trial]['time_series_alignment']
        maestro_PL2_data[trial]['time_series'] -= align_time
        maestro_PL2_data[trial]['time_series_alignment'] = align_time

    return maestro_PL2_data


def subtract_eye_offsets(maestro_PL2_data, align_segment=3, fixation_target=0, fixation_window=(-200, 0), epsilon_eye=0.1, max_iter=10,
                         time_cushion=20, acceleration_thresh=1, velocity_thresh=30):
    """ This subtracts the DC offsets in eye position and velocity by taking the mode of their values in the fixation window aligned on
        align_segment.  After this first adjustment, saccades are found in the fixation window and the mode eye position and velocity
        values are recomputed without saccades and subtracting again.  This is repeated recursively until the absolute value of the
        position and velocity mode during fixation window is less than epsilon_eye or max_iter recursive calls is reached.  The call to
        find_saccade_windows means that this variable will also be assigned to maestro_PL2_data.  The modes
        for both position and velocity are only computed for values where eye_velocity is less than velocity_thresh.  DC offsets in
        eye velocity greater than velocity_thresh will cause this to fail.
        """

    delta_eye = np.zeros(len(maestro_PL2_data))
    n_iters = 0
    next_trial_list = [x for x in range(0, len(maestro_PL2_data))]
    while n_iters < max_iter and len(next_trial_list) > 0:
        trial_indices = next_trial_list
        next_trial_list = []
        for trial in trial_indices:# len(maestro_PL2_data)):
            if not maestro_PL2_data[trial]['maestro_events'][align_segment]:
                # Current trial lacks desired align_segment
                continue

            align_time = maestro_PL2_data[trial]['targets'][fixation_target].get_next_refresh(maestro_PL2_data[trial]['maestro_events'][align_segment][0])
            time_index = np.arange(align_time + fixation_window[0], align_time + fixation_window[1] + 1, 1, dtype='int')

            if 'saccade_windows' in maestro_PL2_data[trial]:
                saccade_index = maestro_PL2_data[trial]['saccade_index'][time_index]
                time_index = time_index[~saccade_index]
                if np.all(saccade_index):
                    # Entire fixation window has been marked as a saccade
                    # TODO This is an error, or needs to be fixed, or skipped??
                    print("Entire fixation window marked as saccade for trial {} and was skipped".format(trial))
                    continue

            if (not np.any(np.abs(maestro_PL2_data[trial]['eye_velocity'][0][time_index]) < velocity_thresh) or
                not np.any(np.abs(maestro_PL2_data[trial]['eye_velocity'][1][time_index]) < velocity_thresh)):
                # Entire fixation window is over velocity threshold
                # TODO This is an error, or needs to be fixed, or skipped??
                print("Entire fixation window marked as saccade for trial {} and was skipped".format(trial))
                continue

            # Get modes of eye data in fixation window, excluding saccades and velocity times over threshold, then subtract them from eye data
            horizontal_vel_mode, n_horizontal_vel_mode = stats.mode(maestro_PL2_data[trial]['eye_velocity'][0][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][0][time_index]) < velocity_thresh])
            maestro_PL2_data[trial]['eye_velocity'][0] = maestro_PL2_data[trial]['eye_velocity'][0] - horizontal_vel_mode
            vertical_vel_mode, n_vertical_vel_mode = stats.mode(maestro_PL2_data[trial]['eye_velocity'][1][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][1][time_index]) < velocity_thresh])
            maestro_PL2_data[trial]['eye_velocity'][1] = maestro_PL2_data[trial]['eye_velocity'][1] - vertical_vel_mode


            # Position mode is also taken when VELOCITY is less than threshold and adjusted by the difference between eye data and target position
            h_position_offset = maestro_PL2_data[trial]['eye_position'][0][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][0][time_index]) < velocity_thresh] - np.nanmean(maestro_PL2_data[trial]['targets'][fixation_target].position[0, time_index])
            horizontal_pos_mode, n_horizontal_pos_mode = stats.mode(h_position_offset)
            maestro_PL2_data[trial]['eye_position'][0] = maestro_PL2_data[trial]['eye_position'][0] - horizontal_pos_mode
            v_position_offset = maestro_PL2_data[trial]['eye_position'][1][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][1][time_index]) < velocity_thresh] - np.nanmean(maestro_PL2_data[trial]['targets'][fixation_target].position[1, time_index])
            vertical_pos_mode, n_vertical_pos_mode = stats.mode(v_position_offset)
            maestro_PL2_data[trial]['eye_position'][1] = maestro_PL2_data[trial]['eye_position'][1] - vertical_pos_mode

            delta_eye[trial] = np.amax(np.abs((horizontal_vel_mode, vertical_vel_mode, horizontal_pos_mode, vertical_pos_mode)))
            if delta_eye[trial] >= epsilon_eye:
                # This trial's offsets aren't good enough so try again next loop
                next_trial_list.append(trial)

        maestro_PL2_data = find_saccade_windows(maestro_PL2_data, time_cushion=time_cushion, acceleration_thresh=acceleration_thresh, velocity_thresh=velocity_thresh)
        n_iters += 1

    return maestro_PL2_data


def find_saccade_windows(maestro_PL2_data, time_cushion=20, acceleration_thresh=1, velocity_thresh=30):
    # Force time_cushion to be integer so saccade windows are integers that can be used as indices
    time_cushion = np.array(time_cushion).astype('int')

    for trial in range(0, len(maestro_PL2_data)):
        # Compute normalized eye speed vector and corresponding acceleration
        # print(trial)
        # print(maestro_PL2_data[trial]['horizontal_eye_velocity'])
        eye_speed = la.norm(np.vstack((maestro_PL2_data[trial]['eye_velocity'][0], maestro_PL2_data[trial]['eye_velocity'][1])), ord=None, axis=0)
        eye_acceleration = np.zeros(eye_speed.shape)
        eye_acceleration[1:] = np.diff(eye_speed, n=1, axis=0)

        # Find saccades as all points exceeding input thresholds
        threshold_indices = np.where(np.logical_or((eye_speed > velocity_thresh), (np.absolute(eye_acceleration) > acceleration_thresh)))[0]
        if threshold_indices.size == 0:
            # No saccades this trial
            maestro_PL2_data[trial]['saccade_windows'] = np.empty(0).astype('int')
            maestro_PL2_data[trial]['saccade_index'] = np.zeros(maestro_PL2_data[trial]['duration_ms'], 'bool')
            maestro_PL2_data[trial]['saccade_time_cushion'] = time_cushion
            continue

        # Find the end of all saccades based on the time gap between each element of threshold indices.
        # Gaps between saccades must exceed the time_cushion padding on either end to count as separate saccades.
        switch_indices = np.zeros_like(threshold_indices, dtype=bool)
        switch_indices[0:-1] =  np.diff(threshold_indices) > time_cushion * 2

        # Use switch index to find where one saccade ends and the next begins and store the locations.  These don't include
        # the beginning of first saccade or end of last saccade, so add 1 element to saccade_windows
        switch_points = np.where(switch_indices)[0]
        saccade_windows = np.full((switch_points.shape[0] + 1, 2), np.nan, dtype='int')

        # Check switch points and use their corresponding indices to find actual time values in threshold_indices
        for saccade in range(0, saccade_windows.shape[0]):
            if saccade == 0:
                # Start of first saccade not indicated in switch points so add time cushion to the first saccade
                # time in threshold indices after checking whether it is within time limit of trial data
                if threshold_indices[saccade] - time_cushion > 0:
                    saccade_windows[saccade, 0] = threshold_indices[saccade] - time_cushion
                else:
                    saccade_windows[saccade, 0] = 0

            # Making this an "if" rather than "elif" means in case of only 1 saccade, this statement
            # and the "if" statement above both run
            if saccade == saccade_windows.shape[0] - 1:
                # End of last saccade, which is not indicated in switch points so add time cushion to end of
                # last saccade time in threshold indices after checking whether it is within time of trial data
                if len(eye_speed) < threshold_indices[-1] + time_cushion:
                    saccade_windows[saccade, 1] = len(eye_speed)
                else:
                    saccade_windows[saccade, 1] = threshold_indices[-1] + time_cushion
            else:
                # Add cushion to end of current saccade.
                saccade_windows[saccade, 1] = threshold_indices[switch_points[saccade]] + time_cushion

            # Add cushion to the start of next saccade if there is one
            if saccade_windows.shape[0] > saccade + 1:
                saccade_windows[saccade + 1, 0] = threshold_indices[switch_points[saccade] + 1] - time_cushion

        maestro_PL2_data[trial]['saccade_windows'] = saccade_windows
        maestro_PL2_data[trial]['saccade_time_cushion'] = time_cushion

        # Set boolean index for marking saccade times and save
        saccade_index = np.zeros(maestro_PL2_data[trial]['duration_ms'], 'bool')
        for saccade_win in maestro_PL2_data[trial]['saccade_windows']:
            saccade_index[saccade_win[0]:saccade_win[1]] = True
        maestro_PL2_data[trial]['saccade_index'] = saccade_index

    return maestro_PL2_data


def get_trial_gain_latency(maestro_PL2_data, data_win, gain_win, test_latencies):
    """ data_win is the entire eye velocity data considered, aligned on current time series.
        gain_win is the smaller window in which gain will be calculated at each time step
        relative to pursuit onset.  Subtracting fixation speed doesn't seem to change
        much or help with weird trials or saccades. """

    eye_speed = la.norm(neuron_tuning.eye_data_window(maestro_PL2_data, data_win)[:, :, 2:4], axis=2)
    eye_speed = neuron_tuning.nan_sac_data_window(maestro_PL2_data, data_win, eye_speed)
    template = np.nanmean(eye_speed, axis=1)
    acc_template = signal.savgol_filter(template, 31, 1, deriv=1, axis=0) * 1000
    template_latency = np.argmax(acc_template >= np.amax(acc_template) / 2)
    template = np.tile(template, eye_speed.shape[1]).reshape(template.shape[0], -1, order='F')
    if template_latency + gain_win[1] + np.amax(test_latencies) > data_win[1]:
        raise ValueError("Gain window and test lags exceed data window given calculated template latency of {}".format(template_latency + data_win[0]))
    if template_latency - gain_win[0] + np.amin(test_latencies) < data_win[0]:
        raise ValueError("Gain window and test lags precede data window given calculated template latency of {}".format(template_latency + data_win[0]))
    if template_latency < 25:
        raise ValueError("Template latency of {} is too close to start of data window to calculate fixation eye speed.  Decrease data window start time".format(template_latency))
    if np.isnan(template_latency):
        raise ValueError("Cannot find template pursuit latency")

    # Get mean fixaiton speed and subtract from everything
    # mean_fix_speed = np.nanmean(template[template_latency-25:template_latency, 0])
    # template = template - mean_fix_speed
    # eye_speed = eye_speed - np.nanmean(eye_speed[0:50, :], axis=0)

    sq_errors = np.ones(len(maestro_PL2_data)) * np.finfo('float64').max
    gains = np.full(len(maestro_PL2_data), np.nan)
    latencies = np.full(len(maestro_PL2_data), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lat in test_latencies:
            # Use nanmean to account for nan's since only the ratio of these actually matters
            fit_ratios = (np.nanmean(template[template_latency+gain_win[0]:template_latency+gain_win[1], :] *
                                    eye_speed[template_latency+gain_win[0]+lat:template_latency+gain_win[1]+lat, :], axis=0) /
                          np.nanmean(eye_speed[template_latency+gain_win[0]+lat:template_latency+gain_win[1]+lat, :] ** 2, axis=0))
            this_error = np.sum((template[template_latency+gain_win[0]:template_latency+gain_win[1], :] -
                                 fit_ratios * eye_speed[template_latency+gain_win[0]+lat:template_latency+gain_win[1]+lat, :]) ** 2,
                                 axis=0)
            # If current gain fit has less error than previous, replace it
            ind = this_error < sq_errors
            sq_errors[ind] = this_error[ind]
            gains[ind] = fit_ratios[ind]
            latencies[ind] = lat
    gains = 1 / gains
    latencies = latencies + template_latency + data_win[0]

    return gains, latencies, template_latency


def align_timeseries_to_latency(maestro_PL2_data, data_win, gain_win, test_latencies, trial_names=None):
    """ Alignment is done for behavioral pursuit latency RELATIVE to the current timeseries
        alignment.  Therefore the timeseries must be aligned first for accurate results!
        Trials without a viable latency (usually due to saccades) are aligned on the average
        template latency. """

    _, latencies, template_latency = get_trial_gain_latency(maestro_PL2_data, data_win, gain_win, test_latencies)
    for trial in range(0, len(maestro_PL2_data)):
        if trial_names is not None:
            if maestro_PL2_data[trial]['trial_name'] not in trial_names:
                continue

        if np.isnan(latencies[trial]):
            latencies[trial] = template_latency
        # Add new alignment and save it's time
        maestro_PL2_data[trial]['time_series'] -= latencies[trial]
        maestro_PL2_data[trial]['time_series_alignment'] += latencies[trial]

    return maestro_PL2_data


def inverse_ISI_FR(spike_times, time_window, time_step=1):
    """This will get ISI FR over all spike times that are input and trim the resulting train to time_window.
        Will compute ISI FR before the first spike in the time window if spike times input extend beyond the
        calculation window, so it is better to include spikes beyond the window than filter them out.
        Assumes that spike_times is a numpy array and will sort and flatten it.  All times are assumed to be
        in milliseconds.  The ISI firing rate is assigned to each time point such that at the time a spike
        occurs, this time point is immediately assigned the value of the subsequent ISI firing rate rather
        than that of the previous ISI.  The output includes the value at time_window[0] but NOT at
        time_window[1].  Failures and window times beyond all spike times return np.nan values.
        """

    # Error checking on input and return nan's of correct size if error
    n_time_points = np.array((time_window[1] - time_window[0]) / time_step).astype(int)
    if spike_times.size == 0:
        return np.full((n_time_points, ), np.nan)
    if len(spike_times.shape) > 1:
        spike_times.flatten()
    elif len(spike_times.shape) == 0:
        # TODO this is an error so I could raise an exception here?  It also assumes that spike_times are
        # a numpy array, which would also be an error if it is not.
        return np.full((n_time_points, ), np.nan)

    # Round window and spike times to nearest step increment rounded to 2 decimal places
    time_step = np.around(np.array(time_step), 2)
    time_window = np.around(np.array(time_window) * (1 / time_step)) / (1 / time_step)

    # Calculate ISI firing rates as difference of sorted spike times before rounding.  Pad the rates
    # array with nan in first and last elements to allow indexing rates from beyond spike times
    spike_times.sort(axis=0)
    rates = np.full(spike_times.shape[0] + 1, np.nan)
    rates[1:-1] = 1000 / np.diff(spike_times, n=1, axis=0)
    spike_times = np.around(spike_times * (1 / time_step)) / (1 / time_step)

    # Get the spike times in time_window and their indices
    win_spike_times = spike_times[np.logical_and((spike_times >= time_window[0]), (spike_times <= time_window[1]))]
    win_spike_index = np.where(np.logical_and((spike_times >= time_window[0]), (spike_times <= time_window[1])))[0]

    # Calculate time_step by time_step ISI FR output by repeating spike ISIs the correct number of times
    # for the duration of time_window.  Number of times is calculated as the difference of step numbers between each
    # spike.  The step numbers between window start and first spike and window stop and last spike are used by
    # inserting their times to round out the ISI_FR array.
    if len(win_spike_times) > 0:
        rate_index = np.insert(win_spike_index, (win_spike_index.shape[0]), [win_spike_index[-1] + 1])

        # Make times preceding first spike, with no defined rate, have the same rate as after first spike
        rates[0], rates[-1] = rates[1], rates[-2]
        ISI_FR = np.repeat(rates[rate_index], (np.diff(np.insert(win_spike_times, (0, len(win_spike_times)),
                                                       [time_window[0], time_window[1]]
                                                       )
                                               ) / time_step).astype(int))
    else:
        ISI_FR = np.zeros((n_time_points, ))

    return ISI_FR


def assign_ISI_FR(maestro_PL2_data, neuron=0, low_filt=25):

    if low_filt is not None:
        b_filt, a_filt = signal.butter(8, low_filt/500)

    for trial in range(0, len(maestro_PL2_data)):
        if 'ISI_FR' not in maestro_PL2_data[trial]:
            maestro_PL2_data[trial]['ISI_FR'] = {}
        # Un-do old spike alignment, align to trial start/stop pulses
        zero_spikes = maestro_PL2_data[trial]['spikes'][neuron] + maestro_PL2_data[trial]['spike_alignment'][neuron] - maestro_PL2_data[trial]['plexon_start_stop'][1] + maestro_PL2_data[trial]['duration_ms']
        maestro_PL2_data[trial]['ISI_FR'][neuron] = inverse_ISI_FR(zero_spikes, [0, maestro_PL2_data[trial]['duration_ms']], time_step=1)

        if low_filt is not None:
            maestro_PL2_data[trial]['ISI_FR'][neuron] = signal.filtfilt(b_filt, a_filt, maestro_PL2_data[trial]['ISI_FR'][neuron], axis=0, padlen=np.amin((maestro_PL2_data[trial]['duration_ms']/2, 100)).astype('int'))

    return maestro_PL2_data


def assign_bin_FR(maestro_PL2_data, neuron=0):

    for trial in range(0, len(maestro_PL2_data)):
        if 'bin_FR' not in maestro_PL2_data[trial]:
            maestro_PL2_data[trial]['bin_FR'] = {}
        # Undo any alignment and realign to trial time to get binned spikes over entire trial duration
        spikes = maestro_PL2_data[trial]['spikes'][neuron] + maestro_PL2_data[trial]['spike_alignment'][neuron] - (maestro_PL2_data[trial]['plexon_start_stop'][1] - maestro_PL2_data[trial]['duration_ms'])
        maestro_PL2_data[trial]['bin_FR'][neuron] = 1000 * np.histogram(spikes, bins=np.arange(-.5, maestro_PL2_data[trial]['duration_ms']+.5, 1))[0].astype('int')

    return maestro_PL2_data


def assign_PSP_decay_FR(maestro_PL2_data, neuron=0, tau_rise=1.0, tau_decay=2.5):
    """ Exponential decay kernal applied to bin_FR data. """
    if 'bin_FR' not in maestro_PL2_data[0]:
        maestro_PL2_data = assign_bin_FR(maestro_PL2_data, neuron)

    xvals = np.arange(0, np.ceil(5 * max(tau_rise, tau_decay)) + 1)
    kernel = np.exp(- 1 * xvals / tau_decay) - np.exp(- 1 * xvals / tau_rise)
    kernel = zero_phase_kernel(kernel, 0) # Shift kernel to be causal at t = 0
    kernel = kernel / np.sum(kernel)
    for trial in range(0, len(maestro_PL2_data)):
        if 'PSP_FR' not in maestro_PL2_data[trial]:
            maestro_PL2_data[trial]['PSP_FR'] = {}
        maestro_PL2_data[trial]['PSP_FR'][neuron] = np.convolve(maestro_PL2_data[trial]['bin_FR'][neuron], kernel, mode='same')

    return maestro_PL2_data


def find_trial_blocks(maestro_PL2_data, trial_names, ignore_trial_names=[''], block_min=0, block_max=math.inf,
                      max_absent=0, max_consec_absent=math.inf, max_absent_pct=1, max_consec_single=math.inf):
    """ Finds the indices of blocks of trials satisfying the input criteria.  The indices are given so that the
        number for BLOCK STOP IS ONE GREATER THAN THE ACTUAL TRIAL! so that it can be SLICED appropriately
        in Python. """

    block_trial_windows = []
    stop_triggers = []
    check_block = False
    final_check = False
    foo_block_start = None
    for trial in range(0, len(maestro_PL2_data)):
        if maestro_PL2_data[trial]['trial_name'] in trial_names and not check_block:
            # Block starts
            # print("STARTED block {}".format(trial))
            n_absent = 0
            n_consec_absent = 0
            n_consec_single = 0
            check_block = True
            foo_block_start = trial
        elif maestro_PL2_data[trial]['trial_name'] in trial_names and check_block:
            # Good trial in the current block being checked
            # print("CONTINUE block {}".format(trial))
            n_consec_absent = 0
            if n_consec_single > max_consec_single:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_consec_single exceeded')
            else:
                if maestro_PL2_data[trial]['trial_name'] == last_trial_name:
                    n_consec_single += 1
                else:
                    n_consec_single = 0
            foo_block_stop = trial
        elif maestro_PL2_data[trial]['trial_name'] not in trial_names and maestro_PL2_data[trial]['trial_name'] not in ignore_trial_names and check_block:
            # Bad trial for block
            # print("FAILED block {}".format(trial))
            n_absent += 1
            if n_absent > max_absent:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_absent exceeded')
            elif n_consec_absent > max_consec_absent:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_consec_absent exceeded')
            # else:
            #     stop_triggers.append('good block')
            n_consec_absent += 1
        else:
            pass
        last_trial_name = maestro_PL2_data[trial]['trial_name']

        if trial == len(maestro_PL2_data)-1:
            final_check = True
            this_trigger = (trial, 'end of file reached')

        if final_check:
            # Check block qualities that can't be done without knowing entire block
            # and then save block for output or discard
            # print("FINAL CHECK {}".format(trial))
            if foo_block_start is None:
                block_len = 0
            else:
                block_len = foo_block_stop - foo_block_start
            if block_len > 0 and check_block:
                # Subtract 1 from n_absent here because 1 was added to it just to break the block and is not included in it
                if block_len >= block_min and block_len <= block_max and ((n_absent - 1)/ block_len) <= max_absent_pct:
                    block_trial_windows.append([foo_block_start, foo_block_stop + 1])
                    stop_triggers.append(this_trigger)
            check_block = False
            final_check = False

    return block_trial_windows, stop_triggers


def trim_trial_blocks(remove_block, keep_block):

    new_keep_block = [trial for trial in keep_block if trial not in remove_block]
    return new_keep_block



# TODO Maestro trials need have their velocity and postion offsets subtracted from every
# trial and then need a saccade detection method - this requires a fixation window or
# alignment pulse.  To get more of this done in one trial loop it could be implemented
# at the read point or at least at the point of maestro object


# class MaestroPL2(object):
# IT WOULD BE REALLY NICE IF THE DATA COULD BE SLICED BY TRIALS BUT ALSO HAD
# INDICES FOR EACH TRIAL NAME IN A DICTIONARY THAT WOULD ALLOW IT TO INDEX THOSE
# otherwise I will just have to loop through everything every time

# Any maestro or data object would
# possibly benefit from having a length property for the object AND EACH TRIAL
# because I am looking these up all the time - trial found here maestro_PL2_data[trial]['header']['_num_saved_scans']
# Make event pulses aligned in time with a lookup table of channel number as is done in assign_trial_events

# A data object should also be setup to call a function on only a subset of trials, but still allowing
# calculations to be returned only to those trials and preserve the entire object.  Currently all functions loop
# with trial = len(maestro_PL2_data), and insert output data in maestro_PL2_data[trial].  Because maestro_PL2_data would be
# implicit input in a method call, I think they would all need a trial_index input key that would then be looped over,
# that could default to something like range(0, self.n_trials)

# Seems like basially everything would be easier if data were simply aligned to some event (or not) and then things
# were calculated on the data, as it sits in alignment.  Otherwise I am basically realigning at every step for each
# calculation.  This alignment would need to be done in such a way that the original data could be recovered so that
# the data could be re-aligned to a new time point !!!

# Pay attention to inputs like time_window, which do not come in as numpy arrays but are often used to do arithmetic
# against numpy arrays.  This can result in returning values that are no longer numpy arrays so I may want to
# control this behavior by casting things to numpy arrays to guarantee numpy array output in some cases

# I really need a way to handle whether or not saccade windows should be nan'ed.  As it stands basically all these
# functions just remove/nan saccade data no matter what.  For instance, even the tuning plot function and the
# firing rate function have different ways of removing data during saccades

# Things need to be clearly specified and internally consistent as to whether the points in things like
# saccade_windows are meant to be used as indices (zero indexing, dropping last value) or whether they are meant
# to be used as time points that must be removed inclusively

# Also need some way of outputing time for plotting things, really this should be combined with some general alignment
# method that also satisfies the other stuff I wnat from above

# ALSO THE EVENTS ARE ALL LISTS OF LISTS UNNECESSARILY
#

#     def __init__(self, maestro_PL2_data, pl2_data):
#
#
# class MaestroData(object):
#
#     def __init__(self, data):
#
#         self.maestro_events = data['maestro_events']
