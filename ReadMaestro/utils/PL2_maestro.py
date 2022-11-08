import operator
import numpy as np



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
    # This is for 32 bit adjustment but should be read in 16 bit now but still check
    if np.all(PL2_strobe_data['strobed'] >= 32768):
        strobe_nums = PL2_strobe_data['strobed'] - 32768
    else:
        strobe_nums = PL2_strobe_data['strobed']

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
    matched with the channel they were detected on.  e.g. Maestro outputs
    XS2 on Plexon event channel 02, and Maestro digital out pulse 01 is on Plexon
    channel 04, 02 is channel 05, and so on, i.e. maestro_pl2_chan_offset = 3.
    The maestro_data input is modified IN PLACE to contain the fields
    "plexon_start_stop" and "plexon_events".

    maestro_data is a list of maestro trial dictionaries as output by
        ReadMaestro.maestro_read.load_directory.
    pl2_reader
    """
def add_plexon_events(maestro_data, pl2_reader, maestro_pl2_chan_offset=3,
                        xs2_evt=2, max_pl2_event_num=9):

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
    pl2_events_times = pl2_events_times[:, np.argsort(pl2_events_times[0, :])]
    pl2_event_index = 0
    remove_ind = []
    for trial in range(0, len(trial_strobe_info)):
        # Check if Plexon strobe and Maestro file names match
        if trial_strobe_info[trial]['trial_file'] in maestro_data[trial]['filename']:

            # First make numpy array of all event numbers and their times for this trial
            # and sort it according to time of events
            maestro_events_times = [[], []]
            for p in range(0, len(maestro_data[trial]['events'])):
                maestro_events_times[0].extend(maestro_data[trial]['events'][p])
                maestro_events_times[1].extend([p + 1] * len(maestro_data[trial]['events'][p]))
            maestro_events_times = np.array(maestro_events_times)
            maestro_events_times = maestro_events_times[:, np.argsort(maestro_events_times[0, :])]

            # Now build a matching event array with Plexon events.  Include 2 extra event
            # slots for the start and stop XS2 events, which are not in Maestro file.  This
            # array will be used for searching the Plexon data to find correct output
            pl2_trial_events = np.zeros([2, len(maestro_events_times[1]) + 2])

            # Strobed start precedes everything else so start here and find XS2 start and stop times
            pl2_event_index = move_index_next(pl2_events_times[0, :], trial_strobe_info[trial]['trial_start_time'], pl2_event_index, '>')
            pl2_event_index = move_index_next(pl2_events_times[1, :], xs2_evt, pl2_event_index - 1, '=')
            pl2_trial_events[:, 0] = pl2_events_times[:, pl2_event_index]
            pl2_trial_events[:, -1] = pl2_events_times[:, move_index_next(pl2_events_times[1, :], xs2_evt, pl2_event_index, '=')]

            # Save start and stop for output
            maestro_data[trial]['plexon_start_stop'] = (pl2_trial_events[0, 0], pl2_trial_events[0, -1])

            # Check trial duration according to plexon XS2 and Maestro file
            pl2_duration = maestro_data[trial]['plexon_start_stop'][1] - maestro_data[trial]['plexon_start_stop'][0]
            maestro_duration = maestro_data[trial]['header']['_num_saved_scans']
            if np.abs(pl2_duration - maestro_duration) > 2.0:
                print("WARNING: difference between recorded pl2 trial duration {0} and maestro trial duration {1} is over 2 ms. This could mean XS2 inital pulse was delayed and unreliable.".format(pl2_duration, maestro_duration))

            # Again, only include Plexon events that were observed in Maestro by looking
            # through all Maestro events and finding their counterparts in Plexon
            # according to the mapping defined in maestro_pl2_chan_offset
            maestro_data[trial]['plexon_events'] = [[] for x in range(0, len(maestro_data[trial]['events']))]
            for event_ind, event_num in enumerate(maestro_events_times[1, :]):
                event_num = np.int64(event_num)
                if (event_num + maestro_pl2_chan_offset) > max_pl2_event_num:
                    continue
                new_pl2_event_index = move_index_next(pl2_events_times[1, :], event_num + maestro_pl2_chan_offset, pl2_event_index, '=')
                # Checkt that we found a usable next index for matching plexon event
                if new_pl2_event_index is None:
                    print("Cannot find matching event {0} for trial {1}, the event doesn't exist in remaining plexon events!".format(event_num + maestro_pl2_chan_offset, trial))
                    continue
                if pl2_events_times[0, new_pl2_event_index] > maestro_data[trial]['plexon_start_stop'][1]:
                    # Next matching event is BEYOND THE CURRENT TRIAL!
                    print("Cannot find matching event {0} for trial {1}, next match is beyond current trial!".format(event_num + maestro_pl2_chan_offset, trial))
                    continue
                pl2_event_index = new_pl2_event_index
                pl2_trial_events[:, event_ind + 1] = pl2_events_times[:, pl2_event_index]

                # Compare Maestro and Plexon inter-event times, Need to convert Maestro times to ms
                if event_ind == 0:
                    # Check first event against the XS2 trial start event
                    aligment_difference = abs(1000 * (maestro_events_times[0, event_ind]) -
                                              (pl2_trial_events[0, event_ind + 1] - pl2_trial_events[0, 0]))
                else:
                    aligment_difference = abs(1000 * (maestro_events_times[0, event_ind] - maestro_events_times[0, event_ind-1]) -
                                              (pl2_trial_events[0, event_ind + 1] - pl2_trial_events[0, event_ind]))
                if aligment_difference > 0.1:
                    remove_ind.append(trial)
                    print("Plexon and Maestro inter-event intervals do not match within 0.1 ms for trial {0} and event number {1} and will be removed.".format(trial, event_num))
                    break
                    # raise ValueError("Plexon and Maestro inter-event intervals do not match within 0.1 ms for trial {0} and event number {1}.".format(trial, event_num))

                # Re-stack plexon events for output by lists of channel number so they match Maestro data in maestro_data[trial]['events']
                if event_num > 0:
                    maestro_data[trial]['plexon_events'][event_num - 1].append(pl2_events_times[0, pl2_event_index])
                else:
                    # This is probably an error
                    print('FOUND A START STOP CODE IN TRIAL EVENTS!? PROBABLY AN ERROR')

        else:
            # This is an error that I am not sure what to do with, unless file names have been changed for some reason
            print("Plexon filename {} and Maestro filename {} don't match!".format(trial_strobe_info[trial]['trial_file'], maestro_data[trial]['filename']))

    if len(maestro_data) != len(trial_strobe_info):
        # At this point it already went through all trial_strobe_info successfully, so assume any extra trials were dropped
        print("The {} extra trials on the end of maestro_data than Plexon strobes were removed by maestroPL2.assign_trial_events.".format(len(maestro_data) - len(trial_strobe_info)))
        del maestro_data[len(trial_strobe_info):]

    if len(remove_ind) > 0:
        remove_ind.reverse()
        for index in remove_ind:
            print("Trial {} did not have matching Plexon and Maestro events and was removed".format(index))
            del maestro_data[index]

    return maestro_data
