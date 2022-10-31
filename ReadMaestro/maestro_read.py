"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>

Added _process_codes_compute_target_motion, a modification of the previous
_process_trial_codes.  This now includes adjustments only at monitor refresh
increments and the ability to handle velocity stabilization.

Above is implemented as a MaestroTargetCalculator class definition that tries to handle
everything automatically

Added keyword 'fail_safe_time' to the data header, which contains the minimum
time reached for data to have been saved by Maestro.
"""

import struct
import io
import os
import sys
import numpy as np
import re
import math
import pickle


def load_directory(directory_name, check_existing=True, save_data=False, save_name=None):
    """Load a directory of maestro files

    Loads a complete directory of maestro files as a list of dictionaries.
    The filenames are assumed to be in the of the form *.[0-9][0-9][0-9][0-9].
    This function attempts to load the files in order of their suffix
    """
    if check_existing:
        try:
            with open(directory_name + ".pickle", 'rb') as fp:
                data = pickle.load(fp)
            return data
        except FileNotFoundError:
            pass
        try:
            with open(directory_name + "_maestro.pickle", 'rb') as fp:
                data = pickle.load(fp)
            return data
        except FileNotFoundError:
            pass
        try:
            if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
                save_name = save_name + ".pickle"
            with open(save_name, 'rb') as fp:
                data = pickle.load(fp)
            return data
        except FileNotFoundError:
            pass
        print("Could not find existing Maestro file. Recomputing from scratch.")

    if not os.path.isdir(directory_name):
        raise RuntimeError('Directory name {:s} is not valid'.format(directory_name))
    pattern = re.compile('\.[0-9]+$')
    filenames = [f for f in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, f)) and pattern.search(f) is not None]

    # Sort by file name
    filenames.sort()
    data = []
    for filename in filenames:
        try:
            data.append(load(os.path.join(directory_name, filename)))
        except:
            print("Encountered error reading file", filename, "and trial was skipped.")
            continue

    if save_name is not None:
        save_data = True
        if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
            save_name = save_name + ".pickle"
    if save_data:
        if save_name is None:
            save_name = directory_name.split("/")[-1]
            root_name = directory_name.split("/")[0:-1]
            save_name = save_name + "_maestro.pickle"
            save_name = "".join(x + "/" for x in root_name) + save_name
        print("Saving Maestro trial data as:", save_name)
        with open(save_name, 'wb') as fp:
            pickle.dump(data, fp, protocol=-1)

    return data


def load(filename):
    """Load a maestro file"""
    if not os.path.isfile(filename):
        raise RuntimeError('File {:s} does not exist'.format(filename))
    fp = open(filename, 'rb')
    buffer = fp.read()
    fp.close()

    # Open the file for reading bytes and determine basic info
    with io.BytesIO(buffer) as fp:
        # Determine the size of the file
        fp.seek(0, os.SEEK_END)
        num_total_bytes = fp.tell()
        fp.seek(0, os.SEEK_SET)

        data = {}
        data['filename'] = filename
        data['header'] = _read_header(fp)
        data['header']['UsedStab'] = False # Default False. Set true later if stabilization is used

        # Read records until the end of the file
        while fp.tell() < num_total_bytes:
            data = _parse_record(fp, data)

    # Decompress ai data
    if 'ai_compressed_data' in data:
        data['ai_data'] = _decompress_ai(data['ai_compressed_data'], data['header']['num_channels'])
        # Convert the AI channels into standard units
    if 'compressed_fast_data' in data:
        data['fast_data'] = _decompress_ai(data['compressed_fast_data'], 1)

    if 'ai_data' in data:
        data['horizontal_eye_position'] = np.array(data['ai_data'][0]) * 0.025 # Scaling factor on website
        data['vertical_eye_position'] = np.array(data['ai_data'][1]) * 0.025
        data['horizontal_eye_velocity'] = np.array(data['ai_data'][2]) * 0.09189
        data['vertical_eye_velocity'] = np.array(data['ai_data'][3]) * 0.09189

    # Convert the channels to appropriate velocities and degress
    if 'trial_codes' in data:
        # Reparse the trial codes into targets
        data['horizontal_target_position'] =  np.zeros((len(data['targets']), data['header']['num_scans']))
        data['vertical_target_position'] = np.zeros((len(data['targets']), data['header']['num_scans']))
        data['horizontal_target_velocity'] =  np.zeros((len(data['targets']), data['header']['num_scans']))
        data['vertical_target_velocity'] = np.zeros((len(data['targets']), data['header']['num_scans']))
        # data = _process_trial_codes(data)
        data = _process_codes_compute_target_motion(data)

    return data


# CONSTANTS
MAX_NAME_SIZE = 40
MAX_ANALOG_INPUT_CHANNELS = 16
CURRENT_VERSION = 20 # Maestro 3.2.0
HEADER_SIZE = 1024
RECORD_SIZE = 1024

# Header flag contants
FLAG_IS_CONTINUOUS = (1<<0)
FLAG_SAVED_SPIKES = (1<<1)
FLAG_REWARD_EARNED = (1<<2)
FLAG_REWARD_GIVEN = (1<<3)
FLAG_FIXATION_1_SELECTED = (1<<4)
FLAG_FIXATION_2_SELECTED = (1<<5)
FLAG_END_SELECT = (1<<6)
FLAG_HAS_TAGGED_SECTIONS = (1<<7)
FLAG_IS_RP_DISTRO = (1<<8)
FLAG_IS_GOT_RP_DISTRO = (1<<9)
FLAG_IS_SEARCH_TASK = (1<<10)
FLAG_IS_ST_OK = (1<<11)
FLAG_IS_DISTRACTED = (1<<12)
FLAG_EYELINK_USED = (1<<13)


def _read_header(fp):
    """Read a maestro header file"""
    header = {}
    read_bytes = fp.read(HEADER_SIZE) # Always 1K header
    current_byte = 0 # Counter to the current byte

    def struct_read(fmt, num_bytes):
        nonlocal current_byte # We want to update the value of current byte
        data =  struct.unpack('<' + fmt, read_bytes[current_byte:(current_byte+num_bytes)])
        current_byte += num_bytes
        return data

    header['name'] = struct_read('{:d}s'.format(MAX_NAME_SIZE), MAX_NAME_SIZE)[0].decode('ascii').split('\0', 1)[0]
    header['_horizontal_direction'] = struct_read('h', 2)[0]
    header['_vertical_direction'] = struct_read('h', 2)[0]
    header['_num_compressed_bytes'] = struct_read('h', 2)[0]
    header['_num_saved_scans'] = struct_read('h', 2)[0]
    header['num_channels'] = struct_read('h', 2)[0]
    header['channel_list'] = struct_read('{:d}h'.format(MAX_ANALOG_INPUT_CHANNELS), 2 * MAX_ANALOG_INPUT_CHANNELS)
    header['display_width_pixels'] = struct_read('h', 2)[0]
    header['display_height_pixels'] = struct_read('h', 2)[0]
    header['_d_crow'] = struct_read('h', 2)[0]
    header['_d_ccol'] = struct_read('h', 2)[0]
    header['display_distance'] = float(struct_read('h', 2)[0]) / 1000.0 # Convert to m
    header['display_width'] = float(struct_read('h', 2)[0]) / 1000.0
    header['display_height'] = float(struct_read('h', 2)[0]) / 1000.0
    header['display_framerate'] = float(struct_read('i', 4)[0]) * 0.001 # Convert to Hz as measured via RMVideo
    header['_i_position_scale'] = float(struct_read('i', 4)[0]) / 1000.0
    header['_i_position_theta'] = float(struct_read('i', 4)[0]) / 1000.0
    header['_i_velocity_scale'] = float(struct_read('i', 4)[0]) / 1000.0
    header['_i_velocity_theta'] = float(struct_read('i', 4)[0]) / 1000.0
    header['reward_pulse_length_1'] = float(struct_read('i', 4)[0]) / 1000.0 # Conver to s
    header['reward_pulse_length_2'] = float(struct_read('i', 4)[0]) / 1000.0 # Conver to s
    header['day'] = struct_read('i', 4)[0] - 1 # Use base 0 indexing for month/day
    header['month'] = struct_read('i', 4)[0] - 1
    header['year'] = struct_read('i', 4)[0]
    header['version'] = struct_read('i', 4)[0]
    header['flags'] = struct_read('I', 4)[0]
    header['scan_interval'] = float(struct_read('i', 4)[0]) * 1E-6 # Convert uS to s
    header['num_compressed_channel_bytes'] = struct_read('i', 4)[0]
    header['num_scans'] = struct_read('i', 4)[0]
    header['_spike_filename'] = struct_read('{:d}s'.format(MAX_NAME_SIZE), MAX_NAME_SIZE)[0].decode('ascii').split('\0', 1)[0]
    header['num_compressed_spike_bytes'] = struct_read('i', 4)[0]
    header['spike_sampling_interval'] = float(struct_read('i', 4)[0]) * 1E-6 # Convert uS to s
    header['xy_random_seed'] = struct_read('I', 4)[0]
    header['i_rp_distro_start'] = float(struct_read('i', 4)[0]) / 1000.0
    header['i_rp_distro_duration'] = float(struct_read('i', 4)[0]) / 1000.0
    header['i_rp_distro_response'] = float(struct_read('i', 4)[0]) / 1000.0
    header['i_rp_windows'] = struct_read('4i', 4 * 4)[0]
    header['i_rp_response_type'] = struct_read('i', 4)[0]
    header['i_horizontal_start_position'] = float(struct_read('i', 4)[0]) / 1000.0
    header['i_vertical_start_position'] = float(struct_read('i', 4)[0]) / 1000.0
    header['trial_flags'] = struct_read('I', 4)[0]
    header['search_task_target_selected'] = struct_read('i', 4)[0]
    header['vstab_sliding_window'] = float(struct_read('i', 4)[0]) / 1000.0 # to s
    header['eyelink_flags'] = struct_read('9i', 4 * 9)

    if header['version'] >= 21: # Maestro 4.0
        header['set_name'] = struct_read('{:d}s'.format(MAX_NAME_SIZE), MAX_NAME_SIZE)[0].decode('ascii').split('\0', 1)[0]
        header['sub_set_name'] = struct_read('{:d}s'.format(MAX_NAME_SIZE), MAX_NAME_SIZE)[0].decode('ascii').split('\0', 1)[0]

        header['rmv_sync_size'] = struct_read('h', 2)[0]
        header['rmv_sync_duration_frames'] = struct_read('h', 2)[0]
        header['timestamp'] = float(struct_read('i', 4)[0]) / 1000.0 # Convert to ms

    return header


# Record type constants
RECORD_ID_LENGTH = 8
RECORD_TYPE_AI = 0
RECORD_TYPE_EVENT_0 = 1
RECORD_TYPE_EVENT_1 = 2
RECORD_TYPE_EVENT_OTHER = 3
RECORD_TYPE_TRIAL_CODE = 4
RECORD_TYPE_XWORK_ACTION = 5
RECORD_TYPE_SPIKE_SORTED_MIN = 8
RECORD_TYPE_SPIKE_SORTED_MAX = 57
RECORD_TYPE_v1_TARGET = 64
RECORD_TYPE_TARGET = 65
RECORD_TYPE_STIMULUS_RUN = 66
RECORD_TYPE_SPIKE_TRACE = 67
RECORD_TYPE_TAG_SECTION = 68
END_OF_RECORD = 0x7fffffff
EYELINK_BLINK_START = (1<<16)
EYELINK_BLINK_END = (1 << 17)


def _parse_record(fp, data):

    # Read the record ID and find the record type using the first byte.  Reset fp.seek position.
    read_bytes = fp.read(RECORD_ID_LENGTH)
    record_type = read_bytes[0]
    fp.seek(fp.tell()-RECORD_ID_LENGTH, os.SEEK_SET)

    if record_type == RECORD_TYPE_AI or record_type == RECORD_TYPE_SPIKE_TRACE:
        return _read_ai(fp, data)
    elif record_type == RECORD_TYPE_EVENT_0 or record_type == RECORD_TYPE_EVENT_1:
        return _read_event(fp, data)
    elif record_type == RECORD_TYPE_EVENT_OTHER:
        return _read_other_event(fp, data)
    elif record_type == RECORD_TYPE_TRIAL_CODE:
        return _read_trial_code(fp, data)
    elif record_type == RECORD_TYPE_XWORK_ACTION:
        return _read_xwork_action(fp, data)
    elif record_type == RECORD_TYPE_TARGET:
        return _read_target(fp, data)
    elif record_type == RECORD_TYPE_STIMULUS_RUN:
        return _read_stimulus_run(fp, data)
    elif record_type == RECORD_TYPE_TAG_SECTION:
        return _read_tag_section(fp, data)
    elif record >= RECORD_TYPE_SPIKE_SORTED_MIN and record <= RECORD_TYPE_SPIKE_SORTED_MAX:
        return _read_spike_sorted(fp, data)
    else:
        raise RuntimeError('Invalid record with type {:d}'.format(record_type))


def _read_ai(fp, data):
    """Reads compressed AI data or spike wave record data"""
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    if (header_bytes[0] == RECORD_TYPE_AI):
        if not 'ai_compressed_data' in data:
            data['ai_compressed_data'] = read_data
        else:
            data['ai_compressed_data'] += read_data
    else: # This must be a spike trace
        if not 'compressed_fast_data' in data:
            data['compressed_fast_data'] = read_data
        else:
            data['compressed_fast_data'] += read_data

    return data


def _decompress_ai(channel_data, num_channels, ):
    """Decompress an AI data channel - input is bytes"""
    current_byte = 0
    decompressed_data = [[] for i in range(0, num_channels)]
    last_sample = [0 for i in range(0, num_channels)]
    while current_byte < len(channel_data):
        for channel in range(num_channels):
            value = channel_data[current_byte]
            if current_byte == len(channel_data) or value == 0 or value == -1:
                # Hit the end...return the compressed data
                return decompressed_data
            if value & 0x080:
                # Bit 7 is set - next dataum is 2 bytes
                temp = (((value & 0x7F) << 8) | (0x00FF & (channel_data[current_byte + 1]))) - 4096
                current_byte += 1 # Used next byte
                last_sample[channel] += temp # Datum is different from last sample
            else:
                # Bit 7 is clear - next data is 1 byte
                last_sample[channel] += (value - 64) # Dataum is difference from last sample
            decompressed_data[channel].append(last_sample[channel])
            current_byte += 1

    return decompressed_data


def _read_event(fp, data):
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    num_ints = int((RECORD_SIZE - RECORD_ID_LENGTH) / 4)
    data_ints = struct.unpack('<{:d}i'.format(num_ints), read_data[:num_ints * 4])
    for current_int in data_ints:
        if current_int == END_OF_RECORD:
            break
        if header_bytes[0] == RECORD_TYPE_EVENT_0:
            field = 'event_0'
        else:
            field = 'event_1'
        if field not in data:
            data[field] = [float(current_int) / 1E5] # Convert to S (stored in 10us ticks)
        else:
            data[field].append(data[field][-1] + (float(current_int) / 1E5))

    return data


def _read_other_event(fp, data):
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    num_ints = int((RECORD_SIZE - RECORD_ID_LENGTH) / 4)
    data_ints = struct.unpack('<{:d}i'.format(num_ints), read_data[:num_ints * 4])
    if not 'events' in data:
        data['events'] = [[] for i in range(0, 16 - 2)]

    # TODO - Eyelink events
    for i in range(0, len(data_ints), 2):
        current_mask = data_ints[i]
        current_time = data_ints[i+1]
        if current_time == END_OF_RECORD or current_mask == END_OF_RECORD:
            break
        for channel in range(2, 16):
            if (current_mask & (1 << channel)):
                data['events'][channel-2].append(float(current_time) / 1E5)

    return data


# TRIAL CODE_CONSTANTS
TRIAL_CODE_TARGET_ON = 1
TRIAL_CODE_TARGET_OFF = 2
TRIAL_CODE_TARGET_HORIZONTAL_VELOCITY_CHANGE = 3
TRIAL_CODE_TARGET_VERTICAL_VELOCITY_CHANGE = 4
TRIAL_CODE_TARGET_HORIZONTAL_POSITION_RELATIVE = 5
TRIAL_CODE_TARGET_VERTICAL_POSITION_RELATIVE = 6
TRIAL_CODE_TARGET_HORIZONTAL_POSITION_ABSOLUTE = 7
TRIAL_CODE_TARGET_VERTICAL_POSITION_ABSOLUTE = 8
TRIAL_CODE_ADC_ON = 10
TRIAL_CODE_ADC_OFF = 11
TRIAL_CODE_FIXATION_1 = 12
TRIAL_CODE_FIXATION_2 = 13
TRIAL_CODE_FIXATION_ACCURACY = 14
TRIAL_CODE_PULSE_ON = 16
TRIAL_CODE_VSYNC_PULSE = 32
TRIAL_CODE_TARGET_HORIZONTAL_ACCELERATION = 18
TRIAL_CODE_TARGET_VERTICAL_ACCELERATION = 19
TRIAL_CODE_TARGET_PERTURBATION = 20
TRIAL_CODE_TARGET_HOPEN = 21
TRIAL_CODE_TARGET_HORIZONTAL_VELOCITY_LO = 27
TRIAL_CODE_TARGET_VERTICAL_VELOCITY_LO = 28
TRIAL_CODE_TARGET_HORIZONTAL_ACCELERATION_LO = 29
TRIAL_CODE_TARGET_VERTICAL_ACCELERATION_LO = 30
TRIAL_CODE_DELTA_TIME = 36
TRIAL_CODE_XYTARGET_USED = 38
TRIAL_CODE_PATTERN_HORIZONTAL_VELOCITY = 39
TRIAL_CODE_PATTERN_VERTICAL_VELOCITY = 40
TRIAL_CODE_PATTERN_HORIZONTAL_VELOCITY_LO = 41
TRIAL_CODE_PATTERN_VERTICAL_VELOCITY_LO = 42
TRIAL_CODE_PATTERN_HORIZONTAL_ACCELERATION = 45
TRIAL_CODE_PATTERN_VERTICAL_ACCELERATION = 46
TRIAL_CODE_PATTERN_HORIZONTAL_ACCLERATION_LO = 47
TRIAL_CODE_PATTERN_VERTICAL_ACCELERATION_LO = 48
TRIAL_CODE_SPECIAL_OP = 60
TRIAL_CODE_REWARD_PULSE_LENGTH = 61
TRIAL_CODE_PULSE_SEQUENCE = 62
TRIAL_CODE_CHECK_RESPONSE_ON = 63
TRIAL_CODE_CHECK_RESPONSE_OFF = 64
TRIAL_CODE_FAIL_SAFE = 65
TRIAL_CODE_MIDTRIAL_REWARD = 66
TRIAL_CODE_RPD_WINDOW = 67
TRIAL_CODE_TARGET_VELOCITY_STABILIZATION = 68
TRIAL_CODE_RANDOM_SEED = 97
TRIAL_CODE_START_TRIAL = 98
TRIAL_CODE_END_TRIAL = 99


def _read_trial_code(fp, data):
    """Stored as 2 shorts"""
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    num_shorts = int((RECORD_SIZE - RECORD_ID_LENGTH) / 2)
    data_shorts = struct.unpack('<{:d}h'.format(num_shorts), read_data[:num_shorts * 2])

    if not 'trial_codes' in data:
        data['trial_codes'] = [[] for i in range(0, 2)]

    for i in range(0, len(data_shorts), 2):
        current_code = data_shorts[i]
        current_time = data_shorts[i+1]
        data['trial_codes'][0].append(current_time)
        data['trial_codes'][1].append(current_code)

    return data


def _read_xwork_action(fp, data):
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    print('XWORK_ACTION NOT IMPLEMENTED')
    return data


TARGET_MAX_NAME_LENGTH = 50
XY_TARGET = 0x001C
RMV_TARGET = 0x01D


def _read_target(fp, data):
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    current_byte = 0 # Counter to the current byte

    if not 'targets' in data:
        data['targets'] = []

    def struct_read(fmt, num_bytes):
        nonlocal current_byte # We want to update the value of current byte
        data =  struct.unpack('<' + fmt, read_data[current_byte:(current_byte+num_bytes)])
        current_byte += num_bytes
        return data

    if data['header']['version'] < 2:
        raise RuntimeError('Invalid file version')
    elif data['header']['version'] >= 13:
        while current_byte < RECORD_SIZE - RECORD_ID_LENGTH:
            target = {}
            w_target_type = struct_read('H', 2)[0]
            target['target_name'] = struct_read('{:d}s'.format(TARGET_MAX_NAME_LENGTH), TARGET_MAX_NAME_LENGTH)[0].decode('ascii').split('\0', 1)[0]
            if w_target_type == XY_TARGET:
                target['target_type'] = struct_read('i', 4)[0]
                target['target_num_dots'] = struct_read('i', 4)[0]
                target['target_dot_units'] = struct_read('i', 4)[0]
                target['target_dot_life'] = struct_read('f', 4)[0]
                target['target_width'] = struct_read('f', 4)[0]
                target['target_height'] = struct_read('f', 4)[0]
                target['target_innter_rectangle_width'] = struct_read('f', 4)[0]
                target['target_inner_height'] = struct_read('f', 4)[0]
                target['target_inner_x'] = struct_read('f', 4)[0]
                target['target_inner_y'] = struct_read('f', 4)[0]
                target['state'] = struct_read('I', 4)[0]
                target['position_x'] = struct_read('f', 4)[0]
                target['position_y'] = struct_read('f', 4)[0]
                current_byte += 128 # Skip for same size as RMV_TARGET
            elif w_target_type == RMV_TARGET:
                target['target_type'] = struct_read('i', 4)[0]
                target['aperture'] = struct_read('i', 4)[0]
                target['flags'] = struct_read('i', 4)[0]

                # Append RGB mean and alpha values
                target['rgb_mean'] = [np.zeros((3, ), dtype=np.int64) for x in range(0, 2)]
                for rgb in range(0, 8):
                    if rgb < 3:
                        target['rgb_mean'][0][rgb] = int(struct_read('B', 1)[0])
                    elif rgb > 3 and rgb < 7:
                        target['rgb_mean'][1][rgb % 4] = int(struct_read('B', 1)[0])
                    else:
                        # rgb == 3 or 7
                        _ = struct_read('B', 1)[0]

                # Append RGB contrast and alpha values
                target['rgb_contrast'] = [np.zeros((3, ), dtype=np.int64) for x in range(0, 2)]
                for rgb in range(0, 8):
                    if rgb < 3:
                        target['rgb_contrast'][0][rgb] = int(struct_read('B', 1)[0])
                    elif rgb > 3 and rgb < 7:
                        target['rgb_contrast'][1][rgb % 4] = int(struct_read('B', 1)[0])
                    else:
                        # rgb == 3 or 7
                        _ = struct_read('B', 1)[0]

                if data['header']['version'] <= 22:
                    current_byte += 152 # TODO
                else:
                    current_byte += 152 + 12 # TODO
            elif w_target_type == 0:
                break
            else:
                raise RuntimeError('Invalid target type {:d}'.format(w_target_type))
            data['targets'].append(target)

    return data


def _read_stimulus_run(fp, data):
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    print('STIMULUS RUN NOT IMPLEMENTED')
    return data


TAG_SECTION_NAME_MAX_LENGTH = 18


def _read_tag_section(fp, data):
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    if not 'sections' in data:
        data['sections'] = []

    current_byte = 0
    while current_byte < RECORD_SIZE - RECORD_ID_LENGTH:
        segment = {}
        segment['name'] = struct.unpack('{:d}s'.format(TAG_SECTION_NAME_MAX_LENGTH), read_data[current_byte:current_byte+TAG_SECTION_NAME_MAX_LENGTH])[0].decode('ascii').split('\0', 1)[0]
        if len(segment['name']) == 0:
            break # Done
        segment['first_segment'] = read_data[current_byte + TAG_SECTION_NAME_MAX_LENGTH]
        segment['last_segment'] = read_data[current_byte + TAG_SECTION_NAME_MAX_LENGTH + 1]

        data['sections'] += [segment]
        current_byte += TAG_SECTION_NAME_MAX_LENGTH + 2

    return data


def _read_spike_sorted(fp, data):
    header_bytes = fp.read(RECORD_ID_LENGTH)
    read_data = fp.read(RECORD_SIZE-RECORD_ID_LENGTH)
    num_ints = int((RECORD_SIZE - RECORD_ID_LENGTH) / 4)
    data_ints = struct.unpack('<{:d}i'.format(num_ints), read_data[:num_ints * 4])

    if not 'sorted_spike_times' in data:
        data['sorted_spike_times'] = {};

    channel = header_bytes[0] - RECORD_TYPE_SPIKE_SORTED_MIN
    for current_int in data_ints:
        if current_int == END_OF_RECORD:
            break
        if not channel in data['sorted_spike_times']:
            data['sorted_spike_times'][channel] = [float(current_int) / 1E5] # Convert to S
        else:
            data['sorted_spike_times'][channel].append(data['sorted_spike_times'][channel][-1] + (float(current_int) / 1E5))

    return data


class MaestroTargetCalculator(object):
    """ An instance of a target with position, velocity and acceleration.  The
        default output is to show "commanded" target trajectories, i.e. not
        including large velocity transients due to sudden position steps.
        Commanded includes velocity stabilization "commands" that result from
        changes in eye position.  All changes are synchronized with the refresh
        rate and smoothly interpolated between refreshes.  Attempting to update
        the target at a time point that has already been passed will ruin things
        or raise an error - input times and updates can never move backward.

        """

    def __init__(self, data):
        self.n_time_points = data['header']['num_scans'] - 1
        self.position = np.zeros((2, data['header']['num_scans']))
        self.velocity = np.zeros((2, data['header']['num_scans']))
        self.acceleration = np.zeros((2, data['header']['num_scans']))
        self.visible = np.full(data['header']['num_scans'], False, 'bool')
        self.stabilization = np.full((2, data['header']['num_scans']), False, 'bool')
        self.v_stab_window = round(data['header']['vstab_sliding_window'] * 1000)
        self.lag_stabilization = True # If true, do stabilization from 2 frames behind
        self.abs_pos_override = [[False, -1], [False, -1]]
        self.last_command_time = None
        self.next_update_time = None
        self.last_update_time = None
        self.new_commands = False
        self.eye_position = np.stack((data['horizontal_eye_position'], data['vertical_eye_position']), axis=0)
        # seems like for some reason this changed with version?
        if data['header']['version'] < 21:
            self.frame_refresh_time = 1000.0 / data['header']['display_framerate']
        else:
            self.frame_refresh_time = 1000 * (1000.0 / data['header']['display_framerate'])

    def set_visible(self, visibility, time):
        # Set visible, assuming visibility is fixed until changed again.
        self.visible[self.get_next_refresh(time):] = visibility

    def get_next_refresh(self, from_time, n_forward=1):
        n_found = 0
        for t in range(from_time, self.n_time_points, 1):
            if t % self.frame_refresh_time < 1:
                n_found += 1
                if n_found == n_forward:
                    return t
        return None

    def get_last_refresh(self, from_time, n_back=1):
        n_found = 0
        for t in range(from_time, -1, -1):
            if t % self.frame_refresh_time < 1:
                n_found += 1
                if n_found == n_back:
                    return t
        return None

    def update_past(self):
        """ Update target trajectory from last_update_time to next_update_time.
            Remember that acceleration at time T affects velocity at T+1 and
            velocity at time T affects position at time T+1. """

        if self.next_update_time == 0:
            self.last_update_time = 0
            return
        if self.next_update_time <= self.last_update_time:
            print('next update {} is less than last update {}'.format(self.next_update_time, self.last_update_time))
            self.last_update_time = self.next_update_time
            return
        if self.next_update_time - self.last_update_time+1 < 1:
            return

        self.acceleration[0, self.last_update_time+1:self.next_update_time+1] = self.acceleration[0, self.last_update_time]
        self.acceleration[1, self.last_update_time+1:self.next_update_time+1] = self.acceleration[1, self.last_update_time]
        self.velocity[0, self.last_update_time+1:self.next_update_time+1] += (self.velocity[0, self.last_update_time] +
                                                                              self.acceleration[0, self.last_update_time:self.next_update_time].cumsum() * 0.001)
        self.velocity[1, self.last_update_time+1:self.next_update_time+1] += (self.velocity[1, self.last_update_time] +
                                                                              self.acceleration[1, self.last_update_time:self.next_update_time].cumsum() * 0.001)
        self.position[0, self.last_update_time+1:self.next_update_time+1] += (self.position[0, self.last_update_time] +
                                                                              self.velocity[0, self.last_update_time:self.next_update_time].cumsum() * 0.001)
        self.position[1, self.last_update_time+1:self.next_update_time+1] += (self.position[1, self.last_update_time] +
                                                                              self.velocity[1, self.last_update_time:self.next_update_time].cumsum() * 0.001)
        self.last_update_time = self.next_update_time

    def push_commands_to_next_frame(self):
        """ Take the target trajectory inputs present in last_command_time and calculate
            the values they produce at the next monitor refresh.  Assign these
            values to the next refresh time, and recalculate the target trajectory
            preceding this refresh time - OVERWRITING trajectory[last_command_time] in
            the process! Because this implements any new_commands, it resets
            new_commands = False.
            NOTE: for this to work properly, traces at last_command_time must
            be updated, either by new commands or update_past(). """

        next = self.get_next_refresh(self.last_command_time)
        if next is None:
            # No more refreshes to push to, so just update to this point
            self.next_update_time = self.n_time_points
            self.update_past()
            return

        if self.last_command_time != self.last_update_time:
            if self.last_command_time == 0:
                # I think this is fine, or necessary
                pass
            elif self.last_update_time is None:
                # Also should be fine, since haven't updated yet
                pass
            elif self.last_update_time > next:
                print('greater next?')
            else:
                # This is probably always an error?
                print('Not updated to command time, push to next frame might fail.')

        # Compute target at next refresh
        self.acceleration[0, self.last_command_time+1:next+1] = self.acceleration[0, self.last_command_time]
        self.acceleration[1, self.last_command_time+1:next+1] = self.acceleration[1, self.last_command_time]
        self.velocity[0, self.last_command_time+1:next+1] += (self.velocity[0, self.last_command_time] +
                                                              self.acceleration[0, self.last_command_time:next].cumsum() * 0.001)
        self.velocity[1, self.last_command_time+1:next+1] += (self.velocity[1, self.last_command_time] +
                                                              self.acceleration[1, self.last_command_time:next].cumsum() * 0.001)
        self.position[0, self.last_command_time+1:next+1] += (self.position[0, self.last_command_time] +
                                                             self.velocity[0, self.last_command_time:next].cumsum() * 0.001)
        self.position[1, self.last_command_time+1:next+1] += (self.position[1, self.last_command_time] +
                                                              self.velocity[1, self.last_command_time:next].cumsum() * 0.001)

        # Delete previous settings at this time point
        if self.last_command_time != next:
            self.acceleration[0, self.last_command_time:next] = 0
            self.acceleration[1, self.last_command_time:next] = 0
            self.velocity[0, self.last_command_time:next] = 0
            self.velocity[1, self.last_command_time:next] = 0
            self.position[0, self.last_command_time:next] = 0
            self.position[1, self.last_command_time:next] = 0

            # Recalculate as if most recent changes haven't occurred yet
            self.last_update_time = self.last_command_time - 1
            self.next_update_time = next - 1
            self.update_past()
        elif next == 0:
            # Still working on first frame, no need to backtrack and delete
            pass
        else:
            # This should imply the following, all of which are NOT errors and are the
            # result of the updates coming in exactly at a refresh time
            # next == self.last_command_time
            # self.last_update_time == self.next_update_time == self.last_command_time == next
            pass

        self.last_update_time = next
        self.new_commands = False

    def update_future(self):
        """ Updates target trajectory from last_update_time to the end of target duration.
            First implement any changes by calling push_commands_to_next_frame, then update
            trajectories to the end of trial under the assumption of no more changes. """

        # Make sure there are no updates that haven't been implemented yet
        if self.new_commands:
            self.push_commands_to_next_frame()
        else:
            pass

        if self.last_update_time == self.n_time_points:
            # Already updated to end
            return

        self.acceleration[0, self.last_update_time+1:] = self.acceleration[0, self.last_update_time]
        self.acceleration[1, self.last_update_time+1:] = self.acceleration[1, self.last_update_time]
        self.velocity[0, self.last_update_time+1:] += (self.velocity[0, self.last_update_time] + self.acceleration[0, self.last_update_time:-1].cumsum() * 0.001)
        self.velocity[1, self.last_update_time+1:] += (self.velocity[1, self.last_update_time] + self.acceleration[1, self.last_update_time:-1].cumsum() * 0.001)
        self.position[0, self.last_update_time+1:] += (self.position[0, self.last_update_time] + self.velocity[0, self.last_update_time:-1].cumsum() * 0.001)
        self.position[1, self.last_update_time+1:] += (self.position[1, self.last_update_time] + self.velocity[1, self.last_update_time:-1].cumsum() * 0.001)

    def check_updates(self, time):
        """ Checks that target updates are valid, then implements target trajectory
            updates only if time has moved forward from the last command time.
            Using this as a sort of "gatekeeper" for push_commands_to_next_frame()
            and update_past() allows the target to store and keep track of input
            commands and implement them appropriately. """

        if self.last_command_time is not None:
            if time > self.n_time_points:
                time = self.n_time_points
            if time < self.last_command_time:
                raise RuntimeError('Checked update at a time point that moved backwards at time {} with last command at {}'.format(time, self.last_command_time))
            if self.last_update_time is not None and time < self.last_update_time:
                raise RuntimeError('Checked update at a time point that moved backwards')

            if time > self.last_command_time:
                if self.new_commands:
                    # If there were new commands not implemented, implement at next frame
                    self.push_commands_to_next_frame()

                # Then update through to present time before accepting new commands
                self.next_update_time = time
                self.update_past()
            elif time == self.last_update_time or time == self.last_command_time:
                # This should be indicator of multiple commands at the same time, where
                # the first command has already been processed by above statement
                pass
            else:
                # Not sure if this is an error or can happen without raising one of above erros
                print('nothing to check?')

            # Check if time has passed beyond an absolute position command and reset if yes
            if self.abs_pos_override[0][1] < time:
                self.abs_pos_override[0][0] = False
            if self.abs_pos_override[1][1] < time:
                self.abs_pos_override[1][0] = False
        else:
            pass
            # Something like this might help catch errors?
            # self.last_command_time = None
            # self.next_update_time = None
            # self.last_update_time = None

    def set_h_postion(self, position, time, rel=False):
        if time >= self.n_time_points:
            return
        self.check_updates(time)
        self.new_commands = True
        self.last_command_time = time
        if rel:
            self.position[0, time] = position + self.position[0, time - 1] + (self.velocity[0, time - 1] * 0.001)
        else:
            self.abs_pos_override[0][0] = True
            self.abs_pos_override[0][1] = time
            self.position[0, time] = position

    def set_v_postion(self, position, time, rel=False):
        if time >= self.n_time_points:
            return
        self.check_updates(time)
        self.new_commands = True
        self.last_command_time = time
        if rel:
            self.position[1, time] = position + self.position[1, time - 1] + (self.velocity[1, time - 1] * 0.001)
        else:
            self.abs_pos_override[1][0] = True
            self.abs_pos_override[1][1] = time
            self.position[1, time] = position

    def set_h_velocity(self, velocity, time):
        if time >= self.n_time_points:
            return
        self.check_updates(time)
        self.new_commands = True
        self.last_command_time = time
        self.velocity[0, time] = velocity

    def set_v_velocity(self, velocity, time):
        if time >= self.n_time_points:
            return
        self.check_updates(time)
        self.new_commands = True
        self.last_command_time = time
        self.velocity[1, time] = velocity

    def set_h_acceleration(self, acceleration, time):
        if time >= self.n_time_points:
            return
        self.check_updates(time)
        self.new_commands = True
        self.last_command_time = time
        self.acceleration[0, time] = acceleration

    def set_v_acceleration(self, acceleration, time):
        if time >= self.n_time_points:
            return
        self.check_updates(time)
        self.new_commands = True
        self.last_command_time = time
        self.acceleration[1, time] = acceleration

    def stabilize_horizontal(self, time1, time2):
        """ Check if velocity stabilization is on for any direction at any time point
            between time1 and time2.  If it is on, implement velocity stabilization
            by adding the change in eye position at intervals of the refresh rate
            while stabilization is true.  Returns true if positions were updated
            and false otherwise. The first 3 lines implement stabilization 2 frames
            behind by shifting t backward, which should be what Maestro actually does
            but is NOT what readcxdata does.  Leaving these lines as noted gives
            output that matches readcxdata. """

        if np.all(self.stabilization[0, time1:time2], axis=0) and self.stabilization[0, time1:time2].size > 1:
            self.check_updates(time2)
            self.new_commands = True
            self.last_command_time = time2

            if self.lag_stabilization:
                eye_time = np.ceil(time1 - 2 * self.frame_refresh_time).astype('int')
            else:
                eye_time = time1
            last_eye_position = np.mean(self.eye_position[0, eye_time+1 - self.v_stab_window:eye_time+1])
            next = self.get_next_refresh(time1 + 1)
            while next < time2:
                if self.lag_stabilization:
                    eye_time = self.get_last_refresh(next-1, n_back=2)
                else:
                    eye_time = next
                curr_eye_position = np.mean(self.eye_position[0, eye_time+1 - self.v_stab_window:eye_time+1])
                self.position[0, next:time2] += curr_eye_position - last_eye_position
                last_eye_position = curr_eye_position
                next = self.get_next_refresh(next + 1)

            if self.lag_stabilization:
                eye_time = np.ceil(time2 - 2 * self.frame_refresh_time).astype('int')
            else:
                eye_time = time2
            curr_eye_position = np.mean(self.eye_position[0, eye_time+1 - self.v_stab_window:eye_time+1])
            if not self.abs_pos_override[0][0]:
                # Need to make sure there wasn't an absolute position command at
                # this time before setting stabilization
                self.position[0, time2] += (self.position[0, time2-1] + curr_eye_position - last_eye_position)
            return True
        else:
            return False

    def stabilize_vertical(self, time1, time2):
        """ Same as horizontal but for vertical. """

        if np.all(self.stabilization[1, time1:time2], axis=0) and self.stabilization[1, time1:time2].size > 1:
            self.check_updates(time2)
            self.new_commands = True
            self.last_command_time = time2

            if self.lag_stabilization:
                eye_time = np.ceil(time1 - 2 * self.frame_refresh_time).astype('int')
            else:
                eye_time = time1
            last_eye_position = np.mean(self.eye_position[1, eye_time+1 - self.v_stab_window:eye_time+1])
            next = self.get_next_refresh(time1 + 1)
            while next < time2:
                if self.lag_stabilization:
                    eye_time = self.get_last_refresh(next-1, n_back=2)
                else:
                    eye_time = next
                curr_eye_position = np.mean(self.eye_position[1, eye_time+1 - self.v_stab_window:eye_time+1])
                self.position[1, next:time2] += curr_eye_position - last_eye_position
                last_eye_position = curr_eye_position
                next = self.get_next_refresh(next + 1)

            if self.lag_stabilization:
                eye_time = np.ceil(time2 - 2 * self.frame_refresh_time).astype('int')
            else:
                eye_time = time2
            curr_eye_position = np.mean(self.eye_position[1, eye_time+1 - self.v_stab_window:eye_time+1])
            if not self.abs_pos_override[1][0]:
                # Need to make sure there wasn't an absolute position command at
                # this time before setting stabilization
                self.position[1, time2] += (self.position[1, time2-1] + curr_eye_position - last_eye_position)
            return True
        else:
            return False

    def set_stabilization(self, stabs, snap, time):
        """ Sets velocity stabilization snap position and on/off flag values.  Snap
            is sent as eye position at the next refresh time.  Stabilization flags
            are set to the input value from current time until trial end. """
        next = self.get_next_refresh(time)
        if next is None:
            # There can't be any stabilization without an ensuing refresh so do nothing
            return
        curr_eye_position = [np.mean(self.eye_position[0, max(0, next+1 - self.v_stab_window):next+1]),
                             np.mean(self.eye_position[1, max(0, next+1 - self.v_stab_window):next+1])]

        # Update targets if velocity stabilization is on
        if snap and stabs[0]:
            print('Not sure how snap will interact with perform_stabilization method')
            self.set_h_postion(curr_eye_position[0], next, rel=False)
        if snap and stabs[1]:
            print('Not sure how snap will interact with perform_stabilization method')
            self.set_v_postion(curr_eye_position[1], next, rel=False)

        self.stabilization[0, time:] = stabs[0]
        if not stabs[0]:
            # Stabilization turned off, so find when it started and execute it only in this window
            for index, vals in zip(range(time - 1, -1, -1), self.stabilization[0, time - 1::-1]):
                if not self.stabilization[0, index]:
                    t_start = index + 1
                    self.stabilize_horizontal(t_start, time)
                    break

        self.stabilization[1, time:] = stabs[1]
        if not stabs[1]:
            # Stabilization turned off, so find when it started and execute it only in this window
            for index, vals in zip(range(time - 1, -1, -1), self.stabilization[1, time - 1::-1]):
                if not self.stabilization[1, index]:
                    t_start = index + 1
                    self.stabilize_vertical(t_start, time) # DELETED time - 1 FOR SECOND INPUT TO STOP BACKWARDS!
                    break

    """ Some extra functions that make the target trace outputs look different, like nan when
        invisible, or make it match readcxdata. """

    def nan_invisible(self):
        self.position[:, ~self.visible] = np.nan
        self.velocity[:, ~self.visible] = np.nan
        self.acceleration[:, ~self.visible] = np.nan


    """ These functions aren't really working/useful as is, but it may be good to
        be able to modify the outputs to reflect things like velocity stabilization
        velocity, and make traces look like readcxdata?? """

    def smooth_on_refresh(self, trace='position', commanded=True):
        first_step = False
        for t in range(1, self.n_time_points):
            if ~first_step and np.any(np.absolute(getattr(self, trace)[:, t] - getattr(self, trace)[:, t - 1]) > 0):
                first_step = True
            else:
                continue
            if np.floor((t // self.frame_refresh_time) * self.frame_refresh_time) < t - 1:
                getattr(self, trace)[:, t] = getattr(self, trace)[:, t-1]
            else:
                break

    def step_on_refresh(self, trace='position', commanded=True):
        for t in range(1, self.n_time_points):
            if np.floor((t // self.frame_refresh_time) * self.frame_refresh_time) < t - 1:
                getattr(self, trace)[:, t] = getattr(self, trace)[:, t - 1]

    def get_readcxdata_velocity(self):
        return np.diff(self.position)


def _process_codes_compute_target_motion(data):
    """Process the trial codes for this trial into target motion events.
    """
    # Create targets for each of the targets we have previously parsed to keep
    # track of changes in position and velocity in between screen refreshes.
    _targets = [MaestroTargetCalculator(data) for _ in range(len(data['targets']))]

    # The current trial code we are processing
    trial_code_index = 0

    while True:
        code = data['trial_codes'][1][trial_code_index]
        code_time = data['trial_codes'][0][trial_code_index]

        # Get index of target being updated (unused if code is not a target update)
        try:
            target_index = data['trial_codes'][1][trial_code_index + 1]
        except (IndexError):
            target_index = None

        if trial_code_index + 1 > len(data['trial_codes'][0]):
            raise RuntimeError("Invalid target value encountered at end of trial codes")
        code_value = data['trial_codes'][0][trial_code_index + 1]

        if code == TRIAL_CODE_START_TRIAL:
            trial_code_index += 1
        elif code == TRIAL_CODE_END_TRIAL:
            trial_code_index += 1
            break # End of trial, we are done
        elif code == TRIAL_CODE_TARGET_ON:
            _targets[target_index].set_visible(True, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_OFF:
            _targets[target_index].set_visible(False, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_HORIZONTAL_VELOCITY_CHANGE:
            _targets[target_index].set_h_velocity(code_value / 10.0, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_HORIZONTAL_VELOCITY_LO:
            _targets[target_index].set_h_velocity(code_value / 500.0, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_VERTICAL_VELOCITY_CHANGE:
            _targets[target_index].set_v_velocity(code_value / 10.0, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_VERTICAL_VELOCITY_LO:
            _targets[target_index].set_v_velocity(code_value / 500.0, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_HORIZONTAL_ACCELERATION_LO:
            _targets[target_index].set_h_acceleration(code_value / 100.0, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_VERTICAL_ACCELERATION_LO:
            _targets[target_index].set_v_acceleration(code_value / 100.0, code_time)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_HORIZONTAL_POSITION_ABSOLUTE:
            _targets[target_index].set_h_postion(code_value / 100.0, code_time, rel=False)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_HORIZONTAL_POSITION_RELATIVE:
            _targets[target_index].set_h_postion(code_value / 100.0, code_time, rel=True)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_VERTICAL_POSITION_ABSOLUTE:
            _targets[target_index].set_v_postion(code_value / 100.0, code_time, rel=False)
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_VERTICAL_POSITION_RELATIVE:
            _targets[target_index].set_v_postion(code_value / 100.0, code_time, rel=True)
            trial_code_index += 2
        elif code == TRIAL_CODE_MIDTRIAL_REWARD:
            trial_code_index += 2 # Next is reward pulse length in ms
        elif code == TRIAL_CODE_REWARD_PULSE_LENGTH:
            trial_code_index += 2 # N = 2 (next is reward pulse length)
        elif code == TRIAL_CODE_ADC_ON:
            trial_code_index += 1
        elif code == TRIAL_CODE_FAIL_SAFE:
            data['header']['fail_safe_time'] = code_time # Time in ms
            trial_code_index += 3
        elif code == TRIAL_CODE_TARGET_PERTURBATION:
            trial_code_index += 5 # TODO: Perturbations not implemented
        elif code == TRIAL_CODE_PULSE_ON or code == TRIAL_CODE_FIXATION_1 \
            or code == TRIAL_CODE_FIXATION_2 or code == TRIAL_CODE_FIXATION_ACCURACY:
            trial_code_index += 2
        elif code == TRIAL_CODE_TARGET_VELOCITY_STABILIZATION:
            stabs = [False, False]
            snap = False
            if (code_value & (1 << 0)): # Velocity stabilization on
                if (code_value & (1 << 1)):
                    snap= True
                else:
                    snap = False
                if (code_value & (1 << 2)):
                    stabs[0] = True
                else:
                    stabs[0] = False
                if (code_value & (1 << 3)):
                    stabs[1] = True
                else:
                    stabs[1] = False
            _targets[target_index].set_stabilization(stabs, snap, code_time)
            # Set flag that stabilization of any kind was used on this trial
            data['header']['UsedStab'] = True
            trial_code_index += 2
        else:
            raise RuntimeError('Error invalid or unhandled trial code {:d} at time {:d}'.format(code, index))

    # Now that all codes are entered, finish target trajectories by calling
    # update_future and output them to data
    for t in range(len(_targets)):
        _targets[t].update_future()
        _targets[t].nan_invisible()
        data['horizontal_target_position'][t, :] = _targets[t].position[0, :]
        data['vertical_target_position'][t, :] = _targets[t].position[1, :]
        data['horizontal_target_velocity'][t, :] = _targets[t].velocity[0, :]
        data['vertical_target_velocity'][t, :] = _targets[t].velocity[1, :]

    return data
