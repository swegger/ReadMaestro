"""
Copyright (c) 2018 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>

Nathan Hall:
Modified to be a class that keeps basic information to make
reading the files easier.
"""

import struct
import os
import numpy as np
import warnings

# Define the constants for this module
# Constants for data blocks
PL2_DATA_BLOCK_ANALOG_CHANNEL = (0x42)
PL2_DATA_BLOCK_EVENT_CHANNEL = (0x5A)
PL2_DATA_BLOCK_SPIKE_CHANNEL = (0x31)
PL2_DATA_BLOCK_START_STOP_CHANNEL = (0x59)

# Constants for footers
PL2_FOOTER_ANALOG_CHANNEL = (0xDD)
PL2_FOOTER_EVENT_CHANNEL = (0xDF)
PL2_FOOTER_SPIKE_CHANNEL = (0xDE)
PL2_FOOTER_START_STOP_CHANNEL = (0xE0)

# Constants for headers
PL2_HEADER_ANALOG_CHANNEL = (0xD4)
PL2_HEADER_SPIKE_CHANNEL = (0xD5)
PL2_HEADER_EVENT_CHANNEL = (0xD6)

# Data types
PL2_ANALOG_TYPE_WB = (0x03)
PL2_ANALOG_TYPE_AI = (0x0C)
PL2_ANALOG_TYPE_AI2 = (0x0D) # New in version 1.19.2  --- not sure why
PL2_ANALOG_TYPE_FP = (0x07)
PL2_ANALOG_TYPE_SPKC = (0x04)
PL2_EVENT_TYPE_SINGLE_BIT = (0x09)
PL2_EVENT_TYPE_STROBED = (0x0A)
PL2_SPIKE_TYPE_SPK = (0x06)
PL2_SPIKE_TYPE_SPK_SPKC = (0x01)


class PL2Reader(object):
    """ Creates a reader for the PL2 file input in filename.  This makes it easier
        to store the file information and request data from input channels. Mostly
        calls David's original functions but there are a couple extra helpful
        functions. """

    def __init__(self, filename):
        self.filename = filename
        self.info = load_file_information(filename)
        # May want to have somewhere here to put data that is read?  Could make
        # an empty dictionary or list or something that adds this stuff on?

    def load_all(self):
        """ Load all of the information from a PL2 and store it in the dictionary
            self.info. """

        with open(self.filename, 'rb') as fp:
            _parse_data_blocks(fp, self.info)

    def load_channel(self, channel, convert_to_mv=True):
        """ Takes a channel name string as input and tries to figure out which
            type of channel loading function to use to return data for the
            channel. This function makes it unnecessary to ever actually call
            a specific load operation and eliminates the burden of figuring
            out channel order indices in the PL2 file. """

        # First do a brute force search through all channel names to ensure
        # that exact inputs are not skipped
        channel = channel.upper()
        for spike_name in self.info['file_channel_indices']['spike'].keys():
            if spike_name == channel:
                return self.load_spike_channel(channel, convert_to_mv=convert_to_mv)
        for event_name in self.info['file_channel_indices']['event'].keys():
            if event_name == channel:
                return self.load_event_channel(channel)
        for analog_name in self.info['file_channel_indices']['analog'].keys():
            if analog_name == channel:
                return self.load_analog_channel(channel, convert_to_mv=convert_to_mv)

        # If above didn't work, try to identify channel type
        channel = channel.lower()
        if 'spk' in channel and not 'spkc' in channel:
            chan_type = 'spike'
        elif 'spk_' in channel:
            chan_type = 'spike'
        elif 'spkc' in channel or 'cont' in channel:
            chan_type = 'analog'
        elif 'sp' in channel:
            chan_type = 'spike'
        elif 'analog' in channel or 'ai' in channel:
            chan_type = 'analog'
        elif 'fp' in channel or 'field' in channel:
            chan_type = 'analog'
        elif 'ev' in channel or 'kbd' in channel:
            chan_type = 'event'
        else:
            chan_type = None

        # Load channel if chan_type was identified
        if chan_type == 'spike':
            return self.load_spike_channel(channel, convert_to_mv=convert_to_mv)
        elif chan_type == 'analog':
            return self.load_analog_channel(channel, convert_to_mv=convert_to_mv)
        elif chan_type == 'event':
            return self.load_event_channel(channel)
        else:
            # If function makes it to here, the channel wasn't found
            raise RuntimeError("Could not identify channel {}".format(channel))

    def load_spike_channel(self, channel, convert_to_mv=True):
        """ Return the waveforms for a given spike channel. """

        # Check the channel input and translate strings to number indices
        channel = self.get_channel_index(channel, 'spike')

        # Ensure that the appropriate spike channel exists and there is data there
        if channel >= len(self.info["spike_channels"]) or channel < 0:
            raise RuntimeError("Invalid channel number.\n")

        if not "block_offsets" in self.info["spike_channels"][channel] or len(self.info["spike_channels"][channel]["block_offsets"]) == 0:
            print("Can't find block_offsets {}".format(self.info["spike_channels"][channel].keys()))
            return None

        # Attempt to load the results
        with open(self.filename, 'rb') as fp:
            total_items = np.sum(self.info["spike_channels"][channel]["block_num_items"])
            # Allocate space for the results
            results = {}
            results["num_points"] = self.info["spike_channels"][channel]["samples_per_spike"]
            results["timestamps"] = np.empty((total_items), dtype=np.uint64)
            results["spikes"] = np.empty((total_items, results["num_points"]), dtype=np.int16)
            results["assignments"] = np.empty((total_items), dtype=np.uint16)

            for i in range(0, len(self.info["spike_channels"][channel]["block_offsets"])):
                block_offset = self.info["spike_channels"][channel]["block_offsets"][i]
                fp.seek(block_offset)
                data_type = _read(fp, "<B")
                data_subtype = _read(fp, "<B")

                _read(fp, "<H")
                _read(fp, "<H") # Channel
                num_sample_points = _read(fp, "<H")
                # print(self.info["spike_channels"][channel]["samples_per_spike"], num_sample_points)
                if num_sample_points != self.info["spike_channels"][channel]["samples_per_spike"]:
                    pass
                    # print(self.info["spike_channels"][channel]["samples_per_spike"], num_sample_points)
                    # raise RuntimeError("Invalid number of samples per spike encounted in file.\n")

                num_items = _read(fp, "<Q")
                if num_items != self.info["spike_channels"][channel]["block_num_items"][i]:
                    raise RuntimeError("Invalid number of items encountered in file: expected {:d} but got {:d}".format(self.info["spike_channels"][channel]["block_num_items"], num_items))

                # _read each of the items
                start = sum(self.info["spike_channels"][channel]["block_num_items"][0:(i)])
                stop = start + num_items
                results["timestamps"][start:stop] = _read(fp, "<{:d}Q".format(num_items))
                results["assignments"][start:stop] = _read(fp, "<{:d}H".format(num_items))
                for j in range(start, stop):
                    results["spikes"][j, :] = _read(fp, "<{:d}h".format(results["num_points"]))

        results["timestamps"] = np.array(results["timestamps"])
        results["assignments"] = np.array(results["assignments"])
        results["spikes"] = np.array(results["spikes"])

        # Scale results to mV
        results["timestamps"] = results["timestamps"].astype(np.float64)
        results["timestamps"] /= self.info["timestamp_frequency"]
        if convert_to_mv:
            results["spikes"] = results["spikes"].astype(np.float64)
            results["spikes"] *= self.info["spike_channels"][channel]["coeff_to_convert_to_units"] * 1000 # to mV

        return results

    def load_analog_channel(self, channel, convert_to_mv=True, out=None):
        """ Loads an analog channel from a given file. This function returns the
        results as an array of double precision floating point numbers scaled
        by the conversion factor specified in the file header if convert_to_mv
        is true. Otherwise it returns the raw 16 bit values. """

        channel = self.get_channel_index(channel, 'analog')

        # Ensure that the appropriate analog channel exists and there is data there
        if channel >= len(self.info["analog_channels"]) or channel < 0:
            raise RuntimeError("Invalid channel number.\n")

        if ( (not "block_offsets" in self.info["analog_channels"][channel]) or
             (len(self.info["analog_channels"][channel]["block_offsets"]) == 0) or
             (self.info["analog_channels"][channel]['num_values'] == 0) ):
            # There is no data on this channel
            print("No analog data found for channel:", channel)
            return None

        total_items = sum(self.info["analog_channels"][channel]["block_num_items"])
        # Allocate space for the ouput if needed
        if out is None:
            if convert_to_mv:
                out = np.empty((total_items), dtype=np.float64)
            else:
                out = np.empty((total_items), dtype=np.int16)
        if type(out) is not np.ndarray:
            raise ValueError("Output must be a numpy array of size self.info['analog_channels'][channel]['num_values']")

        if out.size != total_items:
            raise ValueError("Output must be a numpy array of size self.info['analog_channels'][channel]['num_values']")

        # Attempt to load the results
        with open(self.filename, 'rb') as fp:
            for i in range(0, len(self.info["analog_channels"][channel]["block_offsets"])):
                block_offset = self.info["analog_channels"][channel]["block_offsets"][i]
                fp.seek(block_offset)
                data_type = _read(fp, "<B")
                data_subtype = _read(fp, "<B")

                num_items = _read(fp, "<H")
                if num_items != self.info["analog_channels"][channel]["block_num_items"][i]:
                    raise RuntimeError("Invalid numer of items encountered in file.\n")
                _read(fp, "<H") # Channel
                _read(fp, "<H") # Unknown
                timestamp = _read(fp, "<Q") # Timestamp
                if timestamp != self.info["analog_channels"][channel]["block_timestamps"][i]:
                    raise RuntimeError("Invalid timestamp encountered in file.\n")

                # _read each of the items
                values = _read(fp, "<{:d}h".format(num_items))
                start = sum(self.info["analog_channels"][channel]["block_num_items"][0:(i)])
                stop = start + num_items
                out[start:stop] = values

        if convert_to_mv:
            out *= self.info["analog_channels"][channel]["coeff_to_convert_to_units"] * 1000 # to mV

        return out

    def load_event_channel(self, channel):
        """ Load all of the events from a from a given event channel. The data is returned
        as a dictionary of timestamps and strobed values. """

        channel = self.get_channel_index(channel, 'event')

        # Ensure that the appropriate event channel exists and there is data there
        if channel >= len(self.info["event_channels"]) or channel < 0:
            raise RuntimeError("Invalid channel number.\n")

        if not "block_offsets" in self.info["event_channels"][channel] or len(self.info["event_channels"][channel]["block_offsets"]) == 0:
            return None

        # Attempt to load the results
        with open(self.filename, 'rb') as fp:
            total_items = sum(self.info["event_channels"][channel]["block_num_items"])
            # Allocate space for the results
            results = {}
            results["timestamps"] = np.zeros((total_items), dtype=np.uint64)
            results["strobed"] = np.zeros((total_items), dtype=np.uint16)
            for i in range(0, len(self.info["event_channels"][channel]["block_offsets"])):
                block_offset = self.info["event_channels"][channel]["block_offsets"][i]
                fp.seek(block_offset)
                data_type = _read(fp, "<B")
                data_subtype = _read(fp, "<B")

                _read(fp, "<H")
                _read(fp, "<H") # Channel
                num_items = _read(fp, "<Q")
                if num_items != self.info["event_channels"][channel]["block_num_items"][i]:
                    raise RuntimeError("Invalid number of items encountered in file.\n")
                _read(fp, "<H")

                # _read each of the items
                start = sum(self.info["event_channels"][channel]["block_num_items"][0:(i)])
                stop = start + num_items
                results["timestamps"][start:stop] = _read(fp, "<{:d}Q".format(num_items))
                results["strobed"][start:stop] = _read(fp, "<{:d}H".format(num_items))

        results["timestamps"] = np.array(results["timestamps"])
        results["strobed"] = np.array(results["strobed"])
        results["timestamps"] = results["timestamps"].astype(np.float64)
        results["timestamps"] /= self.info["timestamp_frequency"]

        return results

    def get_channel_index(self, channel, chan_type):
        """ Since data is all read based on index numbers that may have an unclear
            relationship with the channel names and numbers, this function tries
            to assign a channel input with it's correct index. """

        if isinstance(chan_type, str):
            chan_type = chan_type.lower()
            if chan_type == 'spike':
                chan_type = 'spike'
            elif chan_type == 'analog':
                chan_type = 'analog'
            elif chan_type == 'event':
                chan_type = 'event'
            elif 'spkc' in chan_type or 'cont' in chan_type:
                chan_type = 'analog'
            elif 'sp' in chan_type:
                chan_type = 'spike'
            elif 'analog' in chan_type or 'ai' in chan_type:
                chan_type = 'analog'
            elif 'fp' in chan_type or 'field' in chan_type:
                chan_type = 'analog'
            elif 'ev' in chan_type or 'kbd' in chan_type:
                chan_type = 'event'
            else:
                print("Unrecognized channel type {}. Valid types are 'spike', 'analog' and 'continuous'.".format(chan_type))
        else:
            raise TypeError("Input chan_type must be a string")

        # If channel entered as string, pull associated dictionary channel index
        if isinstance(channel, str):
            channel = channel.upper()
            key_data = self.info['file_channel_indices'][chan_type].get(channel.upper())
            if key_data is None:
                try:
                    channel_index = int(channel)
                except ValueError:
                    raise ValueError("Input channel {} not found".format(channel))
            else:
                channel_index = self.info['file_channel_indices'][chan_type][channel]
        else:
            channel_index = int(channel)

        return channel_index

""" These are some core functions used by the reader class to estabilish it, or
    support its method calls, but that are not methods that can just be called. """

def load_file_information(filename):
    """ Loads all of the information in a PL2 file and returns the associated file
        information dictionary. This information can be passed to future functions to
        speed up processing. """

    with open(filename, 'rb') as fp:
        data = {}
        # _read the header from the file (storing contents in the dictionary)
        _read_header(fp, data)

        # Create empty start/stop channel values
        data["start_stop_channels"] = {}
        data["start_stop_channels"]["block_offsets"] = []
        data["start_stop_channels"]["block_timestamps"] = []
        data["start_stop_channels"]["block_num_items"] = []
        data["start_stop_channels"]["num_events"] = 0
        data['file_channel_indices'] = {}

        fp.seek(0x480)
        data["spike_channels"] = []
        data['file_channel_indices']['spike'] = {}
        for i in range(0, data["total_number_of_spike_channels"]):
            # Tag indices for easier, and text input lookups
            spike_dictionary = _read_spike_channel_header(fp)
            data["spike_channels"].append(spike_dictionary)
            data['file_channel_indices']['spike'][spike_dictionary['name'].upper()] = i

        data["analog_channels"] = []
        data['file_channel_indices']['analog'] = {}
        for i in range(0, data["total_number_of_analog_channels"]):
            # Tag indices for easier, and text input lookups
            analog_dictionary = _read_analog_channel_header(fp)
            data["analog_channels"].append(analog_dictionary)
            data['file_channel_indices']['analog'][analog_dictionary['name'].upper()] = i

        data["event_channels"] = []
        data['file_channel_indices']['event'] = {}
        for i in range(0, data["number_of_event_channels"]):
            event_dictionary = _read_event_channel_header(fp)
            data["event_channels"].append(event_dictionary)
            # Tag indices for easier, and text input lookups
            data['file_channel_indices']['event'][event_dictionary['name'].upper()] = i

        if data["internal_value_4"] != 0:
            _read_footer(fp, data)
        else:
            _reconstruct_footer(fp, data)

    return data


"""
    _read(fp, data_type)
Reads series of bytes from a file pointer and unpacks them using struct.unpack
This function serves to avoid needing a byte array that is exactly the same
size of the size of the data type
"""
def _read(fp, data_types, force_list=False):
    num_bytes = struct.calcsize(data_types)
    _read_bytes = fp.read(num_bytes)
    values = struct.unpack(data_types, _read_bytes)
    if len(values) == 1:
        if force_list:
            return [values[0]]
        return values[0]
    else:
        if force_list:
            return list(values)
        return values

"""
    _read_header(fp, data)
Given a file pointer, _read the contents of the PL2 header from the
start of the file.
"""
def _read_header(fp, data):
    fp.seek(0, os.SEEK_END)
    data["file_length"] = fp.tell()
    fp.seek(0)

    data["version"] = {}
    data["version"]["major_version"] = _read(fp, "<B") # "<B"
    data["version"]["minor_version"] = _read(fp, "<B")
    data["version"]["bug_version"] = _read(fp, "<B")

    fp.seek(0x20)
    data["internal_value_1"] = _read(fp, "<Q") # End of header
    data["internal_value_2"] = _read(fp, "<Q") # First data block
    data["internal_value_3"] = _read(fp, "<Q") # End of data blocks
    data["internal_value_4"] = _read(fp, "<Q") # This is the start of footer
    data["start_recording_time_ticks"] = _read(fp, "<Q")
    data["duration_of_recording_ticks"] = _read(fp, "<Q")

    fp.seek(0xE0)
    data["creator_comment"] = bytearray(_read(fp, "<256B")).decode('ascii').split('\0', 1)[0]
    data["creator_software_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["creator_software_version"] = bytearray(_read(fp, "<16B")).decode('ascii').split('\0', 1)[0]
    data["creator_date_time"] = _read_date_time(fp)
    data["timestamp_frequency"] = _read(fp, "<d")
    data["duration_of_recording_sec"] = data["duration_of_recording_ticks"] / data["timestamp_frequency"]
    _read(fp, "<I") # TODO: Off by 4 bytes
    data["total_number_of_spike_channels"] = _read(fp, "<I")
    data["number_of_recorded_spike_channels"] = _read(fp, "<I")
    data["total_number_of_analog_channels"] = _read(fp, "<I")
    data["number_of_recorded_analog_channels"] = _read(fp, "<I")
    data["number_of_event_channels"] = _read(fp, "<I")
    data["minimum_trodality"] = _read(fp, "<I")
    data["maximum_trodality"] = _read(fp, "<I")
    data["number_of_non_omiplex_sources"] = _read(fp, "<I")

    fp.seek(4, os.SEEK_CUR)
    data["reprocessor_comment"] = bytearray(_read(fp, "<256B")).decode('ascii').split('\0', 1)[0]
    data["reprocessor_software_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    #data["reprocessor_date_time"] = _read_date_time(fp)


"""
    parse_data_blocks(fp, data)
Parses the data blocks in sequence, appending items to the internal lists as we
go. This function ensures that all of the data is _read (without relying on the
data in the footer).
"""
def _parse_data_blocks(fp, data):
    # Seek to the first data block and begin _reading
    fp.seek(data["internal_value_2"])

    if data["internal_value_3"] == 0:
        data["internal_value_3"] = data["internal_value_4"] # File is not complete

    while (fp.tell() < data["internal_value_3"]):
        # _read type
        data_type = _read(fp, "<B")
        data_subtype = _read(fp, "<B")

        if (data_type == PL2_DATA_BLOCK_ANALOG_CHANNEL):
            num_items = _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H")) # Python is base 0
            _read(fp, "<H") # Unknown
            timestamp = _read(fp, "<Q") # Unknown

            # _read each of the items
            values = _read(fp, "<{:d}h".format(num_items), True)

            # Store in the output
            if not "values" in data["analog_channels"][channel]:
                data["analog_channels"][channel]["values"] = values
            else:
                data["analog_channels"][channel]["values"].extend(values)
        elif (data_type == PL2_DATA_BLOCK_SPIKE_CHANNEL):
            _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H")) # Python is base 0
            num_sample_points = _read(fp, "<H")
            num_items = _read(fp, "<Q")

            # _read each of the items (64 byte values)
            timestamps = _read(fp, "<{:d}Q".format(num_items), True)
            assignments = _read(fp, "<{:d}H".format(num_items), True) # These are probably assignments
            spikes = np.zeros((num_items, num_sample_points), dtype=np.int16)
            for i in range(0, num_items):
                spikes[i, :] = _read(fp, "<{:d}h".format(num_sample_points)) # _read actual sample points
            if not "timestamps" in data["spike_channels"][channel]:
                data["spike_channels"][channel]["timestamps"] = timestamps
                data["spike_channels"][channel]["assignments"] = assignments
                data["spike_channels"][channel]["spikes"] = spikes
            else:
                data["spike_channels"][channel]["timestamps"].extend(timestamps)
                data["spike_channels"][channel]["assignments"].extend(assignments)
                data["spike_channels"][channel]["spikes"] = np.append(data["spike_channels"][channel]["spikes"], spikes, axis=0)
        elif (data_type == PL2_DATA_BLOCK_EVENT_CHANNEL):
            _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H")) # Python is base 0
            num_items = _read(fp, "<Q")
            _read(fp, "<H")
            timestamps = _read(fp, "<{:d}Q".format(num_items), True)
            strobed = _read(fp, "<{:d}H".format(num_items), True)

            if not "timestamps" in data["event_channels"][channel]:
                data["event_channels"][channel]["timestamps"] = timestamps
                data["event_channels"][channel]["strobed"] = strobed
            else:
                data["event_channels"][channel]["timestamps"].extend(timestamps)
                data["event_channels"][channel]["strobed"].extend(strobed)
        elif (data_type == PL2_DATA_BLOCK_START_STOP_CHANNEL):
            # Start-stop Channel
            _read(fp, "<H")
            channel = _read(fp, "<H") - 1
            num_items = _read(fp, "<Q")
            timestamps = list(_read(fp, "<{:d}Q".format(num_items)))
            assignments = list(_read(fp, "<{:d}H".format(num_items)))
            if not "timestamps" in data["start_stop_channels"]:
                data["start_stop_channels"]["timestamps"] = timestamps
                data["start_stop_channels"]["assignments"] = assignments
            else:
                data["start_stop_channels"]["timestamps"].extend(timestamps)
                data["start_stop_channels"]["assignments"].extend(assignments)
        else:
            raise RuntimeError("Unknown data type at position ", position(fp) - 2, "Got value: ", data_type)
        # Align to next 16 byte boundary
        fp.seek(int((fp.tell() + 15) / 16) * 16)


"""
    _read_date_time(fp)
Given a file pointer, _read a date/time struction from the current location
of the file. A dictionary is returned.
"""
def _read_date_time(fp):
    data = {}
    data["second"] = _read(fp, "<I")
    data["minute"] = _read(fp, "<I")
    data["hour"] = _read(fp, "<I")
    data["month_day"] = _read(fp, "<I")
    data["month"] = _read(fp, "<I")
    data["year"] = _read(fp, "<I")
    data["week_day"] = _read(fp, "<I")
    data["year_day"] = _read(fp, "<I")
    data["is_daylight_savings"] = _read(fp, "<I")
    data["millisecond"] = _read(fp, "<I")
    return data

"""
    _read_spike_channel_header(fp)
Read the header for a spike channel
"""
def _read_spike_channel_header(fp):
    data = {}
    data_type = _read(fp, "<B")
    data_subtype = _read(fp, "<3B")
    if (data_type != PL2_HEADER_SPIKE_CHANNEL): # D5 06 08 05
        raise RuntimeError("Invalid type in spike channel header. Got ", data_type, " expected ", PL2_HEADER_SPIKE_CHANNEL)

    data["plex_channel"] = _read(fp, "<I")

    # Two more empty items
    _read(fp, "<I")
    _read(fp, "<I")

    data["name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source"] = _read(fp, "<I")
    data["channel"] = _read(fp, "<I")
    data["enabled"] = _read(fp, "<I")
    data["recording_enabled"] = _read(fp, "<I")
    data["units"] = bytearray(_read(fp, "<16B")).decode('ascii').split('\0', 1)[0]
    data["samples_per_second"] = _read(fp, "<d")
    data["coeff_to_convert_to_units"] = _read(fp, "<d")
    data["samples_per_spike"] = _read(fp, "<I")
    data["threshold"] = _read(fp, "<i")
    data["pre_threshold_samples"] = _read(fp, "<I")
    data["sort_enabled"] = _read(fp, "<I")
    data["sort_method"] = _read(fp, "<I")
    data["number_of_units"] = _read(fp, "<I")
    data["sort_range_start"] = _read(fp, "<I")
    data["sort_range_end"] = _read(fp, "<I")
    data["unit_counts"] = list(_read(fp, "<256Q"))
    data["source_trodality"] = _read(fp, "<I")
    data["trode"] = _read(fp, "<I")
    data["channel_in_trode"] = _read(fp, "<I")
    data["number_of_channels_in_source"] = _read(fp, "<I")
    data["device_id"] = _read(fp, "<I")
    data["number_of_channels_in_device"] = _read(fp, "<I")
    data["source_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_source_id"] = _read(fp, "<I")
    data["probe_device_channel"] = _read(fp, "<I")
    data["probe_device_channel"] = _read(fp, "<I")
    data["probe_device_id"] = _read(fp, "<I")
    data["input_voltage_minimum"] = _read(fp, "<d")
    data["input_voltage_maximum"] = _read(fp, "<d")
    data["total_gain"] = _read(fp, "<d")

    # Create empty vectors for our block offsets
    data["block_offsets"] = []
    data["block_num_items"] = []
    data["block_timestamps"] = []
    data["num_spikes"] = 0

    # Skip 128 bytes
    fp.seek(128, os.SEEK_CUR)

    return data

def _read_analog_channel_header(fp):
    data = {}

    data_type = _read(fp, "<B")
    data_subtype = _read(fp, "<3B")
    if (data_type != PL2_HEADER_ANALOG_CHANNEL): # D4 03 F8 00 or D4 04 F8 00 01, D4 07 F8 00
        raise RuntimeError("Invalid type in analog channel header. Got ", data_type, " expected ", PL2_HEADER_ANALOG_CHANNEL)
    data["plex_channel"] = _read(fp, "<I")

    # Two more empty items
    _read(fp, "<I")
    _read(fp, "<I")

    data["name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source"] = _read(fp, "<I")
    data["channel"] = _read(fp, "<I")
    data["enabled"] = _read(fp, "<I")
    data["recording_enabled"] = _read(fp, "<I")
    data["units"] = bytearray(_read(fp, "<16B")).decode('ascii').split('\0', 1)[0]
    data["samples_per_second"] = _read(fp, "<d")
    data["coeff_to_convert_to_units"] = _read(fp, "<d")
    data["source_trodality"] = _read(fp, "<I")
    data["trode"] = _read(fp, "<I")
    data["channel_in_trode"] = _read(fp, "<I")
    data["number_of_channels_in_source"] = _read(fp, "<I")
    data["device_id"] = _read(fp, "<I")
    data["number_of_channels_in_device"] = _read(fp, "<I")
    data["source_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_source_id"] = _read(fp, "<I")
    data["probe_source_channel"] = _read(fp, "<I")
    data["probe_device_id"] = _read(fp, "<I")
    data["probe_device_channel"] = _read(fp, "<I")
    data["input_voltage_minimum"] = _read(fp, "<d")
    data["input_voltage_maximum"] = _read(fp, "<d")
    data["total_gain"] = _read(fp, "<d")

    # Create empty vectors for our block offsets
    data["block_offsets"] = []
    data["block_num_items"] = []
    data["block_timestamps"] = []
    data["num_values"] = 0

    # Skip 128 bytes
    fp.seek(128, os.SEEK_CUR)

    return data

def _read_event_channel_header(fp):
    data = {}

    data_type = _read(fp, "<B")
    data_subtype = _read(fp, "<3B")
    if (data_type != PL2_HEADER_EVENT_CHANNEL): # D6
        raise RuntimeError("Invalid type in event channel header. Got ", data_type, " expected ", PL2_HEADER_EVENT_CHANNEL)
    data["plex_channel"] = _read(fp, "<I")

    # Two more empty items
    _read(fp, "<I")
    _read(fp, "<I")

    data["name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source"] = _read(fp, "<I")
    data["channel"] = _read(fp, "<I")
    data["enabled"] = _read(fp, "<I")
    data["recording_enabled"] = _read(fp, "<I")
    data["number_of_channels_in_source"] = _read(fp, "<I")
    data["number_of_channels_in_device"] = _read(fp, "<I")
    data["device_id"] = _read(fp, "<I")
    data["num_events"] = _read(fp, "<I") # TODO - this is not right
    data["source_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]

    # Create empty vectors for our block offsets
    data["block_offsets"] = []
    data["block_num_items"] = []
    data["block_timestamps"] = []
    data["num_events"] = 0

    # Skip 128 bytes
    fp.seek(128, os.SEEK_CUR)

    return data

"""
    _get_channel_offset(data_type)
Returns the offset in the type of channel given the subtype of data.
"""
def _get_channel_offset(data, data_type, channel_number):

    def get_channel_string(chan, search_data, search_string):
        for xr in range(0, len(search_data)):
            if search_data[xr]["name"][0:len(search_string)] == search_string and search_data[xr]["name"][len(search_string)].isnumeric():
                return str(chan).zfill(len(search_data[xr]["name"][len(search_string):]))
        return None

    offset = 0
    if data_type == PL2_ANALOG_TYPE_WB:
        channel = get_channel_string(channel_number, data["analog_channels"], "WB")
        offset = next(x for x in range(len(data["analog_channels"])) if data["analog_channels"][x]["name"]==("WB" + channel))
    elif data_type == PL2_ANALOG_TYPE_AI or data_type == PL2_ANALOG_TYPE_AI2:
        channel = get_channel_string(channel_number, data["analog_channels"], "AI")
        offset = next(x for x in range(len(data["analog_channels"])) if data["analog_channels"][x]["name"]==("AI" + channel))
    elif data_type == PL2_ANALOG_TYPE_FP:
        channel = get_channel_string(channel_number, data["analog_channels"], "FP")
        offset = next(x for x in range(len(data["analog_channels"])) if data["analog_channels"][x]["name"]==("FP" + channel))
    elif data_type == PL2_ANALOG_TYPE_SPKC:
        channel = get_channel_string(channel_number, data["analog_channels"], "SPKC")
        offset = next(x for x in range(len(data["analog_channels"])) if data["analog_channels"][x]["name"]==("SPKC" + channel))
    elif data_type == PL2_EVENT_TYPE_SINGLE_BIT:
        channel = get_channel_string(channel_number, data["event_channels"], "EVT")
        offset = next(x for x in range(len(data["event_channels"])) if data["event_channels"][x]["name"]==("EVT" + channel))
    elif data_type == PL2_EVENT_TYPE_STROBED:
        offset = next(x for x in range(len(data["event_channels"])) if data["event_channels"][x]["name"]=="Strobed")
    elif data_type == PL2_SPIKE_TYPE_SPK:
        channel = get_channel_string(channel_number, data["spike_channels"], "SPK")
        offset = next(x for x in range(len(data["spike_channels"])) if data["spike_channels"][x]["name"]==("SPK" + channel))
    elif data_type == PL2_SPIKE_TYPE_SPK_SPKC:
        channel = get_channel_string(channel_number, data["spike_channels"], "SPK_SPKC")
        offset = next(x for x in range(len(data["spike_channels"])) if data["spike_channels"][x]["name"]==("SPK_SPKC" + channel))
    elif data_type == 0x00:
        offset = 0
    else:
        warnings.warn("Unknown channel type provided: {:x}".format(data_type), Warning)
    return offset

def _read_footer(fp, data):
    fp.seek(data["internal_value_4"]) # Seek to start of footer

    while fp.tell() < data["file_length"]:
        data_type = _read(fp, "<B")
        data_subtype = _read(fp, "<B")

        # All items are stored as the following
        num_words = _read(fp, "<H")
        channel = _get_channel_offset(data, data_subtype, _read(fp, "<H")) # Python is base 0
        _read(fp, "<H") # Skipped

        # Determine how many items we have based on the number of words
        # Each element is stored as position ("<Q"), timestamp ("<Q"),
        # and number of elements ("<H")
        num_items = int(num_words * 2 / (8 + 8 + 2))
        num_values = _read(fp, "<Q")

        if (data_type == PL2_FOOTER_SPIKE_CHANNEL):
            data["spike_channels"][channel]["num_spikes"] = num_values
            if "block_offsets" not in data["spike_channels"][channel].keys():
                data["spike_channels"][channel]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
                data["spike_channels"][channel]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
                data["spike_channels"][channel]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
            else:
                data["spike_channels"][channel]["block_offsets"].extend(_read(fp, "<{:d}Q".format(num_items), True))
                data["spike_channels"][channel]["block_timestamps"].extend(_read(fp, "<{:d}Q".format(num_items), True))
                data["spike_channels"][channel]["block_num_items"].extend(_read(fp, "<{:d}H".format(num_items), True))
        elif (data_type == PL2_FOOTER_ANALOG_CHANNEL):
            data["analog_channels"][channel]["num_values"] = num_values
            if "block_offsets" not in data["analog_channels"][channel].keys():
                data["analog_channels"][channel]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
                data["analog_channels"][channel]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
                data["analog_channels"][channel]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
            else:
                data["analog_channels"][channel]["block_offsets"].extend(_read(fp, "<{:d}Q".format(num_items), True))
                data["analog_channels"][channel]["block_timestamps"].extend(_read(fp, "<{:d}Q".format(num_items), True))
                data["analog_channels"][channel]["block_num_items"].extend(_read(fp, "<{:d}H".format(num_items), True))
        elif (data_type == PL2_FOOTER_EVENT_CHANNEL):
            data["event_channels"][channel]["num_events"] = num_values
            if "block_offsets" not in data["event_channels"][channel].keys():
                data["event_channels"][channel]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
                data["event_channels"][channel]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
                data["event_channels"][channel]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
            else:
                data["event_channels"][channel]["block_offsets"].extend(_read(fp, "<{:d}Q".format(num_items), True))
                data["event_channels"][channel]["block_timestamps"].extend(_read(fp, "<{:d}Q".format(num_items), True))
                data["event_channels"][channel]["block_num_items"].extend(_read(fp, "<{:d}H".format(num_items), True))
        elif (data_type == PL2_FOOTER_START_STOP_CHANNEL):
            data["start_stop_channels"]["num_events"] = _read(fp, "<Q")
            data["start_stop_channels"]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["start_stop_channels"]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["start_stop_channels"]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
        else:
            raise RuntimeError("Unknown data type in footer at position ", position(fp) - 2, " Got value: 0x", hex(data_type))
        # Skip to next 16 byte aligned value
        fp.seek(int((fp.tell() + 15) / 16) * 16)


def _reconstruct_footer(fp, data):
    # Given a data file without a footer, attempt to reconstruct the footer by parsing
    # individual data records

    # Seek to the first data block and begin _reading
    fp.seek(data["internal_value_2"])

    if data["internal_value_3"] == 0:
        data["internal_value_3"] = data["internal_value_4"] # File is not complete

    while (fp.tell() < data["internal_value_3"]):
        # _read type
        block_offset = fp.tell()
        data_type = _read(fp, "<B")
        data_subtype = _read(fp, "<B")

        if (data_type == PL2_DATA_BLOCK_ANALOG_CHANNEL):
            num_items = _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H")) # Python is base 0
            _read(fp, "<H") # Unknown
            timestamp = _read(fp, "<Q") # Unknown

            data["analog_channels"][channel]["num_values"] += num_items
            data["analog_channels"][channel]["block_offsets"].append(block_offset)
            data["analog_channels"][channel]["block_timestamps"].append(timestamp)
            data["analog_channels"][channel]["block_num_items"].append(num_items)

            # Skip over the values
            fp.seek(2 * num_items, os.SEEK_CUR)
        elif (data_type == PL2_DATA_BLOCK_SPIKE_CHANNEL):
            _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H")) # Python is base 0
            num_sample_points = _read(fp, "<H")
            num_items = _read(fp, "<Q")

            # _read each of the items (64 byte values)
            timestamp = _read(fp, "<Q")

            data["spike_channels"][channel]["num_spikes"] += num_items
            data["spike_channels"][channel]["block_offsets"].append(block_offset)
            data["spike_channels"][channel]["block_timestamps"].append(timestamp)
            data["spike_channels"][channel]["block_num_items"].append(num_items)

            # Skip to next instance
            # Spikes: Int16 * * num_items * num_samples pints
            # Assignments: Unit16 * num_items
            # Timestamps = Uint64 * num_items (but we read one already)
            fp.seek(2 * num_items + 2 * num_sample_points * num_items + (num_items - 1) * 8, os.SEEK_CUR)
        elif (data_type == PL2_DATA_BLOCK_EVENT_CHANNEL):
            _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H")) # Python is base 0
            num_items = _read(fp, "<Q")
            _read(fp, "<H")
            timestamp = _read(fp, "<Q")

            data["event_channels"][channel]["num_events"] += num_items
            data["event_channels"][channel]["block_offsets"].append(block_offset)
            data["event_channels"][channel]["block_timestamps"].append(timestamp)
            data["event_channels"][channel]["block_num_items"].append(num_items)

            # Skip to next item
            fp.seek((num_items - 1) * 8 + num_items * 2, os.SEEK_CUR)

        elif (data_type == PL2_DATA_BLOCK_START_STOP_CHANNEL):
            # Start-stop Channel
            _read(fp, "<H")
            channel = _read(fp, "<H") - 1
            num_items = _read(fp, "<Q")
            timestamp = _read(fp, "<Q")

            data["start_stop_channels"]["num_events"] += num_items
            data["start_stop_channels"]["block_offsets"].append(block_offset)
            data["start_stop_channels"]["block_timestamps"].append(timestamp)
            data["start_stop_channels"]["block_num_items"].append(num_items)
            fp.seek((num_items - 1) * 8 + num_items * 2, os.SEEK_CUR)
        else:
            raise RuntimeError("Unknown data type at position ", position(fp) - 2, "Got value: ", data_type)
        # Align to next 16 byte boundary
        fp.seek(int((fp.tell() + 15) / 16) * 16)
