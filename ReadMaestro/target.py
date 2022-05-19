import numpy as np



def compress_target_data(maestro_data):
    """
    """
    for trial in maestro_data:
        try:
            if trial['compressed_target']:
                continue
        except KeyError:
            xpos = []
            ypos = []
            xvel = []
            yvel = []
            # Get compressed data for each target
            for targ in range(0, len(trial['targets'])):
                xpos.append(compress_data(trial['horizontal_target_position'][targ, :]))
                ypos.append(compress_data(trial['vertical_target_position'][targ, :]))
                xvel.append(compress_data(trial['horizontal_target_velocity'][targ, :]))
                yvel.append(compress_data(trial['vertical_target_velocity'][targ, :]))
            # Overwrite existing data with compressed data
            trial['horizontal_target_position'] = xpos
            trial['vertical_target_position'] = ypos
            trial['horizontal_target_velocity'] = xvel
            trial['vertical_target_velocity'] = yvel
            # Add flag indicating that data are compressed
            trial['compressed_target'] = True

    return None


def compress_data(data_vector):
    """ This does a very simple compression based on the fact that commanded
    target velocity (and often position) rarely change and so there is no
    need to store all the data points. This compression will only save storage
    if over 2/3 of the data are redundant with their previous data point. """
    if len(data_vector) == 1:
        return [(0, 1, data_vector[0])]
    last_v = data_vector[0]
    start_x = 0
    compressed_v = []
    # NOTE: need to start iteration at 1 since 0 is already determined and this
    # gives output as slice indices for decompress_data.
    for x_ind in range(1, len(data_vector)):
        curr_v = data_vector[x_ind]
        # Need special handling of np.nan
        if np.isnan(curr_v) or np.isnan(last_v):
            if ( (np.isnan(last_v) and ~np.isnan(curr_v))
                or (~np.isnan(last_v) and np.isnan(curr_v)) ):
                # Nan transition point
                compressed_v.append((start_x, x_ind, last_v))
                last_v = curr_v
                start_x = x_ind
        elif (curr_v != last_v):
            compressed_v.append((start_x, x_ind, last_v))
            last_v = curr_v
            start_x = x_ind
        if (x_ind == len(data_vector) - 1):
            # Need to log the end of data whether it changes or not so that we
            # can assing final values by slice in decompress_data
            compressed_v.append((start_x, x_ind+1, last_v))

    return compressed_v


def decompress_data(compressed_v):
    """ Undoes the simple compression algorithm to return original target
    data vector. """
    v_out = np.zeros(compressed_v[-1][1])
    for start, stop, v in compressed_v:
        v_out[start:stop] = v

    return v_out


class MaestroTarget(object):
    """

    Primary benefit of this object is that it can handle computing the commanded
    and actual velocity (derivative of actual target position) on the fly. These
    should be requested with calls to:
        get_data(data_name) where data_name specifies the target property and is
        one of:
            xpos, ypos, xvel_comm, yvel_comm, xvel, yvel
    Data are reconstructed in sync with the screen refresh rate.
    Data in the maestro_trial 'horizontal_target_velocity' and
    'vertical_target_velocity' are assumed to be the commanded velocity as
    output by maestro_read module.
    """

    def __init__(self, maestro_trial, target_num):
        # seems like for some reason this changed with version?
        if maestro_trial['header']['version'] < 21:
            self.frame_refresh_time = 1000.0 / maestro_trial['header']['display_framerate']
        else:
            self.frame_refresh_time = 1000 * (1000.0 / maestro_trial['header']['display_framerate'])

        # Check if data are already compressed
        try:
            if maestro_trial['compressed_target']:
                self.n_time_points = maestro_trial['horizontal_target_position'][0][-1][1]
                self.horizontal_target_position = maestro_trial['horizontal_target_position'][target_num]
                self.vertical_target_position = maestro_trial['vertical_target_position'][target_num]
                self.horizontal_target_velocity_comm = maestro_trial['horizontal_target_velocity'][target_num]
                self.vertical_target_velocity_comm = maestro_trial['vertical_target_velocity'][target_num]
            else:
                # Has compressed key but not compressed so move to except statement
                raise KeyError("Dummy key error")
        except KeyError:
            self.n_time_points = maestro_trial['horizontal_target_position'].shape[1]
            # Compute then store compressed versions of target data
            self.horizontal_target_position = compress_data(maestro_trial['horizontal_target_position'][target_num, :])
            self.vertical_target_position = compress_data(maestro_trial['vertical_target_position'][target_num, :])
            self.horizontal_target_velocity_comm = compress_data(maestro_trial['horizontal_target_velocity'][target_num, :])
            self.vertical_target_velocity_comm = compress_data(maestro_trial['vertical_target_velocity'][target_num, :])

        # Set valid data keys for referencing target data
        self.__valid_keys = ['horizontal_target_position', 'xpos',
                             'vertical_target_position', 'ypos',
                             'horizontal_target_velocity_comm', 'xvel_comm',
                             'vertical_target_velocity_comm', 'yvel_comm',
                             'horizontal_target_velocity', 'xvel',
                             'vertical_target_velocity', 'yvel']
        self.__myiterator__ = iter(self.__valid_keys)

    def get_next_refresh(self, from_time, n_forward=1):
        n_found = 0
        from_time = np.floor(from_time).astype('int')
        for t in range(from_time, self.n_time_points-1, 1):
            if (t % self.frame_refresh_time < 1) and (t >= from_time):
                n_found += 1
                if n_found == n_forward:
                    return t
        return None

    def get_last_refresh(self, from_time, n_back=1):
        n_found = 0
        from_time = np.ceil(from_time).astype('int')
        for t in range(from_time, -1, -1):
            if (t % self.frame_refresh_time < 1) and (t <= from_time):
                n_found += 1
                if n_found == n_back:
                    return t
        return None

    def get_next_position_threshold(self, time, axis=None):
        pass

    def get_next_velocity_threshold(self, time, axis=None, velocity_data='commanded'):
        pass

    def get_next_acceleration_threshold(self, time, axis=None):
        pass

    def _velocity_from_position(self, axis, remove_transients=True):
        # Always computed over all time points otherwise this gets super confusing in multiple stages
        transient_index = [[], []]
        if axis == 'horizontal':
            position = decompress_data(self.horizontal_target_position)
            vel_comm = decompress_data(self.horizontal_target_velocity_comm)
        elif axis == 'vertical':
            position = decompress_data(self.vertical_target_position)
            vel_comm = decompress_data(self.vertical_target_velocity_comm)
        else:
            raise ValueError("Axis for velocity must be specified as 'horizontal' or 'vertical'.")
        calculated_velocity = np.zeros(len(position))

        last_x = 0
        next_x = self.get_next_refresh(1)
        while next_x is not None:
            # Just set calculated velocity as the average of the position change over the refresh rate
            calculated_velocity[last_x:next_x] = (position[next_x] - position[last_x]) / (next_x - last_x)

            if ( (np.isnan(vel_comm[next_x-1])
                    or vel_comm[next_x-1] == 0)
                    and vel_comm[next_x] != 0 ):
                transient_index[0].append(last_x)
                transient_index[1].append(next_x)

            if ( (np.isnan(vel_comm[next_x])
                    or vel_comm[next_x] == 0)
                    and vel_comm[next_x-1] != 0 ):
                transient_index[0].append(last_x)
                transient_index[1].append(next_x)

            last_x = next_x
            next_x = self.get_next_refresh(next_x + 1)

        if last_x < self.n_time_points:
            # No more refreshes to end of data so just hold velocity steady
            calculated_velocity[last_x:] = calculated_velocity[last_x-1]

        calculated_velocity = 1000 * calculated_velocity # Convert to deg/s
        calculated_velocity[-1] = calculated_velocity[-2]
        if remove_transients:
            for window in range(0, len(transient_index[0])):
                calculated_velocity[transient_index[0][window]:transient_index[1][window]] = vel_comm[transient_index[0][window]:transient_index[1][window]]
        calculated_velocity[np.isnan(calculated_velocity)] = 0

        return calculated_velocity

    def acceleration_from_velocity(self, velocity_data='commanded'):
        pass

    def readcxdata_velocity(self):
        pass

    def get_data(self, data_name):
        """ x must be a continuous set of indices into the data for this to work
        but this is not checked. """
        if data_name in ['horizontal_target_position', 'xpos']:
            return decompress_data(self.horizontal_target_position)
        elif data_name in ['vertical_target_position', 'ypos']:
            return decompress_data(self.vertical_target_position)
        elif data_name in ['horizontal_target_velocity_comm', 'xvel_comm']:
            return decompress_data(self.horizontal_target_velocity_comm)
        elif data_name in ['vertical_target_velocity_comm', 'yvel_comm']:
            return decompress_data(self.vertical_target_velocity_comm)
        elif data_name in ['horizontal_target_velocity', 'xvel']:
            return self._velocity_from_position('horizontal')
        elif data_name in ['vertical_target_velocity', 'yvel']:
            return self._velocity_from_position('vertical')
        else:
            raise ValueError("MaestroTarget object has no data '{0}'.".format(data_name))

    def __getitem__(self, item):
        return self.get_data(item)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = next(self.__myiterator__)
        # Do not return 'hidden' attributes/keys
        while next_item[0:1] == "_":
            next_item = next(self.__myiterator__)
        return next_item

    def __iter__(self):
        return iter(self.__valid_keys)

    def keys(self):
        return self.__valid_keys

    def __len__(self):
        return self.n_time_points
