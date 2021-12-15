import numpy as np



def compress_data(data_vector):
    """ This does a very simple compression based on the fact that commanded
    target velocity (and often position) rarely change and so there is no
    need to store all the data points. """
    last_v = 0
    start_x = 0
    compressed_v = []
    for x_ind in range(0, len(data_vector)):
        curr_v = data_vector[x_ind]
        if np.isnan(curr_v):
            continue
        if (curr_v != last_v):
            compressed_v.append((start_x, x_ind, last_v))
            last_v = curr_v
            start_x = x_ind
        elif (x_ind == len(data_vector) - 1):
            compressed_v.append((start_x, x_ind+1, last_v))
            last_v = curr_v
            start_x = x_ind
    return compressed_v


def decompress_v(x, compressed_v):
    """ Undoes the simple compression algorithm to return original target
    data vector. It is assumed that the values of x represent a series of
    consecutive indices into the original data. If this is violated the
    output behavior is undefined. """
    if x[-1] >= compressed_v[-1][1]:
        raise ValueError("Index 'x' has value greater than length of data ({0} vs. {1}).".format(x, compressed_v[-1][-1]))
    if x[0] < 0:
        raise ValueError("Index 'x' must be >= 0.")
    try:
        v_out = np.zeros(len(x))
        for t_ind, x in enumerate(x):
            for start, stop, v in compressed_v:
                if (x >= start) and (x < stop):
                    v_out[t_ind] = v
                    break
        return v_out
    except TypeError:
        # TypeError is thrown if x does not have len or is not iterable,
        # so assume it's a single value
        for start, stop, v in compressed_v:
            if (x >= start) and (x < stop):
                return v


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

        self.frame_refresh_time = 1000 * (1000.0 / maestro_trial['header']['display_framerate'])
        self.n_time_points = maestro_trial['horizontal_target_position'].shape[1]

        # Store compressed versions of target data
        self.horizontal_target_position = compress_data(maestro_trial['horizontal_target_position'][target_num, :])
        self.vertical_target_position = compress_data(maestro_trial['vertical_target_position'][target_num, :])
        self.horizontal_target_velocity_comm = compress_data(maestro_trial['horizontal_target_velocity'][target_num, :])
        self.vertical_target_velocity_comm = compress_data(maestro_trial['vertical_target_velocity'][target_num, :])

    def get_next_refresh(self, from_time, n_forward=1):
        n_found = 0
        from_time = np.ceil(from_time).astype('int')
        for t in range(from_time, self.n_time_points-1, 1):
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

    def get_next_position_threshold(self, time, axis=None):
        pass

    def get_next_velocity_threshold(self, time, axis=None, velocity_data='commanded'):
        pass

    def get_next_acceleration_threshold(self, time, axis=None):
        pass

    def _velocity_from_position(self, axis, remove_transients=True):
        # Always computed over all time points otherwise this gets super confusing in multiple stages
        transient_index = [[], []]
        x_for_pos = np.arange(0, self.n_time_points)
        if axis == 'horizontal':
            position = decompress_v(x_for_pos, self.horizontal_target_position)
            vel_comm = decompress_v(x_for_pos, self.horizontal_target_velocity_comm)
        elif axis == 'vertical':
            position = decompress_v(x_for_pos, self.vertical_target_position)
            vel_comm = decompress_v(x_for_pos, self.vertical_target_velocity_comm)
        else:
            raise ValueError("Axis for velocity must be specified as 'horizontal' or 'vertical'.")
        calculated_velocity = np.zeros(len(x_for_pos))

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

    def get_data(self, data_name, x=None):
        """ x must be a continuous set of indices into the data for this to work
        but this is not checked. """
        if x == None:
            x = np.arange(0, self.n_time_points)
        if data_name in ['horizontal_target_position', 'xpos']:
            return decompress_v(x, self.horizontal_target_position)
        elif data_name in ['vertical_target_position', 'ypos']:
            return decompress_v(x, self.vertical_target_position)
        elif data_name in ['horizontal_target_velocity_comm', 'xvel_comm']:
            return decompress_v(x, self.horizontal_target_velocity_comm)
        elif data_name in ['vertical_target_velocity_comm', 'yvel_comm']:
            return decompress_v(x, self.vertical_target_velocity_comm)
        elif data_name in ['horizontal_target_velocity', 'xvel']:
            return self._velocity_from_position('horizontal')[x]
        elif data_name in ['vertical_target_velocity', 'yvel']:
            return self._velocity_from_position('vertical')[x]
        else:
            raise


    # def __getattr__(self, attr):
    #     print(attr)
    #
    # def __getattribute__(self, attr):
    #     print(attr, "what the f")
