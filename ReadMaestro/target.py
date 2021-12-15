import numpy as np




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
