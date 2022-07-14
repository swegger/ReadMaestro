"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import numpy as np
import sys # Required for floating point epsilon



class Timeseries(object):
    """A class which encapsulates a timeseries.

    This class is useful to ensapsulate a timeseries with a a set
    sampling rate. That is, rather than storing the entire timeseries,
    we store only the start, stop, and delta_t which defines the timeseries.
    It is important that any timeseries which reference this timeseries
    correctly interpolated to the timeseries associated with this timeseries.

    I still think unexpected behavior and rounding errors could occur if start,
    stop, and dt are not all integers. So this may be worth requiring at some
    point in the future, or making the underlying functions all operate on
    integers.
    """

    def __init__(self, start, stop, dt):
        """Initialize a new timeseries.

        Initializes a new timeseries object. This function takes the first point,
        the final point, and the delta_t which defines the timeseries.
        :param start The time at t[0]
        :param stop The time at t[-1] (i.e., the last timepoint). This point is
        included in the timeseries (goes from star to stop, inclusive).
        :param dt The change in time associated with each timestep.
        """
        self.start = float(start)
        self.stop = float(stop)
        self.dt = float(dt)
        if self.start > self.stop:
            raise RuntimeError('Endpoint of timeseries is greater than start point')
        self.n = int(np.round((stop - start) / dt))
        self.stop = self.start + self.dt * self.n

    def timeseries(self):
        """Returns the complete timeseries associated with this element"""
        return np.arange(self.start, self.stop - self.dt/2, self.dt)

    def indices(self):
        """Returns the indices associated with this timeseries

        The indices are a simple array going from 0 to N-1 (where N is the
        length of the timeseries)
        """
        return np.arange(0, self.n)

    def find_index(self, value):
        """Finds the index associated with a certain value

        Note that this function merely returns the "closest" index to the
        given value. It is not guaranteed that the value is actually represented
        in the timeseries. If the value is not found, None is returned
        """
        if value < self.start or value > self.stop:
            raise IndexError("Index for value {0} is out of timeseries range.".format(value))
        index = np.int32(round((value - self.start) / self.dt))
        return index

    def find_indices(self, values):
        """Returns the indices for a numpy array or list of values. This is
        just a convenience function that iteratively calls "find_index". If
        a consecutive range of indices are desired, using "find_index_range"
        should be much faster. """
        if isinstance(values, int) or isinstance(values, np.integer):
            # Only scalar integer input
            values = [values]
        elif isinstance(values, slice):
            step = 1 if values.step is None else values.step
            start = self.start if values.start is None else values.start
            if step < 0:
                stop = self.start if values.stop is None else values.stop
            else:
                stop = self.stop if values.stop is None else values.stop
            return self.find_index_range(start, stop, step)
        indices = np.zeros(len(values), dtype=np.int32)
        for ind, v in enumerate(values):
            indices[ind] = self.find_index(v)
        return indices

    def find_index_range(self, start, stop, step=1):
        """Returns a range of indices corresponding to a time window. """
        if step < 0:
            return self.__find_index_range_rev(start, stop, step)
        if start > stop:
            raise IndexError("Start value must be <= stop value if positive step size is used.")
        t1 = self.find_index(start)
        t2 = self.find_index(stop)
        return np.arange(t1, t2, step, dtype=np.int32)

    def __find_index_range_rev(self, start, stop, step=-1):
        """Returns a range of indices corresponding to a time window with a
        negative step size, from stop to start. """
        if step > 0:
            return self.find_index_range(start, stop, step)
        if stop > start:
            raise IndexError("Start value must be < stop value if negative step size is used.")
        t1 = self.find_index(start)
        t2 = self.find_index(stop) - 1 # Minus 1 for slicing through last value
        return np.arange(t1, t2, step, dtype=np.int32)

    def valid_index_range(self, start, stop):
        """Returns a vector of the indices that are present in the timeseries
        within the requested window. Also returns a boolean array indicating
        the indices of the valid values in an array with size corresponding to
        the requested start/stop time window. This can allow consistent indexing
        of outputs at an identical shape among multiple timeseries with
        different lengths. """
        # Convert start and stop to usable multiples of timeseries
        start = int(round((start) / self.dt))
        stop = int(round((stop) / self.dt))
        # Force to find what the array size should be if all values were present in timeseries
        pseudo_values = np.arange(start, stop+self.dt, self.dt)
        out_inds = np.zeros((stop - start), dtype='bool')
        if start > stop:
            raise IndexError("Start value must be <= stop value.")
        if (start >= self.stop) or (stop <= self.start):
            # Requested range is totally beyond the timeseries
            return np.empty(0, dtype=np.int32), out_inds
        if start <= self.start:
            start = self.start
        if stop >= self.stop:
            stop = self.stop
        t1 = self.find_index(start)
        t2 = self.find_index(stop)
        valid_inds = np.arange(t1, t2, dtype=np.int32)
        out_start = np.argmax(pseudo_values >= start)
        if stop >= pseudo_values[-1]:
            out_stop = out_inds.shape[0]
        else:
            # Use strictly greater than so slicing indices are valid
            out_stop = np.argmax(pseudo_values >= stop)
        out_inds[out_start:out_stop] = True
        if (out_stop - out_start) != len(valid_inds):
            print(out_stop - out_start, len(valid_inds))
            print(pseudo_values[0], pseudo_values[-1], start, stop, out_start, out_stop, t1, t2, self.start, self.stop)
            raise RuntimeError("Number of out inds indices does not match valid inds. Could be a rounding error or coding error.")

        return valid_inds, out_inds

    def __len__(self):
        """Override the length of this timeseries"""
        return self.n

    def __getitem__(self, index):
        """Overrides the [] operator for this timeseries"""
        if isinstance(index, slice):
            raise ValueError("Slice of timeseries not currently defined.")
            # start = 0 if index.start is None else index.start
            # stop = self.n if index.stop is None else index.stop
            # step = 1 if index.step is None else index.step
            # return Timeseries(start * self.dt + self.start,
            #                   stop * self.dt + self.start,
            #                   self.dt * step)
        elif isinstance(index, list) or isinstance(index, tuple) or \
            (isinstance(index, np.ndarray) and index.ndim == 1):
            return [self[int(i)] for i in index]
        else:
            if index > self.n or index < -self.n:
                raise IndexError
            if index >= 0:
                return int(index) * self.dt + self.start
            else:
                return self.stop + self.dt * (int(index) + 1)

    def __add__(self, y):
        """Adds an offset to the timeseries"""
        self.start += y
        self.stop += y
        return self

    def __sub__(self, y):
        """Subtracts an offset from the timeseries"""
        self.start -= y
        self.stop -= y
        return self

    def __repr__(self):
        """Defines the printing function for this class"""
        return "<Timeseries ({:f}:{:f}:{:f}, length: {:d})>".format(self.start, self.dt, self.stop, self.n)

    def __mul__(self, m):
        """Defines multiplication of a timeseries object"""
        self.start *= m
        self.stop *= m
        self.dt *= m
        return self

    def __truediv__(self, d):
        """Floating point division"""
        self.start /= d
        self.stop /= d
        self.dt /= d
        return self

    def __lt__(self, f):
        """Less than operator"""
        return self.timeseries() < f

    def __le__(self, f):
        """Less than or equal operator"""
        return self.timeseries() <= f

    def __gt__(self, f):
        """Greater than operator"""
        return self.timeseries() > f

    def __ge__(self, f):
        """Greater than or equal operator"""
        return self.timeseries() >= f

    def __array__(self):
        """Converts the timeseries into a numpy array"""
        return self.timeseries()

    @classmethod
    def is_regular(cls, timeseries, eps=0.10):
        """Given a list of times, determine if the times are "regular"

        Given a list of times, this function determines if the times
        can be accurately stored in a timeseries object. That is,
        it determines if there is a regular delta_t thoughout the timeseries.
        Given floating point error, we use an epilson of 10% of the average
        delta_t to determine if the timeseries is regular. If the timeseries
        is not regular, then the original timeseries must be interpolated
        before storage in a timeseries object.
        :param timeseries: The original series of timepoints (np array)
        :param eps: A fractional percentage of the average delta_t to
        account for floating point errors
        """
        mean_dt = np.mean(np.diff(timeseries))

        # Ensure that all the diffs are less than the epsilon
        if np.any(np.abs(np.diff(timeseries) - mean_dt) > mean_dt * eps):
            return False # Not a valid timeseries
        return True

    @classmethod
    def convert_to_timeseries(cls, t):
        """Convert a floating point timeseries to a timeseries object"""
        if isinstance(t, Timeseries):
            return t
        dt = np.mean(np.diff(t))
        return Timeseries(t[0], t[0] + len(t) * dt, dt)
