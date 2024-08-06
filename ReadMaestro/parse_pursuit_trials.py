"""
Copyright (c) 2024 Seth Egger

Written by Seth W. Egger <sethegger@gmail.com>

Defines pursuit object and subclasses for each experiment type.

Finds appropriate pursuit trials in a data set and collates the results.
"""

import struct
import io
import os
import sys
import numpy as np
import re
import math
import pickle

class pursuitDataObject(object):
    """
    Defines the object that collates pursuit data of a specific experiment type for a given dates data.
    """

    def __init__(self):
        self.name = []
        self.direction = []
        self.speed = []

    def setName(self,name):
        self.name = name