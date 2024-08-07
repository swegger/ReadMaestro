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

def pursuitData(data,trialIDs=[],directions=[],speeds=[],coherences=[],perturbations=[]):
    """
    Finds horizontal and vertical eye velocities and assigns each
    to their corresponding numpy array.
    """

    # Preliminary information and preallocation
    trialN = len(data)
    dataPointsMax = 0
    for triali in range(trialN):
        dataPointsMax = max(dataPointsMax,len(data[triali].ai_data[0]))


    hvelocity = np.empty((dataPointsMax,trialN,))
    hvelocity[:] = np.nan

    vvelocity = np.empty((dataPointsMax,trialN,))
    vvelocity[:] = np.nan

    # Iterate through trials and insert eye velocities from matching experiment IDs and trials into array
    acceptedIndex = 0
    dirs = []
    spds = []
    for triali in range(trialN):
        trial_str = data[triali].header.trial_name

        if not trialIDs:
            trialIDcheck = True
        else:
            trialIDcheck = False
            for id in trialIDs:
                if trial_str.__contains__(id):
                    trialIDcheck = True
                else:
                    trialIDcheck = False
                if trialIDcheck:
                    break
        
        pattern = re.compile('\d{3}')
        iterator = pattern.finditer(trial_str)
        a = []
        for match in iterator:
            spans = match.span()
            if trial_str[spans[0]-1] == 'd':
                if not directions or float(trial_str[spans[0]:spans[1]]) in directions:
                    dirsTemp = float(trial_str[spans[0]:spans[1]])
                    directionCheck = True
                else:
                    directionCheck = False
            elif trial_str[spans[0]-1] == 's':
                if not speeds or float(trial_str[spans[0]:spans[1]]) in speeds:
                    spdsTemp = float(trial_str[spans[0]:spans[1]])
                    speedCheck = True
                else:
                    speedCheck = False

        if trialIDcheck and directionCheck and speedCheck:
            hvelocity[0:len(data[triali].ai_data[0]),acceptedIndex] = np.array(data[triali].ai_data[2])*0.09189
            vvelocity[0:len(data[triali].ai_data[0]),acceptedIndex] = np.array(data[triali].ai_data[3])*0.09189
            acceptedIndex = acceptedIndex+1
            dirs.append(dirsTemp)
            spds.append(spdsTemp)

    # Cleave remaining unfilled columns from data matrices
    hvelocity = hvelocity[:,:acceptedIndex]
    vvelocity = vvelocity[:,:acceptedIndex]

    return hvelocity, vvelocity, dirs, spds

        

        

class pursuitDataObject(object):
    """
    Defines the object that collates pursuit data of a specific experiment type for a given dates data.
    """

    def __init__(self):
        self.name = []
        self.hvelocities = []
        self.directions = []
        self.speeds = []
        self.coherences = []

    def setName(self,data):
        self.name = data[0].file_name[0:-17:]

    def pursuitData(self,data,trialIDs=[],directions=[],speeds=[],coherences=[],perturbations=[]):
        """
        Finds horizontal and vertical eye velocities and assigns each
        to their corresponding numpy array.
        """

        # Preliminary information and preallocation
        trialN = len(data)
        dataPointsMax = 0
        for triali in range(trialN):
            dataPointsMax = max(dataPointsMax,len(data[triali].ai_data[0]))


        hvelocity = np.empty((dataPointsMax,trialN,))
        hvelocity[:] = np.nan

        vvelocity = np.empty((dataPointsMax,trialN,))
        vvelocity[:] = np.nan

        # Iterate through trials and insert eye velocities from matching experiment IDs and trials into array
        acceptedIndex = 0
        dirs = []
        spds = []
        for triali in range(trialN):
            trial_str = data[triali].header.trial_name

            if not trialIDs:
                trialIDcheck = True
            else:
                trialIDcheck = False
                for id in trialIDs:
                    if trial_str.__contains__(id):
                        trialIDcheck = True
                    else:
                        trialIDcheck = False
                    if trialIDcheck:
                        break
            
            pattern = re.compile('\d{3}')
            iterator = pattern.finditer(trial_str)
            a = []
            for match in iterator:
                spans = match.span()
                if trial_str[spans[0]-1] == 'd':
                    if not directions or float(trial_str[spans[0]:spans[1]]) in directions:
                        dirsTemp = float(trial_str[spans[0]:spans[1]])
                        directionCheck = True
                    else:
                        directionCheck = False
                elif trial_str[spans[0]-1] == 's':
                    if not speeds or float(trial_str[spans[0]:spans[1]]) in speeds:
                        spdsTemp = float(trial_str[spans[0]:spans[1]])
                        speedCheck = True
                    else:
                        speedCheck = False

            if trialIDcheck and directionCheck and speedCheck:
                hvelocity[0:len(data[triali].ai_data[0]),acceptedIndex] = np.array(data[triali].ai_data[2])*0.09189
                vvelocity[0:len(data[triali].ai_data[0]),acceptedIndex] = np.array(data[triali].ai_data[3])*0.09189
                acceptedIndex = acceptedIndex+1
                dirs.append(dirsTemp)
                spds.append(spdsTemp)

        # Cleave remaining unfilled columns from data matrices
        hvelocity = hvelocity[:,:acceptedIndex]
        vvelocity = vvelocity[:,:acceptedIndex]

        self.hvelocities = hvelocity
        self.vvelocities = vvelocity
        self.directions = dirs
        self.speeds = spds

        return hvelocity, vvelocity, dirs, spds