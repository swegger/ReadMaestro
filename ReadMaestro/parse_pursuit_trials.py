"""
Copyright (c) 2024 Seth Egger

Written by Seth W. Egger <sethegger@gmail.com>

Defines pursuit object and methods to extract data, identify saccades, 
and replace saccade velocities from pursuit traces with NaNs.

To do: Define methods to interpolate NaNs - 1. Classic linear interpolation; 2. Inference using a Gaussian process
To do: Define methods to shuffle saccade indices with respect to trial number so that interpolation performance can be measured

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
    Defines the object that collates pursuit data of a specific experiment type from a given date.
    """

    def __init__(self):
        self.name = []
        self.hvelocities = []
        self.directions = []
        self.speeds = []
        self.coherences = []
        self.perturbations = []
        self.saccades = []
        self.rotationApplied = False
        self.nansaccades = False
        self.vel_theta = []
        self.eye_t = []

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
        cohs = []
        perts = []
        vel_theta = []
        
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
            
            dirsTemp = None
            spdsTemp = None
            cohsTemp = None
            pertsTemp = None
            
            # Set condition values to true if conditions are not specified by user
            if not directions:
                directionCheck = True
            else:
                directionCheck = False
            if not speeds:               
                speedCheck = True
            else:
                speedCheck = False
            if not coherences:
                cohsCheck = True
            else:
                cohsCheck = False
            if not perturbations:
                pertsCheck = True
            else:
                pertsCheck = False


            pattern = re.compile('\d{3}')
            iterator = pattern.finditer(trial_str)
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
                elif trial_str[spans[0]-1] == 'h':
                    if not coherences or float(trial_str[spans[0]:spans[1]]) in coherences:
                        cohsTemp = float(trial_str[spans[0]:spans[1]])
                        cohsCheck = True
                    else:
                        cohsCheck = False
                elif trial_str[spans[0]-1] == 'p':
                    if not perturbations or float(trial[spans[0]:spans[1]]) in perturbations:
                        pertsTemp = float(trial_str[spans[0]:spans[1]])
                        pertsCheck = True
                    else:
                        pertsCheck = False


            if trialIDcheck and directionCheck and speedCheck and cohsCheck and pertsCheck:
                hvelocity[0:len(data[triali].ai_data[0]),acceptedIndex] = np.array(data[triali].ai_data[2])*0.09189
                vvelocity[0:len(data[triali].ai_data[0]),acceptedIndex] = np.array(data[triali].ai_data[3])*0.09189
                acceptedIndex = acceptedIndex+1
                dirs.append(dirsTemp)
                spds.append(spdsTemp)
                cohs.append(cohsTemp)
                perts.append(pertsTemp)
                vel_theta.append(data[triali].header.vel_theta)

        # Cleave remaining unfilled columns from data matrices
        hvelocity = hvelocity[:,:acceptedIndex]
        vvelocity = vvelocity[:,:acceptedIndex]

        # Detect saccades
        if hvelocity.size == 0:
            sacInds = np.full_like(hvelocity,False)
        else:
            sacInds = self.saccadeDetect(hvelocity,vvelocity)

        self.hvelocities = hvelocity
        self.vvelocities = vvelocity
        self.directions = dirs
        self.speeds = spds
        self.coherences = cohs
        self.perturbations = perts
        self.saccades = sacInds
        self.vel_theta = vel_theta
        self.eye_t = np.arange(0,len(hvelocity))

        return hvelocity, vvelocity, dirs, spds, cohs, perts, sacInds
    
    def applyRotationToData(self):
        self.hvelocities, self.vvelocities = self.rotateData(self.hvelocities,self.vvelocities,np.add(self.directions,self.vel_theta))
        self.rotationApplied = True

    def setSaccadeVelocitiesToNaN(self):
        self.hvelocities[self.saccades] = np.nan
        self.vvelocities[self.saccades] = np.nan
        self.nansaccades = True
        
    
    def saccadeDetect(hv,vv,accelerationThreshold=1.1,windowSize=60):
        '''
        Simple method to detect saccadic eye movements from eye velocity data and return the indices of putative saccades
        '''
        ha = np.diff(hv,n=1,axis=0,prepend=0)
        va = np.diff(vv,n=1,axis=0,prepend=0)

        inds = np.add(np.abs(ha) > accelerationThreshold, np.abs(va) > accelerationThreshold)
        filt = np.ones((windowSize,))/windowSize
        sinds = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=0, arr=inds)
        sacInds = np.full(ha.shape, False, dtype=bool)
        sacInds[sinds>0] = True

        return sacInds
    
    def rotateData(x,y,thetas):
        '''
        Method to rotate cartesian data to a common frame based on 
        '''
        xnew = x*np.cos(np.deg2rad(thetas)) + y*np.sin(np.deg2rad(thetas))
        ynew = x*np.sin(np.deg2rad(thetas)) - y*np.cos(np.deg2rad(thetas))

        return xnew, ynew
    
    def computeMeanCov(x,axis=-1):
        '''
        Method for computing the mean and covariance while ignoring NaN data
        '''
        C = np.ma.cov(np.ma.masked_invalid(x.T), rowvar=False)
        mu = np.nanmean(x,axis=axis)

        return mu, C
    
    def linearInterpolation(x):
        return x
    
    def conditionalGaussian(mu,Sig,f,nearSPD=True,x_indices=[]):
        
        # Set up variables
        if not x_indices.any():
            x_indices = np.zeros_like(mu, dtype=np.bool)
            x_indices[:len(f)] = True     # Assume observations are of first q elements of length N vector mu


        N = len(mu)
        q = sum(x_indices)

        # Reshape mean and covariance for partitioning
        indsOrig = np.arange(N)
        indsNew = np.concatenate((indsOrig[np.logical_not(x_indices)], indsOrig[x_indices]))
        Inew, Jnew = np.meshgrid(indsNew, indsNew)
        SigNew = np.empty_like(Sig)
        for i in range(N):
            for j in range(N):
                SigNew[i,j] = Sig[Inew[i,j],Jnew[i,j]]     # Now covariance matrix is of form SigNew = [SigUnUn, SigUnObs; SigObsUn, SigObsObs]

        # Partition mean and covariance
        muObs = mu[x_indices]
        muUn = mu[np.logical_not(x_indices)]

        SigUnUn = SigNew[0:N-q,0:N-q]
        SigUnObs = SigNew[0:N-q,N-q:]
        SigObsObs = SigNew[N-q:,N-q:]
        SigObsUn = SigNew[N-q:,0:N-q]

        temp = SigUnObs @ np.linalg.inv(SigObsObs)
        print(f.shape)
        print(muObs.shape)
        mu_ = muUn + np.dot(temp, f-muObs)
        Sig_ = SigUnUn - temp  @ SigObsUn

        # Local definitions of functions for computing the nearest positive semi-definate covariance matrix
        # from Higham (2000) (see https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix)
        def _getAplus(A):
            eigval, eigvec = np.linalg.eig(A)
            Q = np.matrix(eigvec)
            xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
            return Q*xdiag*Q.T

        def _getPs(A, W=None):
            W05 = np.matrix(W**.5)
            return  W05.I * _getAplus(W05 * A * W05) * W05.I

        def _getPu(A, W=None):
            Aret = np.array(A.copy())
            Aret[W > 0] = np.array(W)[W > 0]
            return np.matrix(Aret)

        def nearPD(A, nit=10):
            n = A.shape[0]
            W = np.identity(n) 
            # W is the matrix used for the norm (assumed to be Identity matrix here)
            # the algorithm should work for any diagonal W
            deltaS = 0
            Yk = A.copy()
            for k in range(nit):
                Rk = Yk - deltaS
                Xk = _getPs(Rk, W=W)
                deltaS = Xk - Rk
                Yk = _getPu(Xk, W=W)
            return Yk
        
        if nearSPD and np.logical_not(np.any(np.isnan(Sig_))):
            Sig_ = nearPD(Sig_)

        return mu_, Sig_