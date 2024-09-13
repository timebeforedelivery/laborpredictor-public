#!/usr/bin/env python3

"""
Methods that can hide the data to look at model hallucination.
"""

import argparse
import numpy as np

def loadData(filename : str):
    """
    Read the data from the file and convert it into a vector of [days, 5minindex]
    """
    data = []
    with open(filename, 'r')  as f:
        for line in f.readlines():
            values = line.split(',')
            values = np.array(values, dtype='float64')
            data.append(values)


    data = np.array(data)

    return data

def flatLine(data, neutral=0.0):
    """
    Replaces the complete data with a neutral value.
    """
    temp = np.empty(data.shape)
    temp.fill(neutral)
    return temp

def flatLineDays(data, days, neutral=0.0):
    """
    Only fill certain days with the neutral values.

    Arguments:
    days - An array indicating which days relative to labor that would need
        to be filled with 0 indicating the last day in the data set and
        data.shape[0] - 1 indicating the first day in the dataset.
    """
    _data = data.copy()
    l = _data.shape[0]

    for d in days:
        _data[l-d-1].fill(neutral)

    return _data


def addGaussianNoise(data, mean=0, std=1):
    """
    Adds gausian noise to the whole dataset.
    """
    noise = np.random.normal(loc=mean, scale=std, size=data.shape)
    return data + noise

def addGaussianNoiseDays(data, days, mean=0, std=1):
    """
    Adds random gaussian noise but only for specific days.
    """
    _data = data.copy()
    l = _data.shape[0]

    for d in days:
        noise = np.random.normal(loc=mean, scale=std, size=_data[d].shape)
        _data[l-d-1] = _data[l-d-1] + noise

    return _data

def onlyGaussianNoise(data, mean=0, std=1):
    """
    Replace the data completely with only gaussian noise.
    """
    return np.random.normal(loc=mean, scale=std, size=data.shape)

def onlyGaussianNoiseDays(data, days, mean=0, std=1):
    """
    Replace certain days with full noise.
    """
    _data = data.copy()
    l = _data.shape[0]

    for d in days:
        noise = np.random.normal(loc=mean, scale=std, size=_data[d].shape)
        _data[l-d-1] = noise

    return _data

def isolateGXY(data):
    """
    As the data files coming in have both X and Y along with Gestational Age.
    The first column is GA.
    The last column in the files in Y.
    This method isolates it into (X,Y) and returns it so that the other portions
    of the fuzzing logic can only focus on the changing the X, not the Y.
    """
    return (data[:, :1], data[:, 1:-1], data[:, -1:])

def mergeGXY(GA, X, Y):
    """
    Call this method to merge the separated (GA, X,Y) values right before writing them to
    the file.
    """
    return np.concat((GA, X, Y), axis=1)

def writeToFile(data, filename):
    """
    Method which writes the coma-separated values to the file
    """
    with open(filename, 'w') as f:
        for d in data:
            f.write(','.join([str(val) for val in d]))
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Hide', description='Only call it directly for testing')
    parser.add_argument('filename')

    args = parser.parse_args()
    data = loadData(args.filename)
    (GA, X, Y) = isolateGXY(data)
    data = mergeGXY(GA, X, Y)
    writeToFile(data, 'test.csv')
