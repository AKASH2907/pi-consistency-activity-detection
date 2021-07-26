import numpy as np
import math
from matplotlib import pyplot as plt

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    return current / rampup_length

def cosine_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return 1 - float(.5 * (np.cos(np.pi * current / rampup_length) + 1))

def log_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(1- np.exp(-5.0 * current / rampup_length))
    
def exp_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(np.exp(5.0 * (current / rampup_length - 1)))

def rampweight(iteration):
    ramp_up_end = 32000
    ramp_down_start = 100000

    if(iteration<ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end),2))
    elif(iteration>ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000),2)) 
    else:
        ramp_weight = 1 


    if(iteration==0):
        ramp_weight = 0

    return ramp_weight


def test_warmup():
    # warmup = rampweight(80, 50, 100)
    xpoints = np.arange(51700)
    ypoints = list()
    for ep in range(51700):
        rw = rampweight(ep)
        # print(ep, rw)
        ypoints.append(rw)

    ypoints = np.array(ypoints)

    plt.figure()
    plt.plot(xpoints, ypoints)
    plt.savefig('exp.png')
# test_warmup()