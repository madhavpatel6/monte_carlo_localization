from namedlist import namedlist
import copy
import math
import matplotlib.pyplot as plt
import random
import warnings
import numpy
warnings.filterwarnings('ignore')

Belief = namedlist('Belief', 'x y theta')
SampleSet = namedlist('Sample', 'beliefs probabilities')
Action = namedlist('Action', 'x y theta')


def main():
    previous_beliefs = []
    for i in range(100):
        previous_beliefs.append(Belief(0, 1, random.uniform(0, 6.28)))
    probability = len(previous_beliefs)*[1/len(previous_beliefs)]
    m = Action(x=0, y=0, theta=0)
    sp = SampleSet(beliefs=previous_beliefs, probabilities=probability)
    while True:
        sp = mcl(previous_sample=sp, motion=m, sensor_data=[])

class MonteCarloLocalization:
    def __init__(self):
        pass

    def mcl(self, previous_sample, motion, sensor_data):

        for i in range(len(previous_sample.beliefs)):
            belief = previous_sample.beliefs[i]
            probability = previous_sample.probabilities[i]

            motion_update(belief, motion)
            '''
            new_belief = motion_update(motion, belief)
            weight = sensor_update(sensor_data, new_belief)
            beliefs_prime.append((new_belief, weight))
            '''

        '''
        x^m_t = draw from beliefs_prime with probability w^m_t
        new_beliefs.append(x^m_t)
        '''
        new_samples_indices = numpy.random.choice(a=len(previous_sample.beliefs), size=len(previous_sample.beliefs), p=previous_sample.probabilities)
        new_samples = SampleSet(beliefs=[], probabilities=[])
        for i in new_samples_indices:
            new_samples.beliefs.append(previous_sample.beliefs[i])
            new_samples.probabilities.append(previous_sample.probabilities[i])
        plot_points(new_samples.beliefs)
        return new_samples

'''
    likelihood_field_range_finder_model

    Description:
        This function should move all the belief to an updated location
'''


def likelihood_field_range_finder_model(sensor_data, pose, map):
    q = 1
    xk, yk = 0, 0
    for i in range(len(sensor_data)):

        thetak = 0  # compute relative angle of each sensor point to pose.theta
        xz = pose.x + xk * math.cos(pose.theta) - yk * math.sin(pose.theta) + sensor_data[i] * math.cos(
            pose.theta + thetak)
        yz = pose.y + yk * math.cos(pose.theta) + xk * math.sin(pose.theta) + sensor_data[i] * math.sin(
            pose.theta + thetak)




'''
    sensor_update

    Description:
        This function should move all the belief to an updated location
'''


def sensor_update(sample, action):
    pass


def plot_points(a):
    plt.ion()
    plt.clf()
    ax = plt.axes()
    for p in a:
        plt.plot(p[0:1], marker='o', color='b', ls='')
        ax.arrow(p[0], p[1], )
    plt.waitforbuttonpress()

if __name__ == "__main__" :
    main()