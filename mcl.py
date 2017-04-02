from namedlist import namedlist
import math
import matplotlib.pyplot as plt
import random
import warnings
import numpy
import scipy.spatial

warnings.filterwarnings('ignore')

Belief = namedlist('Belief', 'x y theta')
SampleSet = namedlist('Sample', 'beliefs probabilities')
Action = namedlist('Action', 'x y theta')

def main():
    previous_beliefs = []
    previous_beliefs.append(Belief(0, 0, 180 * math.pi/180))
    previous_beliefs.append(Belief(1, 1, 180 * math.pi / 180))
    probability = len(previous_beliefs)*[1/len(previous_beliefs)]
    m = Action(x=0, y=0, theta=0)
    sp = SampleSet(beliefs=previous_beliefs, probabilities=probability)
    imp = MonteCarloLocalization([[3, 3, 0], [-2, 3, 0], [0, 0, 0], [1, -2, 0]])
    sensor_data = [[2, 2, 0], [2, 0, 0], [0 , 2, 0]]
    sp = imp.mcl(previous_sample=sp, motion=m, sensor_data=sensor_data)

class MonteCarloLocalization:
    def __init__(self, map):
        self.map = map
        self.kdtree = scipy.spatial.cKDTree(map, leafsize=100)
        self.zhit = 0.5
        self.ghit = 0.5
        self.zrand = 0
        self.zmax = 0.1

    def mcl(self, previous_sample, motion, sensor_data):

        for i in range(len(previous_sample.beliefs)):
            belief = previous_sample.beliefs[i]
            probability = previous_sample.probabilities[i]

            weight = self.likelihood_field_range_finder_model(sensor_data=sensor_data, pose=belief, map=self.map)

#            new_belief = motion_update(motion, belief)
#            beliefs_prime.append((new_belief, weight))

        '''
        x^m_t = draw from beliefs_prime with probability w^m_t
        new_beliefs.append(x^m_t)
        '''
        new_samples_indices = numpy.random.choice(a=len(previous_sample.beliefs), size=len(previous_sample.beliefs), p=previous_sample.probabilities)
        new_samples = SampleSet(beliefs=[], probabilities=[])
        for i in new_samples_indices:
            new_samples.beliefs.append(previous_sample.beliefs[i])
            new_samples.probabilities.append(previous_sample.probabilities[i])
        plt.waitforbuttonpress()
        return new_samples

    '''
        sample_motion_model_odometry

        Description:
            This function should move all the belief to an updated location
    '''

    def sample_motion_model_odometry(self, action, pose):
        absolute_rot1 = math.atan2(action.current.y - action.previous.y, action.current.x - action.previous.x) - action.previous.theta
        absolute_trans = math.sqrt()
    '''
        likelihood_field_range_finder_model

        Description:
            This function should give
    '''

    def likelihood_field_range_finder_model(self, sensor_data, pose, map):
        q = 1
        xk, yk = 0, 0
        plt.clf()
        plt.ion()
        for p in map:
            plt.scatter(p[0], p[1], color='black')
        for i in range(len(sensor_data)):
            thetak = 0  # compute relative angle of each sensor point to pose.theta
            global_position = list((numpy.matrix([[pose.x], [pose.y]]) + numpy.matrix(
                [[math.cos(pose.theta), -1 * math.sin(pose.theta)],
                 [math.sin(pose.theta), math.cos(pose.theta)]]) * numpy.matrix([[sensor_data[i][0]], [sensor_data[i][1]]])).flat)
            global_position.append(0)
            distance, index = self.kdtree.query(global_position, k=1)
            plt.annotate(s='', xy=global_position[0:2], xytext=self.map[index][0:2], arrowprops=dict(arrowstyle='<->'))
            q *= (self.zhit * self.prob(distance, self.ghit) + self.zrand/self.zmax)
            plt.scatter(sensor_data[i][0], sensor_data[i][1], color='blue', marker='x')
            plt.scatter(global_position[0], global_position[1], color='red')
        plt.axis('scaled')
        plt.waitforbuttonpress()
        print('q = ', q)
        return q
    '''
        sensor_update

        Description:
            This function should move all the belief to an updated location
    '''

    def sensor_update(self, sample, action):
        pass

    def prob(self, x, sd):
        return 1 / (math.sqrt(2 * math.pi * (sd ** 2))) * math.exp(-(x ** 2) / (2 * (sd ** 2)))

def plot_points(a):
    plt.ion()
    plt.clf()
    ax = plt.axes()
    for p in a:
        plt.plot(p[0:1], marker='o', color='b', ls='')
    plt.waitforbuttonpress()

if __name__ == "__main__" :
    main()