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


class Action:
    def __init__(self):
        self.previous = Belief(0, 0, 0)
        self.current = Belief(0, 0, 0)


def main():
    previous_beliefs = 5*[Belief(0, 0, 0)]
    plt.scatter(0, 0, color='black')
    probability = len(previous_beliefs)*[1/len(previous_beliefs)]
    action = Belief(2, 2, 2)
    sp = SampleSet(beliefs=previous_beliefs, probabilities=probability)
    imp = MonteCarloLocalization([[3, 3, 0], [-2, 3, 0], [0, 0, 0], [1, -2, 0]])
    sensor_data = [[2, 2, 0], [2, 0, 0], [0, 2, 0]]
    while True:
        l = input()
        x, y, theta = float(l.split(' ')[0]), float(l.split(' ')[1]), float(l.split(' ')[2])
        print('Moving to ', x, y, theta)
        theta *= math.pi/180
        sp = imp.mcl(previous_sample=sp, motion=Belief(x, y, theta), sensor_data=sensor_data)


class MonteCarloLocalization:
    def __init__(self, map):
        self.map = map
        self.kdtree = scipy.spatial.cKDTree(map, leafsize=100)
        self.action = Action()
        self.alpha1 = 0.05      # Error in rot1/rot2 due to rot1/rot2
        self.alpha2 = 0.0001     # Error in rot1/rot2 due to translation
        self.alpha3 = 0.001     # Error in translation due to translation
        self.alpha4 = 0.01     # Error in translation due to rot1/rot2
        self.zhit = 0.5
        self.sdhit = 0.5
        self.zrand = 0
        self.zmax = 0.1

    def mcl(self, previous_sample, motion, sensor_data):
        plt.ion()
        self.action.current = motion
        samples_bar = SampleSet([], [])
        for i in range(len(previous_sample.beliefs)):
            new_belief = self.sample_motion_model_odometry(self.action, previous_sample.beliefs[i])
            weight = self.likelihood_field_range_finder_model(sensor_data=sensor_data, pose=new_belief, map=self.map)
            samples_bar.beliefs.append(new_belief)
            samples_bar.probabilities.append(weight)
        # Normalize probabilities
        print(samples_bar)
        probability_factor = 1 / sum(samples_bar.probabilities)
        for i in range(len(samples_bar.probabilities)):
            samples_bar.probabilities[i] *= probability_factor
        print(samples_bar)
        '''
        x^m_t = draw from beliefs_prime with probability w^m_t
        new_beliefs.append(x^m_t)
        '''
        '''
        new_samples_indices = numpy.random.choice(a=len(sample_set_bar.beliefs), size=len(sample_set_bar.beliefs), p=sample_set_bar.probabilities)
        new_samples = SampleSet(beliefs=[], probabilities=[])
        for i in new_samples_indices:
            new_samples.beliefs.append(sample_set_bar.beliefs[i])
            new_samples.probabilities.append(sample_set_bar.probabilities[i])
        plt.waitforbuttonpress()
        self.action.current = None
        self.action.previous = self.action.current
        '''
        self.action.previous = self.action.current
        plt.pause(0.01)
        return samples_bar

    '''
        sample_motion_model_odometry

        Description:
            This function should move all the belief to an updated location
    '''

    def sample_motion_model_odometry(self, action, pose):
        plt.figure(1)
        absolute_rot1 = math.atan2(action.current.y - action.previous.y,
                                   action.current.x - action.previous.x) - action.previous.theta
        absolute_trans = math.sqrt(
            (action.current.x - action.previous.x) ** 2 + (action.current.y - action.previous.y) ** 2)
        absolute_rot2 = action.current.theta - action.previous.theta - absolute_rot1
        print('%.3f, %.3f, %.3f' % (absolute_rot1, absolute_trans, absolute_rot2))
        estimate_rot1 = absolute_rot1 - self.sample(self.alpha1*(absolute_rot1**2) + self.alpha2*(absolute_trans**2))
        estimate_trans = absolute_trans - self.sample(
            self.alpha3 * (absolute_trans ** 2) + self.alpha4 * (absolute_rot1 ** 2) + self.alpha4 * (
            absolute_rot2 ** 2))
        estimate_rot2 = absolute_rot2 - self.sample(self.alpha1*(absolute_rot2**2) + self.alpha2*(absolute_trans**2))

        x = pose.x + estimate_trans * math.cos(pose.theta + estimate_rot1)
        y = pose.y + estimate_trans * math.sin(pose.theta + estimate_rot1)
        theta = pose.theta + estimate_rot1 + estimate_rot2
        plt.scatter(x, y, color='blue', s=5)
        return Belief(x, y, theta)


    '''
        likelihood_field_range_finder_model

        Description:
            This function should give
    '''

    def likelihood_field_range_finder_model(self, sensor_data, pose, map):
        q = 1
        xk, yk = 0, 0
        plt.figure(2)
        plt.clf()
        plt.ion()
        print(pose)
        for p in map:
            plt.scatter(p[0], p[1], color='black')
        for i in range(len(sensor_data)):
            global_position = list((numpy.matrix([[pose.x], [pose.y]]) + numpy.matrix(
                [[math.cos(pose.theta), -1 * math.sin(pose.theta)],
                 [math.sin(pose.theta), math.cos(pose.theta)]]) * numpy.matrix([[sensor_data[i][0]], [sensor_data[i][1]]])).flat)
            global_position.append(0)
            distance, index = self.kdtree.query(global_position, k=1)
            plt.annotate(s='', xy=global_position[0:2], xytext=self.map[index][0:2], arrowprops=dict(arrowstyle='<->'))
            q *= (self.zhit * self.prob(distance, self.sdhit) + self.zrand/self.zmax)
            plt.scatter(sensor_data[i][0], sensor_data[i][1], color='blue', marker='x')
            plt.scatter(global_position[0], global_position[1], color='red')
        plt.axis('scaled')
        plt.waitforbuttonpress()
        return q
    '''
        sensor_update

        Description:
            This function should move all the belief to an updated location
    '''

    def sensor_update(self, sample, action):
        pass

    def sample(self, var):
        sd = math.sqrt(var)
        return (1/2)*sum(random.uniform(-sd, sd) for i in range(12))  # why 12?

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