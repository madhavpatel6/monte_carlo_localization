#!/usr/bin/python
from __future__ import division
from namedlist import namedlist
import math
import matplotlib.pyplot as plt
import random
import warnings
import numpy
import scipy.spatial
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import sensor_msgs.point_cloud2 as PCL
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
warnings.filterwarnings('ignore')
import tf
import message_filters
import time
Belief = namedlist('Belief', 'x y theta')
SampleSet = namedlist('Sample', 'beliefs probabilities')

class Motion:
    def __init__(self):
        self.previous = Belief(0, 0, 0)
        self.current = Belief(0, 0, 0)
    def __str__(self):
        return str('Previous Pose = ' + str(self.previous) + '\nCurrent Pose = ' + str(self.current))

def main():
    rospy.init_node('mcl', anonymous=True)
    lc = Localizer()
    lc.localize()

class Localizer:
    def __init__(self):
        self.new_odom = None
        self.old_odom = Pose()
        self.sample_size = 10000
        self.mcl_complete = True
        self.rotation_threshold = 0.25
        self.translation_threshold = .1

    def callback(self, odom, scan):
        if self.mcl_complete:
            self.new_odom = odom.pose.pose
            self.new_scan = MapSeverHelper.pointcloud2_to_list(scan)

    def localize(self):
        p = rospy.Publisher('/mcl/map', PointCloud2, queue_size=10)
        sp_publisher = rospy.Publisher('/mcl/beliefs', PointCloud2, queue_size=10)

        map = MapSeverHelper.convert_occupancygrid_to_map()
        rospy.loginfo('Received map of length ' + str(len(map)))

        pcloud = MapSeverHelper.list_2d_to_pointcloud2(map, 'odom')
        p.publish(pcloud)

        inst = MonteCarloLocalization(map)

        odom_sub = message_filters.Subscriber('/odom', Odometry)
        scan_sub = message_filters.Subscriber('/cloud_out', PointCloud2)
        ts = message_filters.TimeSynchronizer([odom_sub, scan_sub], 10)
        ts.registerCallback(self.callback)

        # Find c-space
        min_x = min(map, key=lambda p: p[0])[0]
        max_x = max(map, key=lambda p: p[0])[0]
        min_y = min(map, key=lambda p: p[1])[1]
        max_y = max(map, key=lambda p: p[1])[1]
        c_space = ((min_x, max_x), (min_y, max_y))
        rospy.loginfo('Configuration space is (%.3f, %.3f) x (%.3f, %.3f)' % (min_x, max_x, min_y, max_y))
        sp = self.create_initial_sample_set(c_space)
        #sp = self.create_initial_sample_set_at_position(Belief(0,0,0))
        # Publish sample set
        self.publish_sample_set(sp_publisher, sp.beliefs)
        while True:
            # Receive new odometry information from
            if self.new_odom is not None and self.new_scan is not None and self.check_delta(self.old_odom, self.new_odom):
                self.mcl_complete = False
                motion = self.create_action(self.old_odom, self.new_odom)
                rospy.loginfo('Running MCL')
                sp = inst.mcl(previous_sample=sp, motion=motion, sensor_data=self.new_scan)
                self.publish_sample_set(sp_publisher, sp.beliefs)
                # Convert
                self.old_odom = self.new_odom
                self.new_odom = None
                self.new_scan = None
                self.mcl_complete = True

    def check_delta(self, old_odom, new_odom):
        if self.new_odom is None:
            return False
        if self.old_odom is None:
            return True
        else:
            dx = self.new_odom.position.x - self.old_odom.position.x
            dy = self.new_odom.position.y - self.old_odom.position.y

            if (dx ** 2 + dy ** 2) >= self.translation_threshold ** 2:
                return True
            q = [self.new_odom.orientation.x, self.new_odom.orientation.y,
                 self.new_odom.orientation.z, self.new_odom.orientation.w]
            new_yaw = tf.transformations.euler_from_quaternion(q)[2]
            if new_yaw < 0:
                new_yaw += 2 * math.pi
            q = [self.old_odom.orientation.x, self.old_odom.orientation.y,
                 self.old_odom.orientation.z, self.old_odom.orientation.w]
            old_yaw = tf.transformations.euler_from_quaternion(q)[2]
            if old_yaw < 0:
                old_yaw += 2 * math.pi

            # print('yaw', new_yaw*180/3.14)
            dyaw = abs(new_yaw - old_yaw) * 180 / math.pi
            if dyaw > 180:
                dyaw = 360 - dyaw
            if abs(dyaw) >= self.rotation_threshold:
                return True
        return False

    def publish_sample_set(self, pub, sp):
        pcloud = MapSeverHelper.list_to_pointcloud2(sp, 'odom')
        pub.publish(pcloud)

    def create_initial_sample_set(self, c_space):
        beliefs = [Belief(random.uniform(c_space[0][0], c_space[0][1]), random.uniform(c_space[1][0], c_space[1][1]), random.uniform(-math.pi, math.pi)) for p in range(self.sample_size)]
        probabilities = self.sample_size*[1/self.sample_size]
        sp = SampleSet(beliefs=beliefs, probabilities=probabilities)
        return sp

    def create_initial_sample_set_at_position(self, belief):
        beliefs = [belief for p in range(self.sample_size)]
        probabilities = self.sample_size*[1/self.sample_size]
        sp = SampleSet(beliefs=beliefs, probabilities=probabilities)
        return sp

    def create_action(self, old, new):
        a = Motion()
        a.previous.x = old.position.x
        a.previous.y = old.position.y
        a.previous.theta = tf.transformations.euler_from_quaternion(
            [old.orientation.x, old.orientation.y, old.orientation.z, old.orientation.w])[2]

        a.current.x = new.position.x
        a.current.y = new.position.y
        a.current.theta = tf.transformations.euler_from_quaternion(
            [new.orientation.x, new.orientation.y, new.orientation.z, new.orientation.w])[2]
        return a


class MonteCarloLocalization:
    def __init__(self, map):
        self.map = map
        self.kdtree = scipy.spatial.cKDTree(map, leafsize=100)
        self.alpha1 = 0.02      # Error in rot1/rot2 due to rot1/rot2
        self.alpha2 = 0.001     # Error in rot1/rot2 due to translation
        self.alpha3 = 0.01     # Error in translation due to translation
        self.alpha4 = 0.001    # Error in translation due to rot1/rot2
        self.zhit = 0.5
        self.sdhit = 0.1
        self.zrand = 0.00001
        self.zmax = 1

    def mcl(self, previous_sample, motion, sensor_data):
        samples_bar = SampleSet([], [])
        rospy.loginfo('Beginning to iterate through previous sample')
        avgtime = 0
        for i in range(len(previous_sample.beliefs)):
            new_belief = self.sample_motion_model_odometry(motion, previous_sample.beliefs[i])
            old = time.time()
            weight = self.likelihood_field_range_finder_model(sensor_data=sensor_data, pose=new_belief)
            avgtime += time.time() - old
            samples_bar.beliefs.append(new_belief)
            samples_bar.probabilities.append(weight)
        print("Average time", avgtime/len(previous_sample.beliefs))
        # Normalize probabilities
        rospy.loginfo('Normalizing new samples')
        probability_factor = 1 / sum(samples_bar.probabilities)
        for i in range(len(samples_bar.probabilities)):
            samples_bar.probabilities[i] *= probability_factor
        rospy.loginfo('Drawing from new samples')
        # Draw from samples from samples_bar with new weights
        new_samples_indices = numpy.random.choice(a=len(samples_bar.beliefs), size=len(samples_bar.beliefs), p=samples_bar.probabilities)
        new_samples = SampleSet(beliefs=[], probabilities=[])
        for i in new_samples_indices:
            new_samples.beliefs.append(samples_bar.beliefs[i])
            new_samples.probabilities.append(samples_bar.probabilities[i])
        return new_samples

    '''
        sample_motion_model_odometry

        Description:
            This function should move all the belief to an updated location
    '''

    def sample_motion_model_odometry(self, action, pose):
        absolute_rot1 = math.atan2(action.current.y - action.previous.y,
                                   action.current.x - action.previous.x) - action.previous.theta
        absolute_trans = math.sqrt(
            (action.current.x - action.previous.x) ** 2 + (action.current.y - action.previous.y) ** 2)
        absolute_rot2 = action.current.theta - action.previous.theta - absolute_rot1
        error_rot1 = self.sample(self.alpha1*(absolute_rot1**2) + self.alpha2*(absolute_trans**2))
        error_trans = self.sample(
            self.alpha3 * (absolute_trans ** 2) + self.alpha4 * (absolute_rot1 ** 2) + self.alpha4 * (
            absolute_rot2 ** 2))
        error_rot2 = self.sample(self.alpha1*(absolute_rot2**2) + self.alpha2*(absolute_trans**2))
        estimate_rot1 = absolute_rot1 - error_rot1
        estimate_trans = absolute_trans - error_trans
        estimate_rot2 = absolute_rot2 - error_rot2
        x = pose.x + estimate_trans * math.cos(pose.theta + estimate_rot1)
        y = pose.y + estimate_trans * math.sin(pose.theta + estimate_rot1)
        theta = pose.theta + estimate_rot1 + estimate_rot2
        return Belief(x, y, theta)


    '''
        likelihood_field_range_finder_model

        Description:
            This function should give
    '''

    def likelihood_field_range_finder_model(self, sensor_data, pose):
        q = 1
#        plt.ion()
#        for s in sensor_data:
#            plt.scatter(s[0], s[1], color="red", s=5)

        for i in range(len(sensor_data)):
            global_position = list((numpy.matrix([[pose.x], [pose.y]]) + numpy.matrix(
                [[math.cos(pose.theta), -1 * math.sin(pose.theta)],
                 [math.sin(pose.theta), math.cos(pose.theta)]]) * numpy.matrix([[sensor_data[i][0]], [sensor_data[i][1]]])).flat)
            plt.scatter(global_position[0], global_position[1], color='blue', s=3)
            distance, index = self.kdtree.query(global_position, k=1)
#            print(sensor_data[i], " -> ", self.map[index])
#            print("distance =", distance)
            p = self.prob(distance, self.sdhit)
#            print("prob", p)
            factor = (self.zhit * p + self.zrand/self.zmax)
#            print("factor", factor)
            q *= factor
#            print("new q", q)
#            plt.scatter(self.map[index][0], self.map[index][1], color='black', s=5)
#        print('Likelihood %0.5f' % q)
#        plt.waitforbuttonpress()
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
        sp = [random.uniform(-sd, sd) for i in range(12)]
        return sum(sp)/2.0  # why 12?

    def prob(self, x, sd):
        return (1.0 / (math.sqrt(2.0 * math.pi * (sd ** 2)))) * math.exp(-(x ** 2) / (2 * (sd ** 2)))


class MapSeverHelper():
    @staticmethod
    def convert_occupancygrid_to_map():
        data = rospy.wait_for_message('/map', OccupancyGrid)
        map = []
        for i in range(0, data.info.height):
            for j in range(0, data.info.width):
                if data.data[(i)*data.info.width + j] >= 65:
                    map_x = j * data.info.resolution + data.info.origin.position.x
                    map_y = i * data.info.resolution + data.info.origin.position.y
                    map.append([map_x, map_y])
        return map

    @staticmethod
    def list_2d_to_pointcloud2(points, frame):
        points_3d = []
        for p in points:
            points_3d.append([p[0], p[1], 0])
        pcloud = PointCloud2()
        pcloud = PCL.create_cloud_xyz32(pcloud.header, points_3d)
        pcloud.header.stamp = rospy.Time.now()
        pcloud.header.frame_id = frame
        return pcloud

    @staticmethod
    def list_to_pointcloud2(points, frame):
        pcloud = PointCloud2()
        pcloud = PCL.create_cloud_xyz32(pcloud.header, points)
        pcloud.header.stamp = rospy.Time.now()
        pcloud.header.frame_id = frame
        return pcloud

    @staticmethod
    def pointcloud2_to_list(cloud):
        gen = PCL.read_points(cloud, skip_nans=True, field_names=('x', 'y', 'z'))
        list_of_tuples = list(gen)
        return list_of_tuples

if __name__ == "__main__" :
    main()