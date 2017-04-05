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
from geometry_msgs.msg import Pose, TransformStamped
import sensor_msgs.point_cloud2 as PCL
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
warnings.filterwarnings('ignore')
import tf
import message_filters
import sys
import tf2_sensor_msgs.tf2_sensor_msgs
Belief = namedlist('Belief', 'x y theta')
SampleSet = namedlist('Sample', 'beliefs probabilities')

def main():
    rospy.init_node('mcl', anonymous=True)
    lc = Localizer()
    lc.localize()

class Localizer:
    def __init__(self):
        self.new_odom = None
        self.old_odom = None
        self.new_scan = None
        self.sample_size = 5000
        self.mcl_complete = True
        self.rotation_threshold = 0.25
        self.translation_threshold = .1

    def callback(self, odom, scan):
        if self.mcl_complete:
            self.new_odom = odom.pose.pose
            self.new_scan = MapSeverHelper.pointcloud2_to_list(scan)
            # If this is the first callback we initialize the old position to be the current one
            if self.old_odom is None:
                self.old_odom = self.new_odom
                self.new_odom = None

    def localize(self):
        p = rospy.Publisher('/mcl/map', PointCloud2, queue_size=10)
        sp_publisher = rospy.Publisher('/mcl/beliefs', PointCloud2, queue_size=10)

        map, res = MapSeverHelper.convert_occupancygrid_to_map()
        rospy.loginfo('Received map of length ' + str(len(map)))

        pcloud = MapSeverHelper.list_2d_to_pointcloud2(map, 'odom')
        p.publish(pcloud)

        # Find c-space
        min_x = min(map, key=lambda p: p[0])[0]
        max_x = max(map, key=lambda p: p[0])[0]
        min_y = min(map, key=lambda p: p[1])[1]
        max_y = max(map, key=lambda p: p[1])[1]
        c_space = ((min_x, max_x), (min_y, max_y))

        inst = MonteCarloLocalization(map, c_space, res)
        inst.preprocess_map()
        rospy.loginfo('Configuration space is (%.3f, %.3f) x (%.3f, %.3f)' % (min_x, max_x, min_y, max_y))
        sp = self.create_initial_sample_set(c_space)

        #sp = self.create_initial_sample_set_at_position(Belief(0,0,0))
        # Publish sample set
        self.publish_sample_set(sp_publisher, sp.beliefs)

        odom_sub = message_filters.Subscriber('/odom', Odometry)
        scan_sub = message_filters.Subscriber('/cloud_out', PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([odom_sub, scan_sub], 10, 1)
        ts.registerCallback(self.callback)

        while True:
            # Receive new odometry information from
            if self.new_odom is not None and self.new_scan is not None and self.check_delta(self.old_odom, self.new_odom):
                self.mcl_complete = False
                motion = self.create_action(self.old_odom, self.new_odom)
                rospy.loginfo('Running MCL')
                #sp = inst.mcl(previous_sample=sp, motion=motion, sensor_data=self.new_scan)
                sp = inst.augmented_mcl(previous_sample=sp, motion=motion, sensor_data=self.new_scan)
                self.publish_sample_set(sp_publisher, sp.beliefs)
                # Convert
                self.old_odom = self.new_odom
                self.new_odom = None
                self.new_scan = None
                self.mcl_complete = True

    def check_delta(self, old_odom, new_odom):
        if self.new_odom is None:
            return False
        else:
            dx = self.new_odom.position.x - self.old_odom.position.x
            dy = self.new_odom.position.y - self.old_odom.position.y

            if (dx * dx + dy * dy) >= self.translation_threshold * self.translation_threshold:
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
        probabilities = (self.sample_size)*[1/(self.sample_size)]
        sp = SampleSet(beliefs=beliefs, probabilities=probabilities)
        return sp

    def create_initial_sample_set_at_position(self, belief):
        beliefs = [belief for p in range(self.sample_size)]
        probabilities = self.sample_size*[1/self.sample_size]
        sp = SampleSet(beliefs=beliefs, probabilities=probabilities)
        return sp

    def create_action(self, old, new):
        previous_theta = tf.transformations.euler_from_quaternion([old.orientation.x, old.orientation.y, old.orientation.z, old.orientation.w])[2]
        current_theta = tf.transformations.euler_from_quaternion([new.orientation.x, new.orientation.y, new.orientation.z, new.orientation.w])[2]
        motion = [math.atan2(new.position.y - old.position.y, new.position.x - old.position.x) - previous_theta,
             math.sqrt((new.position.x - old.position.x) ** 2 + (new.position.y - old.position.y) ** 2)]
        motion.append(current_theta - previous_theta - motion[0])
        return motion


class MonteCarloLocalization:
    def __init__(self, map, c_space, res):
        self.map = map
        self.c_space = c_space
        self.resolution = res
        self.likelihood_field = None
        self.origin = None
        self.dimensions = None
        self.kdtree = scipy.spatial.cKDTree(map, leafsize=100)
        self.alpha1 = 0.01      # Error in rot1/rot2 due to rot1/rot2
        self.alpha2 = 0.001     # Error in rot1/rot2 due to translation
        self.alpha3 = 0.01     # Error in translation due to translation
        self.alpha4 = 0.001    # Error in translation due to rot1/rot2
        self.z_hit = 0.95
        self.sigma_hit = 0.2
        self.z_rand = 0.005
        self.z_max = 1
        self.alpha_slow = 0.05
        self.alpha_fast = 0.2
        self.w_slow = 0
        self.w_fast = 0

    def mcl(self, previous_sample, motion, sensor_data):
        samples_bar = SampleSet([], [])
        length = len(previous_sample.beliefs)
        for i in range(length):
            new_belief = self.sample_motion_model_odometry(motion, previous_sample.beliefs[i])
            weight = self.likelihood_field_range_finder_model(sensor_data=sensor_data, pose=new_belief)
            samples_bar.beliefs.append(new_belief)
            samples_bar.probabilities.append(weight)
        # Normalize probabilities
        total = sum(samples_bar.probabilities)
        if total > 0:
            probability_factor = 1 / sum(samples_bar.probabilities)
            for i in range(length):
                samples_bar.probabilities[i] *= probability_factor
        else:
            for i in range(length):
                samples_bar.probabilities[i] = 1 / length

        new_samples_indices = numpy.random.choice(a=length, size=length, p=samples_bar.probabilities)
        new_samples = SampleSet(beliefs=[], probabilities=[])
        for i in new_samples_indices:
            new_samples.beliefs.append(samples_bar.beliefs[i])
            new_samples.probabilities.append(samples_bar.probabilities[i])
        return new_samples

    def augmented_mcl(self, previous_sample, motion, sensor_data):
        samples_bar = SampleSet([], [])
        length = len(previous_sample.beliefs)
        for i in range(length):
            new_belief = self.sample_motion_model_odometry(motion, previous_sample.beliefs[i])
            weight = self.likelihood_field_range_finder_model(sensor_data=sensor_data, pose=new_belief)
            samples_bar.beliefs.append(new_belief)
            samples_bar.probabilities.append(weight)
        total = sum(samples_bar.probabilities)
        w_avg = 0
        if total > 0:
            probability_factor = 1 / sum(samples_bar.probabilities)
            for i in range(length):
                w_avg += samples_bar.probabilities[i]
                samples_bar.probabilities[i] *= probability_factor

            if self.w_slow != 0:
                self.w_slow += self.alpha_slow * (w_avg - self.w_slow)
            else:
                self.w_slow = self.alpha_slow * w_avg

            if self.w_fast != 0:
                self.w_fast += self.alpha_fast * (w_avg - self.w_fast)
            else:
                self.w_fast = self.alpha_fast * w_avg
        else:
            for i in range(length):
                samples_bar.probabilities[i] = 1 / length
        w_diff = 1.0 - self.w_fast/self.w_slow
        print('w_avg %0.3f' % w_avg)
        print('slow %0.3f fast %0.3f w_diff %0.3f' % (self.w_slow, self.w_fast, w_diff))
        new_samples = SampleSet(beliefs=[], probabilities=[])
        if w_diff > 0:

            ctr = 0
            for i in range(length):
                if random.random() < w_diff:
                    ctr += 1
                    new_samples.beliefs.append(Belief(random.uniform(self.c_space[0][0], self.c_space[0][1]), random.uniform(self.c_space[1][0], self.c_space[1][1]), random.uniform(-math.pi, math.pi)))
                    new_samples.probabilities.append(1.0)
            print('added ', ctr, 'random samples')
        new_samples_indices = numpy.random.choice(a=length, size=length-len(new_samples.beliefs), p=samples_bar.probabilities)

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
        absolute_rot1 = action[0]
        absolute_trans = action[1]
        absolute_rot2 = action[2]
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
        pos = theta <= math.pi
        neg = theta > -math.pi
        if pos and neg:
            return Belief(x, y, theta)
        elif not neg:
            theta_n = theta + 2.0*math.pi*math.floor((abs(theta) + math.pi)/(2.0*math.pi))
        elif not pos:
            theta_n = theta - 2.0 * math.pi * math.floor((theta + math.pi) / (2.0 * math.pi))
        return Belief(x, y, theta_n)

    '''
        likelihood_field_range_finder_model

        Description:
            This function should give
    '''

    def likelihood_field_range_finder_model(self, sensor_data, pose):
        q = 1
        cos_theta = math.cos(pose.theta)
        sin_theta = math.sin(pose.theta)
        for i in range(len(sensor_data)):
            global_position_x, global_position_y = pose.x + cos_theta * sensor_data[i][0] - sin_theta * sensor_data[i][1], pose.y + sin_theta * sensor_data[i][0] + cos_theta * sensor_data[i][1]
            x, y = int((global_position_x - self.origin[0])/self.resolution), int((global_position_y - self.origin[1])/self.resolution)
            if 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]:
                pgaus = self.likelihood_field[y][x]
            else:
                pgaus = 0
            p = pgaus + random.uniform(0, self.z_rand) #/self.z_max
            q += p*p*p
        return q

    def preprocess_map(self):
        min_x = self.c_space[0][0]
        max_x = self.c_space[0][1]
        min_y = self.c_space[1][0]
        max_y = self.c_space[1][1]
        self.origin = [min_x, min_y]
        width = int(math.ceil((abs(min_x) + abs(max_x)) / self.resolution))
        height = int(math.ceil((abs(min_y) + abs(max_y)) / self.resolution))
        self.dimensions = [width, height]
        rospy.loginfo("Creating a field with dimensions %d %d" % (width, height))
        self.likelihood_field = []
        for y in range(height):
            self.likelihood_field.append([])
            for x in range(width):
                global_position = [x * self.resolution + self.origin[0], y * self.resolution + self.origin[1]]
                distance, index = self.kdtree.query(global_position, k=1)
                self.likelihood_field[y].append(self.z_hit * self.prob(distance, self.sigma_hit))
        '''
        x_index = []
        y_index = []
        dist = []
        for y in range(int(height)):
            for x in range(int(width)):
                x_index.append(x)
                y_index.append(y)
                dist.append(self.likelihood_field[y][x])

        plt.scatter(x_index, y_index, s=50, c=dist, cmap='gray_r')
        ptt.show()
        '''
        rospy.loginfo('Completed preprocessing.')


    def sample(self, var):
        sd = math.sqrt(var)
        sp = [random.uniform(-sd, sd) for i in range(12)]
        return sum(sp)/2.0  # why 12?

    def prob(self, x, sd):
        # (1.0 / (math.sqrt(2.0 * math.pi * (sd ** 2)))) *
        return math.exp(-(x * x) / (2 * (sd * sd)))


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
        return map, data.info.resolution

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