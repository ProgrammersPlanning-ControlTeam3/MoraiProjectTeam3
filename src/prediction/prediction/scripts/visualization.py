#!/usr/bin/env python3
# coding: utf-8
import rospy
import tf.transformations
from morai_msgs.msg  import ObjectStatusList
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList
from std_msgs.msg import Int32
from math import radians

class DynamicObstacleVisualizeNode:
    def __init__(self):
        rospy.init_node('dynamic_obstacle_visualize_node', anonymous=True)
        rospy.Subscriber('/Object_topic/tracked_object_pose_topic', TrackedObjectPoseList, self.object_info_callback)
        rospy.Subscriber('/Object_topic/tracked_object_path_topic', PredictedObjectPathList, self.object_path_callback)
        rospy.Subscriber('/Object_topic/deleted_object_id', Int32, self.deleted_object_callback)
        self.pred_pose = rospy.Publisher('/rviz/pred_pose_dynamic_object', MarkerArray, queue_size=10)
        self.pred_path = rospy.Publisher('/rviz/pred_path_dynamic_object', MarkerArray, queue_size=10)

        self.rate = rospy.Rate(30)
        self.is_pose_received = False
        self.object_pose = None
        self.is_path_received = False
        self.object_path = None

        self.deleted_ids = set()

    def object_info_callback(self, msg):
        self.is_pose_received=True
        self.object_pose = msg

    def object_path_callback(self, msg):
        self.is_path_received=True
        self.object_path = msg

    def deleted_object_callback(self, msg):
        self.deleted_ids.add(msg.data)

    def pub_object_pose(self):
        Objects = MarkerArray()
        for obstacle in self.object_pose.pose_list:
            unique_id = obstacle.unique_id
            pose = obstacle.pose

            q = tf.transformations.quaternion_from_euler(0, 0, radians(pose.theta))
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = str(unique_id)
            marker.id = unique_id
            marker.type = Marker.CUBE

            marker.pose.position.x = pose.x
            marker.pose.position.y = pose.y
            marker.pose.position.z = 0.15

            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = 1.87
            marker.scale.y = 4.97
            marker.scale.z = 1.47

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            Objects.markers.append(marker)

        self.pred_pose.publish(Objects)
        self.delete_markers()

    def pub_object_path(self):
        Paths = MarkerArray()

        for obstacle in self.object_path.path_list:
            path = Marker()
            path.header.frame_id = "map"
            path.header.stamp = rospy.Time.now()
            path.ns = str(obstacle.unique_id)
            path.id = obstacle.unique_id
            path.type = Marker.LINE_STRIP
            path.action = Marker.ADD
            path.pose.orientation.w = 1.0
            path.scale.x = 0.7

            path.color.r = 1.0
            path.color.g = 0.0
            path.color.b = 0.0
            path.color.a = 1.0

            # pred_path: [[x, y, v, a, theta, theta_rate],...]
            for pose in obstacle.path:
                point = Point()
                point.x = pose.x
                point.y = pose.y
                point.z = 0
                path.points.append(point)

            Paths.markers.append(path)

        self.pred_path.publish(Paths)
        self.delete_markers()

    def delete_markers(self):
        if not self.deleted_ids:
            return

        delete_array = MarkerArray()
        for unique_id in self.deleted_ids:
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.ns = str(unique_id)
            delete_marker.id = unique_id
            delete_marker.action = Marker.DELETE
            delete_array.markers.append(delete_marker)

        self.pred_pose.publish(delete_array)
        self.pred_path.publish(delete_array)

        rospy.sleep(0.1)

        self.deleted_ids.clear()

    def run(self):
        while not rospy.is_shutdown():
            if self.is_pose_received == True :
                self.pub_object_pose()
            if self.is_path_received == True:
                self.pub_object_path()

            self.rate.sleep()

if __name__ == '__main__':
    try:
        visualize = DynamicObstacleVisualizeNode()
        visualize.run()
    except rospy.ROSInterruptException:
        pass
