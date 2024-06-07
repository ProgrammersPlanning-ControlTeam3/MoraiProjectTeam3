#!/usr/bin/env python3
# coding: utf-8
import rospy
import tf.transformations
from morai_msgs.msg import ObjectStatusList
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList, PredictedHMM
from std_msgs.msg import Int32, ColorRGBA
from math import radians

class HMMPredictionVisualizeNode:
    def __init__(self):
        rospy.init_node('hmm_prediction_visualization_node', anonymous=True)
        rospy.Subscriber('/Object_topic/hmm_prediction', PredictedHMM, self.prediction_info_callback)
        self.intention_publisher = rospy.Publisher('/rviz/pred_intention_object', MarkerArray, queue_size= 10 )
        self.maneuver_publisher = rospy.Publisher('rviz/pred_probability_object', MarkerArray, queue_size= 10 )

        self.rate = rospy.Rate(30)
        self.is_prediction_received = False
        self.prediction_data = None

    def prediction_info_callback(self, msg):
        rospy.loginfo("Received prediction msg data")
        self.is_prediction_received = True
        self.prediction_data = msg
        self.publish_visualization_data(msg)

    def publish_visualization_data(self, prediction):
        intention_markers = MarkerArray()
        maneuver_marker = Marker(
            type=Marker.TEXT_VIEW_FACING,
            header=prediction.header,
            id=prediction.unique_id,
            text=prediction.maneuver,
            scale=Vector3(0.2, 0.2, 0.2),
            color=ColorRGBA(1.0, 0.0, 0.0, 1.0)
        )
        intention_markers.markers.append(maneuver_marker)

        maneuver_markers = MarkerArray()
        probability_marker = Marker(
            type=Marker.CUBE,
            header=prediction.header,
            id=prediction.unique_id,
            text=str(prediction.probability),
            scale=Vector3(0.2, 0.2, prediction.probability),
            color=ColorRGBA(0.0, 1.0, 0.0, 1.0)
        )

        maneuver_markers.markers.append(probability_marker)
        self.intention_publisher.publish(intention_markers)
        self.maneuver_publisher.publish(maneuver_markers)
if __name__ == '__main__':
    try:
        node=HMMPredictionVisualizeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass