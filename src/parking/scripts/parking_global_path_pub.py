#!/usr/bin/env python3

import hybrid_a_star
import rospy
import sys
import os
import copy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from lib.mgeo.class_defs import *


class hybrid_a_star_pub :
    def __init__(self):
        rospy.init_node('dijkstra_path_pub', anonymous=True)

        self.global_path_pub = rospy.Publisher('/global_path',Path, queue_size = 1)

        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = '/map'

        _, self.global_path_msg = hybrid_a_star.hybrid_a_star()

        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown() :
            self.global_path_pub.publish(self.global_path_msg)
            rate.sleep()

def main() :
    hybrid = hybrid_a_star_pub()
    # print(hybrid)

if __name__ == "__main__":
    main()