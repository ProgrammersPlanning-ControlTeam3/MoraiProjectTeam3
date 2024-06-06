#!/usr/bin/env python3
# written by zozibush

import rospy

from morai_msgs.msg import CtrlCmd

vehicle_control = CtrlCmd()

def pub_motor():
	global pub
	global vehicle_control

	vehicle_control.longlCmdType = 1
	vehicle_control.accel = 1.0

	pub.publish(vehicle_control)

def main():
	print("running")
	global pub

	rospy.init_node('send_topic')
	pub = rospy.Publisher('/ctrl_cmd_0', CtrlCmd, queue_size = 1)
	rate = rospy.Rate(50)

	while not rospy.is_shutdown():
		pub_motor()
		rate.sleep()


if __name__ == '__main__':
	main()