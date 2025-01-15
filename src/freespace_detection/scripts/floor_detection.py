#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import numpy as np

class RoiChecker:
    def __init__(self):
        rospy.init_node('roi_checker', anonymous=True)

        # Publisher
        self.roi_pub = rospy.Publisher('/roi_check', Bool, queue_size=10)

        # Subscriber
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        # YOLO model
        self.model = YOLO("/home/cgvlab/catkin_ws/src/freespace_detection/scripts/best.pt")

        # ROI Definition
        self.roi_x_min, self.roi_y_min, self.roi_x_max, self.roi_y_max = 140, 240, 180, 250
        self.width = self.roi_x_max - self.roi_x_min
        self.height = self.roi_y_max - self.roi_y_min
        self.roi_area = self.width * self.height

        # CV Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        rospy.loginfo("ROI Checker initialized.")

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        # Run YOLO model on the frame
        results = self.model(frame, stream=True, conf=0.5)

        in_roi_80_percent = False  # Default value

        # Process results
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

                for mask in masks:
                    y_indices, x_indices = np.where(mask)

                    # Count mask pixels within ROI
                    within_roi = ((x_indices >= self.roi_x_min) & (x_indices <= self.roi_x_max) &
                                  (y_indices >= self.roi_y_min) & (y_indices <= self.roi_y_max))
                    mask_area = within_roi.sum()

                    # Check if 80% or more of the mask area is within the ROI
                    if mask_area >= self.roi_area * 0.8:
                        in_roi_80_percent = True
                        break

        # Publish the result
        self.roi_pub.publish(Bool(data=in_roi_80_percent))

        # Show the result for debugging purposes
        cv2.putText(frame, f"In ROI 80% or more: {in_roi_80_percent}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if in_roi_80_percent else (0, 0, 255), 2)
        cv2.imshow("Segmentation Result", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown.")

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        roi_checker = RoiChecker()
        roi_checker.run()
    except rospy.ROSInterruptException:
        pass
