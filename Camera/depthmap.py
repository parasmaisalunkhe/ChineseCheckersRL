import depthai as dai
import cv2
import numpy as np

# Create a pipeline
pipeline = dai.Pipeline()

# Create nodes for mono cameras (left and right) and depth
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Configure the mono cameras
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Set StereoDepth configuration
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # Align depth to RGB camera
stereo.setSubpixel(True)  # Improves depth accuracy

# Link mono cameras to the stereo depth node
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Define XLink output for depth data
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Start the device and pipeline
with dai.Device(pipeline) as device:
    # Output queue to get depth data
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        in_depth = depth_queue.get()  # Get depth frame
        depth_frame = in_depth.getFrame()  # Get depth as a numpy array

        # Normalize depth data for visualization
        depth_frame_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_frame_normalized, cv2.COLORMAP_JET)

        # Display the depth map
        cv2.imshow("Depth Map", depth_colored)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
