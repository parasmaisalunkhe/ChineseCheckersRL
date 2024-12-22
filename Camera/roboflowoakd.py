from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np

if __name__ == '__main__':
    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="ChineseCheckers", confidence=0.05,
    overlap=0.5, version="5",
    api_key="xf0hNfp2MXJ7Jb6B1ZHL", rgb=True, depth=True,
    device=None, blocking=True)
    while True:
        t0 = time.time()
        result, frame, raw_frame, depth = rf.detect()    
        predictions = result["predictions"]
        #{
        #    predictions:
        #    [ {
        #        x: (middle),
        #        y:(middle),
        #        width: ,
        #        height: ,
        #        depth: ###->,
        #        confidence: ,
        #        class: ,
        #        mask: { }
        #       }
        #    ]
        #}
        #frame - frame after preprocs, with predictions
        #raw_frame - original frame from your OAK
        #depth - depth map for raw_frame, center-rectified
        # to the center camera 
        # To access specific values within "predictions" use:
        # p.json()[a] for p in predictions
        # set "a" to the index you are attempting to access
        # Example: accessing the "y"-value:
        # p.json()['y'] for p in predictions
        
        t = time.time()-t0
        print("INFERENCE TIME IN MS ", 1/t)
        print("PREDICTIONS ", [p.json() for p in predictions])
        
        # setting parameters for depth calculation
        # comment out the following 2 lines out if you're using an OAK
        # without Depth
        max_depth = np.amax(depth)
        cv2.imshow("depth", depth/max_depth)
        # displaying the video feed as successive frames      
        cv2.imshow("frame", frame)
        
        # how to close the OAK inference window/stop inference:
        # CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break