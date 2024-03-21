from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import numpy as np

client = RemoteAPIClient()
sim = client.getObject('sim')

clientID = sim.startSimulation()

quadcopter_target_handle = sim.getObjectHandle('Quadcopter')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bounds = np.array([100, 150, 0]) 
    upper_bounds = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_bounds, upper_bounds)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 100:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                world_x = ((cx - 640) * 0.001) * 3
                world_y = ((720 - cy) * 0.001) * 3

                sim.setObjectPosition(quadcopter_target_handle, -1, [world_x, -0.525, world_y])

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

sim.stopSimulation()

