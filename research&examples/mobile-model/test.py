import cv2
import numpy as np
import imutils
import requests
  
# Replace the below URL with your own IP provided by the IP WEBCAM APP.
# Make sure to add "/shot.jpg" at last.
url = "http://192.168.224.30:8080/shot.jpg"
  
# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=800, height=600)
    cv2.imshow("Android Cam", img)
  
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()