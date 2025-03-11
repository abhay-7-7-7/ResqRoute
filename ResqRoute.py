import numpy as np
import math
import cv2
import streamlit as st
from PIL import Image

# Function to process the image and detect contours
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            hull = cv2.convexHull(cnt)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            arearatio = ((areahull - areacnt) / areacnt) * 100

            # Check if the contour and hull are valid
            if len(approx) >= 4 and len(hull) >= 3:  # Ensure enough points for convexityDefects
                hull = cv2.convexHull(approx, returnPoints=False)
                try:
                    defects = cv2.convexityDefects(approx, hull)
                except cv2.error as e:
                    st.warning(f"Convexity defects error: {e}")
                    continue

                if objCor == 12 and arearatio > 30:
                    l = 0
                    if defects is not None:  # Check if defects were computed
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(approx[s][0])
                            end = tuple(approx[e][0])
                            far = tuple(approx[f][0])

                            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                            s = (a + b + c) / 2
                            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
                            d = (2 * ar) / a

                            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 55
                            if 75 <= angle <= 105 and d > 7:
                                l += 1

                    if l == 4:
                        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
                        aspRatio = w / float(h)
                        if 0.95 < aspRatio < 1.05:
                            objectType = "Detected !!!"
                            cv2.putText(imgContour, objectType, (x + w // 2 - 50, y + h + 25),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 255, 0), 2)

# Streamlit UI
st.title("AMBULANCE DETECTION SYSTEM")
st.write("Every second counts - every route optimized")
st.write("""
Every second counts in a medical emergency, yet countless lives are lost due to delayed ambulance arrivals caused by traffic congestion. 
ResQRoute is an AI-powered system designed to detect ambulances in traffic, create a clear pathway, and ensure they reach their destination as quickly as possible.
""")


# Upload image or use webcam
option = st.radio("Choose input source:", ("Upload an image", "Use webcam"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        imgContour = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0.5)
        imgCanny = cv2.Canny(imgBlur, 50, 50)
        getContours(imgCanny, imgContour)

        st.image(imgContour, caption="Processed Image", use_container_width=True)

elif option == "Use webcam":
    st.write("Press the button below to start webcam capture.")
    start_webcam = st.button("Start Webcam")

    if start_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_webcam = st.button("Stop Webcam")

        while cap.isOpened() and not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            img = cv2.flip(frame, 1)
            imgContour = img.copy()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0.5)
            imgCanny = cv2.Canny(imgBlur, 50, 50)
            getContours(imgCanny, imgContour)

            # Display the processed frame in Streamlit
            stframe.image(imgContour, channels="BGR", use_container_width=True)

            # Check if the "Stop Webcam" button is pressed
            if stop_webcam:
                break

        # Release the webcam and clean up
        cap.release()
        cv2.destroyAllWindows()
        st.write('''Webcam stopped we are facing technical issues please run the code locally 
                 please try uploading thhe image or try after some time''')