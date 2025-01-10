import cv2
import sys
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with the correct IP Webcam URL from your phone
# Ensure you're using http:// and the correct IP and port
ip_webcam_url = "http://192.168.137.200:8080/video"  # Example URL

logger.info(f"Attempting to connect to IP Webcam stream at {ip_webcam_url}")

# Open the IP Webcam stream using OpenCV
cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    logger.error("Error: Unable to connect to the IP Webcam stream.")
    sys.exit()

logger.info("Successfully connected to the IP Webcam stream.")

while True:
    # Read frame from the stream
    ret, frame = cap.read()
    if not ret:
        logger.error("Error: Failed to retrieve frame.")
        break

    # Display the frame
    cv2.imshow('IP Webcam Stream', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Exit key pressed. Exiting...")
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
logger.info("Stream closed and resources released.")
