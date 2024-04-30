import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to detect hand landmarks
def detect_hand_landmarks(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Check if hand(s) detected
    if results.multi_hand_landmarks:
        # Iterate through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            for landmark in hand_landmarks.landmark:
                # Get the coordinates of the landmark
                height, width, _ = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # Draw a small circle at the landmark position
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

# Function to detect hand gestures
def detect_hand_gestures(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Check if hand(s) detected
    if results.multi_hand_landmarks:
        # Iterate through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions
            landmarks = {}
            for i, landmark in enumerate(hand_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y, landmark.z]
            
            # Detect thumbs up gesture
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            if thumb_tip[1] < index_tip[1]:
                print("Thumbs up gesture detected!")
            
            # Detect thumbs down gesture
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            if thumb_tip[1] > index_tip[1]:
                print("Thumbs down gesture detected!")
            
            # Detect live long (showing palm) gesture
            palm = landmarks[0]
            index_tip = landmarks[8]
            if palm[2] < index_tip[2]:  # Checking if palm is below the index finger
                print("Live long (showing palm) gesture detected!")
                
# Main function
def main():
    # Open camera
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hand landmarks
        detect_hand_landmarks(frame)
        
        # Detect hand gestures
        detect_hand_gestures(frame)
        
        # Display the frame
        cv2.imshow('Hand Gesture Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

