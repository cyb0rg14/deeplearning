import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
def load_gesture_model(model_path):
    return load_model(model_path)

# Load class names
def load_class_names(names_path):
    with open(names_path, 'r') as f:
        class_names = f.read().split('\n')
    return class_names

# Initialize the webcam
def initialize_webcam():
    return cv2.VideoCapture(0)

# Function to detect hand gestures
def detect_hand_gesture(cap, model, class_names):
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()

        if frame is None:
            break

        x, y, _ = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = class_names[classID]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam
    cap.release()

    cv2.destroyAllWindows()


# Main function
def main():
    model_path = 'mp_hand_gesture'
    names_path = 'gesture.names'

    model = load_gesture_model(model_path)
    class_names = load_class_names(names_path)
    cap = initialize_webcam()

    gesture = detect_hand_gesture(cap, model, class_names)
    print(gesture)

if __name__ == '__main__':
    main()

