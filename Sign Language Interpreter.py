import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Define gesture recognition logic
                if results.multi_handedness[0].classification[0].label == 'Right':
                    if results.multi_hand_landmarks[0].landmark[4].y < results.multi_hand_landmarks[0].landmark[3].y:
                        cv2.putText(image, 'Hii', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif results.multi_hand_landmarks[0].landmark[8].y < results.multi_hand_landmarks[0].landmark[7].y:
                        cv2.putText(image, 'Peace', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[11].y:
                        cv2.putText(image, 'How are you', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    elif results.multi_hand_landmarks[0].landmark[16].y < results.multi_hand_landmarks[0].landmark[15].y:
                        cv2.putText(image, 'Goodbye', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[3].y:
                        cv2.putText(image, 'Great job', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(image, 'Stop', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    # Additional condition for a generic hand pose (no specific gesture)
                    cv2.putText(image, 'Wrong gesture', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
        else:
            cv2.putText(image, 'No hand detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Interpreter', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
