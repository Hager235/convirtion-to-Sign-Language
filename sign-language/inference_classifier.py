import pickle

import cv2
import mediapipe as mp
import numpy as np
import numpy as np

from matplotlib import pyplot as plt
import time
import mediapipe as mp



model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)




def draw_styled_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
        




###################################################################################################




def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])



def mediapipe_detection(image, model):
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False
    if(image is None): 
       print("fi") 

    else:                # Image is no longer writeable
     results = model.process(image)

                      # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results






labels_dict = {0: 'family', 1: '', 2: 'father',3:'mother',4:'brother',
               5:'sister',6:'grandfather',7:'grandmother',8:'son',9:'baby',10:'hurts',
               11:'satiate',12:'thirsty',13:'book',14:'came',15:'enter',16:'play',17:'stood up',18:'raise',
               19:'send',20:'gave',21:'thank',22:'taking',23:'delete',24:'hit',25:'die',
               26:'escape',27:'hello',28:'how are you',29:'good morning',30:'good evening',31:'congrats',
               32:'please',33:'what is time',34:'where is place',35:'happy',36:'sad',37:'sorry',38:'worry',39:'good',40:'bad'}
labels = []
mp_holistic = mp.solutions.holistic # Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
 while True:

    ret, frame = cap.read()

        # Make detections
    if(ret): image, results = mediapipe_detection(frame, holistic)
        # print(results)
    else: break
        # Draw landmarks
    draw_styled_landmarks(image, results)
    keypoints = extract_keypoints(results)

        # Show to screen
    
        #frames.append(results)


    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(
    #             frame,  # image to draw
    #             hand_landmarks,  # model output
    #             mp_hands.HAND_CONNECTIONS,  # hand connections
    #             mp_drawing_styles.get_default_hand_landmarks_style(),
    #             mp_drawing_styles.get_default_hand_connections_style())

    #     for hand_landmarks in results.multi_hand_landmarks:
    #         for i in range(len(hand_landmarks.landmark)):
    #             x = hand_landmarks.landmark[i].x
    #             y = hand_landmarks.landmark[i].y

    #             x_.append(x)
    #             y_.append(y)

    #         for i in range(len(hand_landmarks.landmark)):
    #             x = hand_landmarks.landmark[i].x
    #             y = hand_landmarks.landmark[i].y
    #             data_aux.append(x - min(x_))
    #             data_aux.append(y - min(y_))

    #     x1 = int(min(x_) * W) - 10
    #     y1 = int(min(y_) * H) - 10

    #     x2 = int(max(x_) * W) - 10
    #     y2 = int(max(y_) * H) - 10

    prediction = model.predict([np.asarray(keypoints)])

    predicted_character = labels_dict[int(prediction[0])]

    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
    #                 cv2.LINE_AA)
    cv2.rectangle(image, (0,430), (640, 480), (255, 255, 255), -1)
    cv2.putText(image, ' '.join(predicted_character), (300,460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # Break gracefully
    cv2.imshow('OpenCV Feed', image)    


    
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
