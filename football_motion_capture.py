import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawiing_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose_detector = mp_pose.Pose()


cap = cv2.VideoCapture("football_skills.mp4")


while True:
    success, img = cap.read()

    img = cv2.resize(img, (620, 520))

    if not success:
        break

    # preprocess the image
    results = pose_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        continue

    mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        # landmark_drawings_spec=mp_drawiing_style.get_default_pose_landmarks_style()

    )

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
