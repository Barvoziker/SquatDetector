import cv2
import mediapipe as mp
import numpy as np

# Initialisation de MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialisation de MediaPipe Drawing pour dessiner les points de pose.
mp_drawing = mp.solutions.drawing_utils

# Ouvrir la caméra virtuelle.
cap = cv2.VideoCapture(0)  # Assurez-vous que la caméra virtuelle est configurée comme caméra par défaut.

def calculate_angle(a, b, c):
    a = np.array(a)  # Premier point
    b = np.array(b)  # Deuxième point
    c = np.array(c)  # Troisième point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo")
        break

    # Convertir l'image en RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processer l'image pour détecter les poses.
    results = pose.process(image)

    # Convertir l'image en BGR pour l'affichage.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dessiner les points de pose si des poses sont détectées.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extraire les coordonnées des hanches, genoux et chevilles.
        landmarks = results.pose_landmarks.landmark
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculer l'angle du genou.
        angle = calculate_angle(hip, knee, ankle)

        # Afficher l'angle sur l'image.
        cv2.putText(image, str(angle),
                    tuple(np.multiply(knee, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )

        # Détecter si c'est un squat.
        if angle > 160:
            cv2.putText(image, 'Up', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif angle < 90:
            cv2.putText(image, 'Down', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Afficher l'image avec les annotations.
    cv2.imshow('Squat Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libérer les ressources.
cap.release()
cv2.destroyAllWindows()
