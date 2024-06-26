import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import os

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

# Fonction pour enregistrer les données dans un fichier CSV
def save_to_csv(data, filename='squat_history.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Squats', 'Elapsed Time (s)', 'Calories Burned'])
        writer.writerow(data)

# Demander le poids de l'utilisateur.
poids = float(input("Entrez votre poids en kilogrammes : "))

# Estimation des calories brûlées (MET pour les squats modérés à intenses).
MET = 5.0  # Vous pouvez ajuster ce nombre selon l'intensité
calories_brulees = 0

# Initialiser les variables pour le comptage des squats.
counter = 0
stage = None
squat_in_progress = False
min_up_duration = 10
min_down_duration = 10
up_frame_count = 0
down_frame_count = 0
timer_started = False
start_time = 0
elapsed_time = 0

# Créer une fenêtre nommée et définir sa taille.
cv2.namedWindow('Squat Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Squat Detection', 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo")
        break

    # Convertir l'image en RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processer l'image pour détecter les poses.
    results_pose = pose.process(image)

    # Convertir l'image en BGR pour l'affichage.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Afficher le compteur de squats et le timer sur la même ligne.
    if timer_started:
        elapsed_time = time.time() - start_time
    timer_text = f'Timer: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}'
    counter_text = f'Counter: {counter}'
    calories_text = f'Calories: {calories_brulees:.2f}'
    cv2.putText(image, counter_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, timer_text, (300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if timer_started else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, calories_text, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Dessiner les points de pose si des poses sont détectées.
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extraire les coordonnées des hanches, genoux et chevilles.
        landmarks = results_pose.pose_landmarks.landmark
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculer l'angle du genou.
        angle = calculate_angle(hip, knee, ankle)

        # Afficher l'angle sur l'image.
        cv2.putText(image, str(angle),
                    tuple(np.multiply(knee, [1280, 720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )

        # Détecter si c'est un squat et mettre à jour le compteur avec des durées minimales.
        if angle > 160:
            up_frame_count += 1
            down_frame_count = 0
            if up_frame_count > min_up_duration:
                if squat_in_progress:
                    counter += 1
                    squat_in_progress = False
                    winsound.Beep(1000, 200)  # Son pour un squat réussi
                    calories_brulees += MET * poids * (1 / 60)  # Ajouter les calories pour 1 squat (approx. 1 min)
                stage = "up"
        if angle < 90:
            down_frame_count += 1
            up_frame_count = 0
            if down_frame_count > min_down_duration:
                stage = "down"
                squat_in_progress = True
                if not timer_started:
                    timer_started = True
                    start_time = time.time()

        # Afficher l'état actuel (Up/Down).
        if stage == 'up':
            cv2.putText(image, 'Up', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif stage == 'down':
            cv2.putText(image, 'Down', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Afficher l'image avec les annotations.
    cv2.imshow('Squat Detection', image)

    # Vérifier si la touche "s" est pressée pour mettre en pause et enregistrer les données.
    key = cv2.waitKey(10)
    if key & 0xFF == ord('s'):
        if timer_started:
            # Enregistrer les données de la session dans un fichier CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            data = [timestamp, counter, int(elapsed_time), calories_brulees]
            save_to_csv(data)
        timer_started = False

    if key & 0xFF == ord('q'):
        break

# Enregistrer les données de la session dans un fichier CSV à la fin du programme
if timer_started:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    data = [timestamp, counter, int(elapsed_time), calories_brulees]
    save_to_csv(data)

# Libérer les ressources.
cap.release()
cv2.destroyAllWindows()
