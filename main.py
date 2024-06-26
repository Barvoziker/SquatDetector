import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz

# Initialisation de MediaPipe Pose et Hands.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialisation de MediaPipe Drawing pour dessiner les points de pose et des mains.
mp_drawing = mp.solutions.drawing_utils

# Ouvrir la caméra virtuelle.
cap = cv2.VideoCapture(0)  # Assurez-vous que la caméra virtuelle est configurée comme caméra par défaut.

# Fuseau horaire français
paris_tz = pytz.timezone('Europe/Paris')

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
            writer.writerow(['Date', 'Squats', 'Duree', 'Calories Brulees'])
        writer.writerow(data)

# Fonction pour lire les données du fichier CSV
def read_csv(filename='squat_history.csv'):
    data = []
    if os.path.isfile(filename):
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                data.append(row)
    return data

# Fonction pour générer un graphique
def generate_graph(data, filename='squat_graph.png'):
    timestamps = [datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc).astimezone(paris_tz) for row in data]
    squats = [int(row[1]) for row in data]
    calories = [float(row[3]) for row in data]

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))

    ax[0].plot(timestamps, squats, label='Squats')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Squats')
    ax[0].legend()
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M:%S'))
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].plot(timestamps, calories, label='Calories Brulees', color='orange')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Calories Brulees')
    ax[1].legend()
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M:%S'))
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Fonction pour vérifier si une main est dans la zone de pause
def is_hand_in_pause_zone(hand_landmarks, pause_zone):
    for landmark in hand_landmarks.landmark:
        if pause_zone[0] <= landmark.x <= pause_zone[2] and pause_zone[1] <= landmark.y <= pause_zone[3]:
            return True
    return False

# Demander le poids de l'utilisateur.
poids = float(input("Entrez votre poids en kilogrammes : "))

# Estimation des calories brûlées (MET pour les squats modérés à intenses).
MET = 5.0  # Vous pouvez ajuster ce nombre selon l'intensité

# Initialiser les variables pour le comptage des squats.
counter = 0
stage = None
squat_in_progress = False
min_up_duration = 10
min_down_duration = 10
up_frame_count = 0
down_frame_count = 0
timer_started = False
timer_active = False  # Variable pour suivre l'état d'activation du timer
hand_in_pause_zone = False  # Variable pour suivre si la main est dans la zone de pause
start_time = 0
elapsed_time = 0
calories_brulees = 0
session_counter = 1  # Variable pour suivre le numéro de la session

# Définir la zone de pause (x1, y1, x2, y2) en pourcentage de la taille de l'image.
pause_zone = (0.05, 0.8, 0.15, 0.9)

# Créer une fenêtre nommée et définir sa taille.
cv2.namedWindow('Squat Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Squat Detection', 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo")
        break

    height, width, _ = frame.shape
    x1, y1, x2, y2 = int(pause_zone[0] * width), int(pause_zone[1] * height), int(pause_zone[2] * width), int(pause_zone[3] * height)

    # Convertir l'image en RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processer l'image pour détecter les poses et les mains.
    results_pose = pose.process(image)
    results_hands = hands.process(image)

    # Convertir l'image en BGR pour l'affichage.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dessiner la zone de pause.
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Afficher le compteur de squats et le timer sur la même ligne.
    if timer_started:
        elapsed_time = time.time() - start_time
    timer_text = f'Timer: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}'
    counter_text = f'Compteur: {counter}'
    calories_text = f'Calories: {calories_brulees:.2f}'
    session_text = f'Session: {session_counter}'
    cv2.putText(image, counter_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, timer_text, (300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if timer_started else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, calories_text, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, session_text, (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Dessiner les points de pose si des poses sont détectées.
    if results_pose.pose_landmarks and timer_active:
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
                    tuple(np.multiply(knee, [width, height]).astype(int)),
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
            cv2.putText(image, 'Up', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif stage == 'down':
            cv2.putText(image, 'Down', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Dessiner les points de main si des mains sont détectées.
    hand_in_zone = False
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_hand_in_pause_zone(hand_landmarks, pause_zone):
                hand_in_zone = True

    if hand_in_zone and not hand_in_pause_zone:
        hand_in_pause_zone = True
        if timer_started:
            # Enregistrer les données de la session dans un fichier CSV
            timestamp = datetime.now(pytz.utc).astimezone(paris_tz).strftime("%Y-%m-%d %H:%M:%S")
            data = [timestamp, counter, int(elapsed_time), calories_brulees]
            save_to_csv(data)
            all_data = read_csv()
            generate_graph(all_data)
            timer_started = False
            timer_active = False
            counter = 0
            elapsed_time = 0
            calories_brulees = 0
            session_counter += 1
        else:
            timer_started = True
            timer_active = True
            start_time = time.time()
    elif not hand_in_zone:
        hand_in_pause_zone = False

    # Afficher l'image avec les annotations.
    cv2.imshow('Squat Detection', image)

    # Vérifier si la touche "q" est pressée pour arrêter le programme.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Enregistrer les données de la session dans un fichier CSV à la fin du programme
if timer_started:
    timestamp = datetime.now(pytz.utc).astimezone(paris_tz).strftime("%Y-%m-%d %H:%M:%S")
    data = [timestamp, counter, int(elapsed_time), calories_brulees]
    save_to_csv(data)

# Générer le graphique avec toutes les données
all_data = read_csv()
if all_data:
    generate_graph(all_data)

# Libérer les ressources.
cap.release()
cv2.destroyAllWindows()
