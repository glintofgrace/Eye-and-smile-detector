# Eye-and-smile-detector
# Detects the average no. of eyes and smiles in the span of 30 seconds
import cv2
import time

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

start_time = time.time()
frame_count = 0
faces_per_frame = []
eyes_per_frame = []
smiles_per_frame = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    total_eyes = 0
    total_smiles = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        total_eyes += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        total_smiles += len(smiles)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    # Display counts on frame
    cv2.putText(frame, f"Faces: {len(faces)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Eyes: {total_eyes}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Smiles: {total_smiles}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save stats
    faces_per_frame.append(len(faces))
    eyes_per_frame.append(total_eyes)
    smiles_per_frame.append(total_smiles)
    frame_count += 1

    cv2.imshow("People, Eyes, and Smile Counter", frame)

    # Stop after 30 seconds
    if time.time() - start_time > 30:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print summary
print(f"Total frames processed: {frame_count}")
print(f"Faces per frame: {faces_per_frame}")
print(f"Eyes per frame: {eyes_per_frame}")
print(f"Smiles per frame: {smiles_per_frame}")
print(f"Average faces per frame: {sum(faces_per_frame)/frame_count:.2f}")
print(f"Average eyes per frame: {sum(eyes_per_frame)/frame_count:.2f}")
print(f"Average smiles per frame: {sum(smiles_per_frame)/frame_count:.2f}")
[EYES AND SMILES.ipynb](https://github.com/user-attachments/files/22638692/EYES.AND.SMILES.ipynb)
