from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
from collections import deque

app = Flask(__name__)

# Camera (Windows fix)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Smoothing
emotion_queue = deque(maxlen=10)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            try:
                result = DeepFace.analyze(
                    face,
                    actions=['emotion'],
                    enforce_detection=False
                )

                emotions = result[0]['emotion']
                dominant = result[0]['dominant_emotion']

                # Smooth output
                emotion_queue.append(dominant)
                final_emotion = max(set(emotion_queue), key=emotion_queue.count)

                # Box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Dominant
                cv2.putText(frame, final_emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                # All emotions
                y_offset = y + h + 20
                for emo, score in emotions.items():
                    text = f"{emo}: {round(score,1)}%"
                    cv2.putText(frame, text, (x, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                    y_offset += 15

            except:
                cv2.putText(frame, "Detecting...", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Convert frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("Starting Emotion Detection Server...")
    app.run(debug=False)