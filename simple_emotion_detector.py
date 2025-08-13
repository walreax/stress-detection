import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        result = results[0]
        
        dominant_emotion = result['dominant_emotion']

        # --- MODIFIED PART: Simplify the emotion output ---
        if dominant_emotion == 'happy':
            simple_emotion = 'Happy'
        elif dominant_emotion == 'sad':
            simple_emotion = 'Sad'
        else:
            # Group neutral, angry, fear, etc. into "Neutral"
            simple_emotion = 'Neutral'
        # --- END OF MODIFIED PART ---

        region = result['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the simplified emotion
        text = f"Status: {simple_emotion}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        pass

    cv2.imshow('Simple Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
