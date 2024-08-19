import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

danish = cv2.VideoCapture(0)

while True:
 
    ret, frames = danish.read()
    
    if not ret:
        print("Failed to grab frame")
        break
 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
      
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    
    cv2.imshow('Face Detection', frames)
    
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

danish.release()
cv2.destroyAllWindows()
