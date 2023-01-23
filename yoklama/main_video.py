import cv2
from simple_facerec import SimpleFacerec

# sfr değerine eşitlememiz gerekti encode yapılabilmesi için.
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)
#0 değeri bilgisayarımızın ana kamerasını gösteriyor.

while True:
    ret, frame = cap.read()

    #burada detect_known_faces fonksiyonu bize ekranda gördüğü yüzlerin konumlarını ve isimlerini veriyor.
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        #burada putText konumuyla adını yazdırıyor ardından rectangle komutuyla ekranda algılanan yüze çerçeve ekliyoruz.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
