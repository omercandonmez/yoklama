import face_recognition
import cv2
import os
import glob
import numpy as np
#face recognition opencv ve numpy kütüphanelerinden faydalandik.

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

       
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        #bu fonksiyonu kullanarak encode edilmiş resimlerin yüklenmesini sağlıyoruz.
        
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Görüntünün bulunduğu yere gidip dosya adını alarak tanınan kişiyi isimlendiriyoruz.
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

           
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

           
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Ekrandaki yüzleri algılayıp isimlendiren ve konumlandıran fonksiyon.
        
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Yüzün eşleşmesinin kontrol edildiği nokta.
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Çerçeveyi hızlı bir şekilde yeniden boyutlandırmak ve koordinatları ayarlamak için numpy dizisine dönüştürülüyor.
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
