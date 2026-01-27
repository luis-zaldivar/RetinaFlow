from PIL import Image
from tkinter import Tk, filedialog
from retinaface import RetinaFace
import cv2


def PathImagen():
    root = Tk()
    root.withdraw()
    ruta_imagen = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if ruta_imagen:
        return ruta_imagen
    else:
        return None


def detectar_rostros(ruta_imagen):
    # Carga la imagen
    img = cv2.imread(ruta_imagen)
    
    # Detecta rostros
    faces = RetinaFace.detect_faces(img)
    
    # Imprime información de los rostros detectados
    for key, face in faces.items():
        print(f"Rostro {key}: {face['facial_area']}")
    
    # Dibuja rectángulos en la imagen (opcional)
    for face in faces.values():
        facial_area = face['facial_area']
        cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 0, 0), 2)
    
    # Muestra la imagen
    cv2.imshow('Rostros detectados', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


PathFoto = PathImagen()
detectar_rostros(PathFoto)