#from PIL import Image
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


def cargar_imagen(ruta_imagen):
    # Carga la imagen usando OpenCV
    img = cv2.imread(ruta_imagen)
    
    # Verifica si la imagen se cargó correctamente
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    return img


def detectar_rostros(img):
    # Detecta rostros
    faces = RetinaFace.detect_faces(img)
    
    # Si hay rostros detectados, retorna las coordenadas
    if faces:
        # Retorna las coordenadas de los rostros detectados como lista de tuplas (x_min, y_min, x_max, y_max)
        coordenadas = [tuple(int(coord) for coord in face['facial_area']) for face in faces.values()]
        return coordenadas
    else:
        return []
    
    

if __name__ == "__main__":
    RutaImagen = PathImagen()
    if RutaImagen:
        ImaCargada= cargar_imagen(RutaImagen)
        coordenadas = detectar_rostros(ImaCargada)
        if coordenadas:
            print("Rostros detectados:", coordenadas)
        else:
            print("No hay rostros en la imagen.")
    else:
        print("No se seleccionó ninguna imagen.")