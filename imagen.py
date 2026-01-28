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


def recortar_rostros(img, coordenadas):
    # Si no hay coordenadas, retorna lista vacía
    if not coordenadas:
        return []
    
    # Lista para guardar las imágenes recortadas
    rostros_recortados = []
    
    # Obtener dimensiones de la imagen
    altura, ancho = img.shape[:2]
    
    # Recortar cada rostro según sus coordenadas (x_min, y_min, x_max, y_max) con 10% de padding
    for i, (x_min, y_min, x_max, y_max) in enumerate(coordenadas, 1):
        # Calcular el ancho y alto del rostro
        ancho_rostro = x_max - x_min
        alto_rostro = y_max - y_min
        
        # Calcular el padding (10%)
        padding_x = int(ancho_rostro * 0.1)
        padding_y = int(alto_rostro * 0.1)
        
        # Aplicar padding, asegurando que no salga de los límites de la imagen
        x_min_padding = max(0, x_min - padding_x)
        y_min_padding = max(0, y_min - padding_y)
        x_max_padding = min(ancho, x_max + padding_x)
        y_max_padding = min(altura, y_max + padding_y)
        
        # Recortar con padding
        rostro = img[y_min_padding:y_max_padding, x_min_padding:x_max_padding]
        rostros_recortados.append(rostro)
        
        # Guardar cada rostro como una imagen nueva
        nombre_archivo = f"rostro_{i}.jpg"
        cv2.imwrite(nombre_archivo, rostro)
        print(f"Imagen guardada: {nombre_archivo}")
    
    return rostros_recortados

if __name__ == "__main__":
    RutaImagen = PathImagen()
    if RutaImagen:
        ImaCargada= cargar_imagen(RutaImagen)
        coordenadas = detectar_rostros(ImaCargada)
        if coordenadas:
            recortar_rostros(ImaCargada, coordenadas)
        else:
            print("No hay rostros en la imagen.")
    else:
        print("No se seleccionó ninguna imagen.")