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
    
    return rostros_recortados


def obtener_metricas_rostro(rostro_recortado, tamano_original=None):
    """
    Calcula valores numéricos y genera las etiquetas de clasificación.
    """
    alto, ancho = rostro_recortado.shape[:2]
    area_pixeles = ancho * alto
    
    # --- Cálculos Base ---
    porcentaje_total = 0
    if tamano_original:
        area_original = tamano_original[0] * tamano_original[1]
        porcentaje_total = (area_pixeles / area_original) * 100

    gris = cv2.cvtColor(rostro_recortado, cv2.COLOR_BGR2GRAY)
    
    # Métrica de Nitidez (Varianza de Laplace)
    valor_blur = cv2.Laplacian(gris, cv2.CV_64F).var()
    
    # Métrica de Ruido
    valor_ruido = cv2.meanStdDev(gris)[1][0][0]
    
    # Métrica de Contraste
    valor_contraste = gris.std()

    # --- Clasificación (Lógica de Umbrales) ---
    clasificacion = {
        "tamaño_rostro": "GRANDE" if porcentaje_total > 15 else ("MEDIO" if porcentaje_total >= 7 else "PEQUEÑO"),
        "nitidez": "ALTA" if valor_blur > 300 else ("MEDIA" if valor_blur >= 120 else "BAJA"),
        "ruido": "BAJO" if valor_ruido < 15 else ("MEDIO" if valor_ruido <= 30 else "ALTO"),
        "contraste": "ALTO" if valor_contraste > 45 else ("MEDIO" if valor_contraste >= 25 else "BAJO")
    }

    # Estructura final solicitada
    return {
        "registro_depuracion": {
            "area_px": area_pixeles,
            "porcentaje": round(porcentaje_total, 2),
            "valor_nitidez": round(valor_blur, 2),
            "valor_ruido": round(valor_ruido, 2),
            "valor_contraste": round(valor_contraste, 2)
        },
        "clasificacion": clasificacion
    }

def validar_rostro_para_biometria(objeto_metricas):
    """
    Función que SOLO lee la estructura de clasificación.
    Ideal para decidir si el rostro se guarda o se descarta.
    """
    c = objeto_metricas["clasificacion"]
    
    # Ejemplo de lógica interna de decisión
    es_apto = (
        c["tamaño_rostro"] != "PEQUEÑO" and 
        c["nitidez"] == "ALTA" and 
        c["ruido"] != "ALTO"
    )
    
    return es_apto, c
    
if __name__ == "__main__":
    # 1. Seleccionar y cargar imagen
    RutaImagen = PathImagen()
    
    if RutaImagen:
        ImaCargada = cargar_imagen(RutaImagen)
        # Extraemos dimensiones para el cálculo de porcentaje
        alto_orig, ancho_orig = ImaCargada.shape[:2]
        
        # 2. Detectar rostros
        coordenadas = detectar_rostros(ImaCargada)
        
        if coordenadas:
            # 3. Recortar rostros detectados
            rostros = recortar_rostros(ImaCargada, coordenadas)
            
            print(f"Se encontraron {len(rostros)} rostro(s). Analizando calidad...\n")
            
            for i, rostro_img in enumerate(rostros):
                # 4. Obtener el objeto de métricas y clasificación
                # Pasamos (ancho, alto) como tamano_original
                resultado = obtener_metricas_rostro(rostro_img, (ancho_orig, alto_orig))
                
                # 5. Uso de la función de validación (Solo lee la clasificación)
                apto, clasificacion = validar_rostro_para_biometria(resultado)
                
                # --- SALIDA POR CONSOLA ---
                print(f"=== ANÁLISIS ROSTRO {i+1} ===")
                print(f"Estado: {'✅ APTO' if apto else '❌ NO APTO'}")
                
                # Solo lectura de clasificación (lo que pediste)
                print(f"  > Tamaño:    {clasificacion['tamaño_rostro']}")
                print(f"  > Nitidez:   {clasificacion['nitidez']}")
                print(f"  > Ruido:     {clasificacion['ruido']}")
                print(f"  > Contraste: {clasificacion['contraste']}")
                
                # Los números quedan para tu registro/depuración en el objeto 'resultado'
                # print(f"DEBUG: {resultado['registro_depuracion']}") 
                print("-" * 30)
                
                # Opcional: Mostrar el recorte analizado
                # cv2.imshow(f"Rostro {i+1}", rostro_img)
            
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
        else:
            print("No se detectaron rostros en la imagen seleccionada.")
    else:
        print("Operación cancelada: No se seleccionó ningún archivo.")