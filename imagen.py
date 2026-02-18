import cv2
import numpy as np
import os
from tkinter import Tk, filedialog
from retinaface import RetinaFace

# ==========================================
# 1. MÉTRICAS AVANZADAS Y DIMENSIONES
# ==========================================

def estimar_ruido_real(gris):
    """Estima el ruido ignorando bordes (variación fina en zonas planas)."""
    mediana = cv2.medianBlur(gris, 3)
    residuo = cv2.absdiff(gris, mediana)
    return np.percentile(residuo, 50)

def obtener_metricas(rostro, tam_original):
    h, w = rostro.shape[:2]
    area = w * h
    porcentaje = (area / (tam_original[0] * tam_original[1])) * 100
    
    gris = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
    val_n = cv2.Laplacian(gris, cv2.CV_64F).var()
    val_r = estimar_ruido_real(gris)
    val_c = gris.std()

    clasif = {
        "tamaño": "GRANDE" if porcentaje > 15 else ("MEDIO" if porcentaje >= 7 else "PEQUEÑO"),
        "nitidez": "ALTA" if val_n > 300 else ("MEDIA" if val_n >= 120 else "BAJA"),
        "ruido": "BAJO" if val_r < 3 else ("MEDIO" if val_r <= 6 else "ALTO"),
        "contraste": "ALTO" if val_c > 50 else ("MEDIO" if val_c >= 30 else "BAJO")
    }
    
    return {
        "raw": {"n": val_n, "r": val_r, "c": val_c},
        "clasificacion": clasif
    }

def evaluar_dimensiones(c):
    biometrico = (c["tamaño"] != "PEQUEÑO" and c["nitidez"] == "ALTA" and c["ruido"] != "ALTO")
    estetico = (c["contraste"] == "ALTO" and c["ruido"] == "BAJO")
    restaurable = not (c["tamaño"] == "PEQUEÑO" and c["nitidez"] == "BAJA")
    return {"biometrico": biometrico, "estetico": estetico, "restaurable": restaurable}

# ==========================================
# 2. MOTOR DE MEJORA Y REGLA DE ORO
# ==========================================

def validar_mejora_estricta(m_old, m_new):
    """Política de protección: prohíbe regresiones y exige ganancias reales."""
    regresion_ruido = m_new["r"] > (m_old["r"] * 1.05)
    regresion_nitidez = m_new["n"] < (m_old["n"] * 0.95)
    
    if regresion_ruido or regresion_nitidez:
        return False, "Regresión detectada (daño a la integridad)"
    
    ganancia_n = m_new["n"] > (m_old["n"] * 1.10)
    ganancia_r = m_new["r"] < (m_old["r"] * 0.90)
    
    if ganancia_n or ganancia_r:
        return True, "Mejora técnica validada"
    
    return False, "Cambio insignificante"

def ejecutar_mejora_clasica(rostro, res_ini, tam_orig):
    temp = rostro.copy()
    m_old = res_ini["raw"]
    c_old = res_ini["clasificacion"]
    
    # 1. Denoise sutil (Solo si es necesario)
    if c_old["ruido"] == "ALTO":
        temp = cv2.fastNlMeansDenoisingColored(temp, None, 3, 3, 7, 21)
    
    # 2. Sharpen adaptativo
    if c_old["nitidez"] != "ALTA":
        gauss = cv2.GaussianBlur(temp, (0, 0), 3)
        p = 0.8 if c_old["nitidez"] == "BAJA" else 0.4
        temp = cv2.addWeighted(temp, 1+p, gauss, -p, 0)
    
    # 3. CLAHE suave (Contraste)
    if c_old["contraste"] != "ALTO":
        lab = cv2.cvtColor(temp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        temp = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    res_new = obtener_metricas(temp, tam_orig)
    apto, mensaje = validar_mejora_estricta(m_old, res_new["raw"])
    
    if apto:
        print(f"   [OK] {mensaje}")
        return temp, res_new["clasificacion"], True
    
    print(f"   [!] {mensaje}. Rollback ejecutado.")
    return rostro, c_old, False

# ==========================================
# 3. MAIN: DETECCIÓN Y RECONSTRUCCIÓN
# ==========================================

if __name__ == "__main__":
    Tk().withdraw()
    ruta = filedialog.askopenfilename(title="Seleccionar Imagen")
    
    if ruta:
        img_original = cv2.imread(ruta)
        img_final = img_original.copy()
        h_orig, w_orig = img_original.shape[:2]
        
        print("Buscando rostros...")
        faces = RetinaFace.detect_faces(img_original)
        
        if not faces:
            print("No se detectaron rostros."); exit()

        for i, f in enumerate(faces.values()):
            coords = [int(c) for c in f['facial_area']]
            x1, y1, x2, y2 = coords
            recorte = img_original[y1:y2, x1:x2]
            
            res_ini = obtener_metricas(recorte, (w_orig, h_orig))
            cl_ini = res_ini["clasificacion"]
            dim_ini = evaluar_dimensiones(cl_ini)
            
            print(f"\n--- ROSTRO {i+1} ---")
            print(f"ESTADO: N:{cl_ini['nitidez']} | R:{cl_ini['ruido']} | C:{cl_ini['contraste']}")
            
            # Lógica de decisión (Solo procesamos si no es óptimo pero es restaurable)
            if not dim_ini["biometrico"] and not (cl_ini["tamaño"] == "PEQUEÑO"):
                recorte_mej, cl_fin, exito = ejecutar_mejora_clasica(recorte, res_ini, (w_orig, h_orig))
                
                if exito:
                    # Inserción en la imagen final completa
                    # Redimensionamos para asegurar match perfecto de matriz
                    recorte_mej = cv2.resize(recorte_mej, (x2-x1, y2-y1))
                    img_final[y1:y2, x1:x2] = recorte_mej
                    
                    final_dim = evaluar_dimensiones(cl_fin)
                    status = "✅ APTO" if final_dim["biometrico"] else "⚠️ MEJORADO"
                    print(f"RESULTADO: {status}")
                else:
                    print("RESULTADO: Se mantiene original por falta de ganancia técnica.")
            else:
                print("RESULTADO: No requiere mejora o es demasiado pequeño para restauración clásica.")

        # Guardado del producto final
        output_name = "RESULTADO_RECONSTRUIDO.jpg"
        cv2.imwrite(output_name, img_final)
        print(f"\n{'='*50}\nSISTEMA CERRADO: {output_name} guardado con éxito.\n{'='*50}")