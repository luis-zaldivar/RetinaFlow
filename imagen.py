import cv2
import numpy as np
import os
from tkinter import Tk, filedialog
from retinaface import RetinaFace

# ==========================================
# 0. DETECCIÓN Y REDUCCIÓN DE RUIDO
# ==========================================

def estimar_nivel_ruido(gris):
    """
    Estima el nivel de ruido (ej. ruido de color/pixelación) mediante
    varianza en regiones de alta frecuencia (Laplaciano).
    """
    lap = cv2.Laplacian(gris, cv2.CV_64F)
    return np.var(lap)

def reducir_ruido(rostro, fuerza="auto"):
    """
    Reduce ruido digital preservando bordes. Usa fastNlMeansDenoisingColored
    y opcionalmente bilateral para suavizado final.
    """
    # Parámetros según tamaño del recorte (rostros pequeños = menos filtro)
    h, w = rostro.shape[:2]
    area = h * w
    if area < 80 * 80:
        h_filter, template, search = 6, 5, 11
    elif area < 200 * 200:
        h_filter, template, search = 8, 6, 13
    else:
        h_filter, template, search = 10, 7, 21

    denoised = cv2.fastNlMeansDenoisingColored(
        rostro, None, h_filter, h_filter * 2, template, search
    )
    # Bilateral suave para suavizar ruido residual sin borrar bordes
    denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
    return denoised

# ==========================================
# 1. DIAGNÓSTICO FÍSICO (TIPO Y SEVERIDAD)
# ==========================================

def analizar_degradacion(gris):
    """
    Detecta si el blur es por movimiento (Motion) o desenfoque (Defocus)
    y mide su severidad mediante FFT y varianza angular.
    Incluye indicador de ruido alto para priorizar denoising.
    """
    # --- 0. Nivel de ruido ---
    nivel_ruido = estimar_nivel_ruido(gris)
    ruido_alto = nivel_ruido > 800  # Umbral empírico para ruido visible

    # --- 1. Dirección del gradiente ---
    gx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    angulos = cv2.phase(gx, gy)
    varianza_angular = np.var(angulos)
    
    # --- 2. Severidad (FFT) ---
    f = np.fft.fft2(gris)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    ch, cw = gris.shape[0]//2, gris.shape[1]//2
    r = int(min(gris.shape) * 0.1)
    ratio_blur = np.mean(mag[ch-r:ch+r, cw-r:cw+r]) / (np.mean(mag) + 1e-6)

    # Clasificación
    tipo = "MOTION" if varianza_angular < 1.2 else "DEFOCUS"
    if ratio_blur < 1.4: sev = "LEVE"
    elif ratio_blur < 2.3: sev = "MEDIO"
    else: sev = "SEVERO"
    
    return tipo, sev, ratio_blur, ruido_alto

# ==========================================
# 2. PIPELINES DE RESTAURACIÓN ESPECÍFICA
# ==========================================

def restaurar_motion_blur(rostro, severidad):
    """
    Simulación de Deconvolución Wiener simplificada.
    Usa un kernel de movimiento lineal para revertir el desplazamiento.
    """
    # Enfoque direccional (Kernel de movimiento aproximado)
    size = 5 if severidad == "LEVE" else 9
    kernel_motion = np.zeros((size, size))
    kernel_motion[int((size-1)/2), :] = np.ones(size)
    kernel_motion /= size
    
    # Deconvolución iterativa (Lucy-Richardson simplificada en CPU)
    deconvolved = np.copy(rostro)
    for _ in range(3): # Pocas iteraciones para evitar ruido en CPU
        blur = cv2.filter2D(deconvolved, -1, kernel_motion)
        ratio = rostro / (blur + 1e-6)
        deconvolved *= cv2.filter2D(ratio, -1, kernel_motion)
    
    return np.clip(deconvolved, 0, 255).astype(np.uint8)

def restaurar_defocus_blur(rostro, severidad):
    """
    Estrategia de reconstrucción isotrópica:
    Super-resolución matemática + Enfoque de máscara de bordes.
    """
    h, w = rostro.shape[:2]
    # Reconstrucción de bordes mediante Lanczos
    temp = cv2.resize(rostro, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    
    # Sharpen adaptativo (Máscara de bordes)
    gris = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gris, 50, 150)
    mask = cv2.GaussianBlur(mask, (5, 5), 0) / 255.0
    
    suave = cv2.GaussianBlur(temp, (0, 0), 2)
    enfocado = cv2.addWeighted(temp, 1.8, suave, -0.8, 0)
    
    # Blend basado en la máscara de bordes
    for c in range(3):
        temp[:,:,c] = (enfocado[:,:,c] * mask + temp[:,:,c] * (1 - mask))
        
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)

# ==========================================
# 3. MEJORA DE CONTRASTE (CLAHE)
# ==========================================

def mejorar_contraste(rostro, clip_limit=2.0, grid=(8, 8)):
    """
    Aplica CLAHE en LAB para realzar contraste sin afectar color,
    haciendo la mejora más visible.
    """
    lab = cv2.cvtColor(rostro, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ==========================================
# 4. CONTROL DE CALIDAD Y HALOS
# ==========================================

def evaluar_resultado(orig, mejorado, aplicamos_denoising=False):
    """
    Valida la mejora: acepta si hay ganancia de nitidez.
    No rechaza por cambio global grande cuando hubo denoising (ruido).
    Solo penaliza oversharpening extremo (halos).
    """
    g_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    g_mejo = cv2.cvtColor(mejorado, cv2.COLOR_BGR2GRAY)
    
    s_orig = cv2.Laplacian(g_orig, cv2.CV_64F).var()
    s_mejo = cv2.Laplacian(g_mejo, cv2.CV_64F).var()
    
    # Halos: nitidez excesiva (ratio muy alto) indica oversharpening
    if s_orig > 1e-6 and s_mejo > 5 * s_orig:
        return 0  # Rechazar solo por oversharpening extremo
    
    # Si aplicamos denoising, no rechazar por diff grande (el cambio es esperado)
    if not aplicamos_denoising:
        diff = cv2.absdiff(g_mejo, g_orig)
        if np.mean(diff) > 55:
            return 0
        
    return s_mejo - s_orig

# ==========================================
# 5. MOTOR PRINCIPAL
# ==========================================

if __name__ == "__main__":
    Tk().withdraw()
    ruta = filedialog.askopenfilename()
    
    if ruta:
        img = cv2.imread(ruta)
        if img is None:
            print("No se pudo cargar la imagen.")
            exit(1)
        img_final = img.copy()
        faces = RetinaFace.detect_faces(img)
        
        if faces:
            for i, f in enumerate(faces.values()):
                x1, y1, x2, y2 = [int(c) for c in f['facial_area']]
                recorte = img[y1:y2, x1:x2].copy()
                gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                
                # 1. Diagnóstico (incluye si hay ruido alto)
                tipo, sev, score, ruido_alto = analizar_degradacion(gris)
                print(f"\nRostro {i+1}: {tipo} | Severidad {sev}" + (" | Ruido alto" if ruido_alto else ""))
                
                # 2. Denoising primero (siempre en rostros; más importante si ruido_alto)
                aplicamos_denoising = True
                trabajo = reducir_ruido(recorte)
                gris_t = cv2.cvtColor(trabajo, cv2.COLOR_BGR2GRAY)
                
                # 3. Re-diagnóstico post-denoising para blur
                tipo_t, sev_t, _, _ = analizar_degradacion(gris_t)
                
                # 4. Pipeline de blur sobre imagen ya denoised
                if tipo_t == "MOTION":
                    candidato = restaurar_motion_blur(trabajo, sev_t)
                else:
                    candidato = restaurar_defocus_blur(trabajo, sev_t)
                
                # 5. Mejora de contraste para que la mejora se note más
                candidato = mejorar_contraste(candidato, clip_limit=2.0)
                
                # 6. Validar y reinsertar (no rechazar por diff cuando hubo denoising)
                if evaluar_resultado(recorte, candidato, aplicamos_denoising=aplicamos_denoising) > 0:
                    print("   ✨ Mejora validada e insertada.")
                    img_final[y1:y2, x1:x2] = cv2.resize(candidato, (x2-x1, y2-y1))
                else:
                    # Si rechazamos por halos, al menos insertar la versión denoised + CLAHE
                    fallback = mejorar_contraste(trabajo, clip_limit=1.8)
                    print("   ⚠️ Halos detectados; se usa versión suave (denoise + contraste).")
                    img_final[y1:y2, x1:x2] = cv2.resize(fallback, (x2-x1, y2-y1))
            
            cv2.imwrite("RESTORE_PRO.jpg", img_final)
            print("\nProceso terminado. Archivo: RESTORE_PRO.jpg")
        else:
            print("No se detectaron rostros en la imagen.")