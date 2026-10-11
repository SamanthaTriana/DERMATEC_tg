# DERMATEC en la Raspberry Pi 5

Paquete completo de DERMATEC para la Raspberry Pi 5 (Raspberry Pi OS de 64 bits): interfaz, procesamiento para el especialista, calibración de color y escala del acople de 8 cm, y el clasificador final (ensamble Z.18 + Z.19 en TensorFlow Lite). No hay que compilar nada: se instalan las librerías y se abre la app.

## 1. Primer arranque de la Pi
1. Conecte monitor (micro-HDMI), teclado, mouse y el capilógrafo USB. Encienda.
2. Termine el asistente de bienvenida: país, usuario y contraseña, Wi-Fi. Si pregunta por actualizaciones, acepte.
3. Compruebe que el sistema es de 64 bits: abra la Terminal y escriba `uname -m` → debe decir `aarch64`.

## 2. Copiar el paquete
- Con una USB: copie `DERMATEC_APP.zip` a una memoria USB, conéctela a la Pi y arrastre el ZIP a la carpeta personal (`/home/SU_USUARIO`).
- O descárguelo desde Google Drive con el navegador de la Pi (queda en `Descargas`).

En la Terminal (cambie la ruta si quedó en Descargas):
```bash
cd ~
unzip ~/DERMATEC_APP.zip -d ~        # o: unzip ~/Downloads/DERMATEC_APP.zip -d ~
cd ~/DERMATEC_APP
```

## 3. Instalar (una sola vez, 5–15 min)
```bash
bash pi/instalar_pi.sh
```
Instala python3-tk, v4l-utils y las librerías de `pi/requirements_pi.txt` (LiteRT, OpenCV, NumPy, Pillow, scikit-learn) en `.venv`.
Pide la contraseña del usuario. Al final debe decir `OK · Python … · modelos: ['Z.18', 'Z.19']` y crea el ícono DERMATEC en el escritorio.

## 4. Probar la cámara
```bash
bash pi/probar_camara.sh
```
Debe listar el capilógrafo y guardar `prueba_camara.jpg`.

## 5. Abrir la app
Doble clic en **DERMATEC** del escritorio (si pregunta, elija *Ejecutar*), o:
```bash
bash pi/ejecutar_dermatec.sh
```
En la app: DISPOSITIVO → elija la cámara USB → conectar. El botón físico SNAP funciona si la cámara lo publica en Linux (verifíquelo con `bash pi/diagnostico_pi.sh`); siempre puede usar CAPTURAR o la tecla C.

## 6. Medir el tiempo (benchmark)
Copie una carpeta con 10 capturas del acople (por ejemplo `~/fotos_prueba`) y ejecute:
```bash
.venv/bin/python herramientas/benchmark_pi.py ~/fotos_prueba --n 10
```
Guarda `benchmark_pi_resultados.json`. Para la temperatura, mientras usa la app ejecute en otra terminal `vcgencmd measure_temp` cada pocos minutos. Si el total por imagen es aceptable (≈ 2–4 s), se queda el ensamble Z.18 + Z.19; si no, se pasa a Z.19 solo.

## Regla en milímetros
El video en vivo y la vista ampliada de cada captura muestran una regla milimetrada a lo largo del borde inferior (marca cada 1 mm, número cada 5 mm), calculada con la escala del acople (`calibracion_escala.json`). Solo se dibuja en pantalla: las imágenes guardadas y las que analiza la red no cambian. Si la escala no se ha calibrado con regla (MENÚ → Calibrar escala), la regla usa el valor estimado.

## Si algo falla
- El registro de cada ejecución queda en `~/dermatec_ultima_ejecucion.log`.
- `uname -m` dice `armv7l` → la SD se grabó con el sistema de 32 bits: regrábela con *Raspberry Pi OS (64-bit)*.
- La cámara no aparece → `v4l2-ctl --list-devices`; cambie de puerto USB (use los azules, USB 3).
