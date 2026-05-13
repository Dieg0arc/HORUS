"""core/inference_worker.py

Hilo de fondo para inferencia ML (YOLO y LSTM).

Desacopla completamente la inferencia del render loop de Pygame:
- El hilo principal sigue dibujando a 30 FPS sin importar cuánto tarde la IA.
- Política "drop-if-busy": si el worker no terminó el frame anterior,
  el nuevo reemplaza al pendiente (siempre procesa el frame más reciente).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from core.logger import get_logger

_log = get_logger("inference_worker")


class InferenceWorker:
    """Ejecuta predicciones YOLO o LSTM en un hilo daemon.

    Uso básico::

        worker = InferenceWorker()
        worker.start('yolo')

        # En el game loop:
        worker.submit(frame)
        cls, conf, box = worker.get_result()

        # Al cerrar:
        worker.stop()

    El resultado es ``(None, 0.0, None)`` hasta que llega la primera
    predicción.  Los resultados nunca bloquean: siempre devuelven el
    último valor disponible.
    """

    # Resultado nulo por defecto
    _EMPTY: tuple = (None, 0.0, None)   # (class_name, confidence, box|None)

    def __init__(self) -> None:
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._result: tuple = self._EMPTY
        self._lock = threading.Lock()
        self._mode: str = "yolo"
        self._active: bool = False
        self._thread: Optional[threading.Thread] = None

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def start(self, mode: str = "yolo") -> None:
        """Arranca el hilo daemon.  ``mode`` puede ser ``'yolo'`` o ``'lstm'``."""
        if self._active:
            return
        self._mode = mode
        self._result = self._EMPTY
        self._active = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="InferenceWorker"
        )
        self._thread.start()
        _log.info("InferenceWorker arrancado en modo '%s'.", mode)

    def stop(self) -> None:
        """Detiene el hilo de forma ordenada (espera máx. 2 s)."""
        if not self._active:
            return
        self._active = False
        # Desbloquear get() si está esperando
        try:
            self._frame_q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._result = self._EMPTY
        _log.info("InferenceWorker detenido.")

    # ── API pública ───────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Cambia el modo de inferencia ('yolo' | 'lstm') de forma thread-safe."""
        self._mode = mode

    def submit(self, frame: np.ndarray) -> None:
        """Envía un frame.  Si hay uno pendiente sin procesar, lo descarta."""
        if not self._active:
            return
        # Vaciar la cola para garantizar que siempre se procesa el frame más
        # reciente y no uno que lleva varios ms esperando.
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_result(self) -> tuple:
        """Devuelve ``(class_name, confidence, box|None)`` sin bloquear."""
        with self._lock:
            return self._result

    # ── Hilo de fondo ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        # Importaciones diferidas para no arrastrar módulos pesados al inicio
        from core.detector import detector
        from core.lstm_detector import lstm_detector

        while self._active:
            # Esperar frame con timeout para poder revisar _active
            try:
                frame = self._frame_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if frame is None:   # señal de parada
                break

            try:
                mode = self._mode
                if mode == "yolo":
                    cls, conf, box = detector.predict(frame)
                    with self._lock:
                        self._result = (cls, conf, box)
                elif mode == "lstm":
                    cls, conf = lstm_detector.process_frame(frame)
                    with self._lock:
                        self._result = (cls, conf, None)
            except Exception as exc:
                _log.error("Error en inferencia (%s): %s", self._mode, exc, exc_info=True)
