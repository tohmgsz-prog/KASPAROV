
import os, sys, time, threading, asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple

import cv2, numpy as np, chess, chess.engine
from ultralytics import YOLO
import pygame
import tempfile
import edge_tts

from PySide6.QtCore    import Qt, QTimer, Signal, QThread
from PySide6.QtGui     import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QCheckBox,
    QVBoxLayout, QHBoxLayout, QListWidget, QTextEdit, QMessageBox,
    QStatusBar, QTabWidget, QSplitter, QSizePolicy, QGroupBox,
    QSlider, QProgressBar, QFrame, QScrollArea, QDialog, QDialogButtonBox
)

# ─────────────────────────────────────────────────────────── CONFIG ──────────
MODEL_PATH     = r"/Users/tomasserrasnm/Downloads/chess-bot-pc/chesspiece-detection-model.pt"
STOCKFISH_PATH = r"/Users/tomasserrasnm/Downloads/Chess-Tracker-main/content/code/Chess Tracker/stockfishserra"
PIECES_DIR     = r"/Users/tomasserrasnm/Downloads/chess-bot-pc/assets/pieces"
CAM_CANDIDATES = [0, 1, 2, 3]

CONFIDENCE     = 0.10
FRAME_SIZE     = 640
YOLO_SIZE      = 640
STOCKFISH_TIME = 1.5

HUMAN_COLOR    = chess.BLACK
ENGINE_COLOR   = chess.WHITE
BOARD_ORIENTATION = "WHITE_RIGHT"
CAMERA_SIDE    = "A_NEAR"

SQUARE_PX      = 88
BOARD_BORDER   = 12
FILES          = "abcdefgh"
BRIGHTNESS = 100
GAMMA      = 1.8
CONTRAST   = 15
# ── Baseline ─────────────────────────────────────────────────────────────────
BASELINE_NEED      = 4
BASELINE_NOISE_TOL = 2

# ── Deteção de jogada ─────────────────────────────────────────────────────────
VOTE_WINDOW    = 16
VOTE_THRESHOLD = 8
STABLE_NEEDED  = VOTE_THRESHOLD
CASTLE_VOTES   = 6
CASTLE_STABLE  = CASTLE_VOTES
NOISE_TOL      = 1
COOLDOWN_SEC   = 1.5

# ── Watchdog ──────────────────────────────────────────────────────────────────
WATCHDOG_SEC   = 40.0

# ── Tracker ──────────────────────────────────────────────────────────────────
TRACK_UP       = 2
TRACK_DOWN     = 2
TRACK_OCC_THR  = 8
TRACK_FREE_THR = 1
TRACK_MAX      = 14

# ── Scoring ───────────────────────────────────────────────────────────────────
WHITE_SQ_CONF        = 0.40
MIN_SCORE            = 45
GAP_REQUIRED         = 25
WHITE_MOVE_CONFIRM   = 15

# ── Histórico de occupancy ────────────────────────────────────────────────────
OCC_HISTORY_LEN = 6

C_LIGHT  = (236, 217, 185)
C_DARK   = (174, 122,  89)
C_BORDER = (40,  32,  24)

SYMBOL_TO_FILE = {
    'P':'wP.png','N':'wN.png','B':'wB.png','R':'wR.png','Q':'wQ.png','K':'wK.png',
    'p':'bP.png','n':'bN.png','b':'bB.png','r':'bR.png','q':'bQ.png','k':'bK.png',
}

TTS_VOICES = {
    "pt": "pt-PT-DuarteNeural",
    "en": "en-GB-RyanNeural",
    "fr": "fr-FR-HenriNeural",
}
TTS_LANG = "pt"

BG    = "#1A1F2E"; PANEL  = "#222840"; BORDER = "#3A4260"
AMBER = "#F5A623"; CYAN   = "#00D4FF"; GREEN  = "#2ECC9A"
RED   = "#FF5564"; MUTED  = "#7A8299"; WHITE  = "#E8ECF4"

# ─────────────────────────────────────────────────────── GAME STATE ──────────
class GameState(Enum):
    SETUP            = "SETUP"
    IDLE             = "IDLE"
    BASELINE         = "BASELINE"
    WAITING_MOVE     = "WAITING"
    WAITING_OPERATOR = "WAITING_OP"
    ENGINE_THINK     = "THINKING"
    GAME_OVER        = "OVER"
    ERROR            = "ERROR"

# ──────────────────────────────────────────────────────── THEME ──────────────
_QSS = f"""
* {{ font-family:"Consolas","Courier New",monospace; font-size:12px;
color:{WHITE}; background:transparent; }}
QMainWindow,QWidget#root {{ background:{BG}; }}
QGroupBox {{ border:1px solid {BORDER}; border-radius:8px;
margin-top:20px; padding:8px 6px 6px 6px; background:{PANEL}; }}
QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top
left; left:12px; padding:2px 8px; color:{AMBER}; font-size:10px;
font-weight:bold; letter-spacing:2px; background:{BG}; border:1px
solid {BORDER}; border-radius:4px; }}
QPushButton {{ padding:8px 14px; border-radius:6px; border:1px solid
{BORDER}; background:#252B40; color:{WHITE}; font-weight:bold; }}
QPushButton:hover {{ border:1px solid {AMBER}; background:#2E3550; color:{AMBER}; }}
QPushButton:pressed {{ background:#151A28; }}
QPushButton:disabled {{ color:{MUTED}; border-color:#2A3050; background:#1E2438; }}
QPushButton#primary {{ border:1px solid {GREEN}; color:{GREEN}; background:#152A22; }}
QPushButton#primary:hover {{ background:#1C3828; border-color:#40E8C0; }}
QPushButton#danger  {{ border:1px solid {RED};   color:{RED};   background:#2A1520; }}
QPushButton#start {{ border:2px solid {CYAN}; color:{CYAN}; background:#102038;
font-size:13px; padding:10px 18px; letter-spacing:1px; }}
QPushButton#start:hover {{ background:#162848; border-color:#40E8FF; color:#40E8FF; }}
QPushButton#start:disabled {{ border-color:#2A3848; color:#3A5068; background:#1C2638; }}
QListWidget,QTextEdit {{ background:#151A28; border:1px solid {BORDER};
border-radius:6px; color:{WHITE}; selection-background-color:#2E3550; }}
QListWidget::item {{ padding:3px 6px; border-bottom:1px solid #252B40; }}
QListWidget::item:hover {{ background:#252B40; }}
QSlider::groove:horizontal {{ height:4px; background:#252B40; border-radius:2px; }}
QSlider::sub-page:horizontal {{ background:{AMBER}; border-radius:2px; }}
QSlider::handle:horizontal {{ width:14px; height:14px; margin:-5px 0;
background:{WHITE}; border:2px solid {AMBER}; border-radius:7px; }}
QProgressBar {{ border:1px solid {BORDER}; border-radius:4px;
background:#151A28; text-align:center; color:{WHITE}; font-size:11px; height:18px; }}
QProgressBar::chunk#baseline {{
background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #604010,stop:1 {AMBER});
border-radius:3px; }}
QProgressBar::chunk#stable {{
background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0A3828,stop:1 {GREEN});
border-radius:3px; }}
QCheckBox {{ spacing:8px; color:{MUTED}; font-size:11px; }}
QCheckBox:checked {{ color:{WHITE}; }}
QCheckBox::indicator {{ width:14px; height:14px; border:1px solid {BORDER};
border-radius:3px; background:#151A28; }}
QCheckBox::indicator:checked {{ background:{AMBER}; border-color:{AMBER}; }}
QTabWidget::pane {{ border:1px solid {BORDER}; border-radius:0 6px 6px 6px; background:{PANEL}; }}
QTabBar::tab {{ background:#151A28; border:1px solid {BORDER}; border-bottom:none;
padding:6px 14px; border-radius:4px 4px 0 0; color:{MUTED}; font-size:11px; margin-right:2px; }}
QTabBar::tab:selected {{ background:{PANEL}; color:{AMBER}; border-bottom:2px solid {AMBER}; }}
QSplitter::handle {{ background:{BORDER}; width:3px; }}
QScrollBar:vertical {{ width:6px; background:#151A28; }}
QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; min-height:20px; }}
QStatusBar {{ background:#151A28; border-top:1px solid {BORDER};
color:{MUTED}; font-size:11px; padding:2px 8px; }}
"""

def apply_theme(app: QApplication):
    app.setStyle("Fusion"); app.setStyleSheet(_QSS)

# ────────────────────────────────────────────────────────── UTILS ────────────
def bgr_to_qpixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h, w, c = rgb.shape
    return QPixmap.fromImage(
        QImage(rgb.data, w, h, c*w, QImage.Format.Format_RGB888).copy())

def overlay_png(bg, fg, x, y):
    h, w = fg.shape[:2]
    if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]: return
    if fg.shape[2] == 4:
        a = fg[...,3:4].astype(np.float32)/255.0
        bg[y:y+h, x:x+w] = (a*fg[...,:3] + (1-a)*bg[y:y+h, x:x+w]).astype(np.uint8)
    else:
        bg[y:y+h, x:x+w] = fg[...,:3]

def load_piece_images(px: int) -> dict:
    imgs = {}
    for sym, fname in SYMBOL_TO_FILE.items():
        p = os.path.join(PIECES_DIR, fname)
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is not None:
                imgs[sym] = cv2.resize(im, (px,px), interpolation=cv2.INTER_AREA)
    return imgs

def board_occupancy(board: chess.Board) -> Set[str]:
    return {chess.square_name(sq) for sq in board.piece_map()}

# ─────────────────────── INFER_MOVE ──────────────────────────────────────────
def _expected_delta(board: chess.Board,
                    mv: chess.Move) -> Tuple[Set[str], Set[str]]:
    b = board.copy()
    before = board_occupancy(b)
    b.push(mv)
    after = board_occupancy(b)
    return before - after, after - before

def score_move(board: chess.Board, mv: chess.Move,
               gone: Set[str], appeared: Set[str],
               occ_after: Set[str]) -> float:
    exp_gone, exp_appeared = _expected_delta(board, mv)
    sc  = len(exp_gone    & gone)      * 40.0
    sc += len(exp_appeared & appeared) * 40.0
    sc -= len(exp_gone    - gone)      * 20.0
    sc -= len(exp_appeared - appeared) * 20.0
    sc -= len(gone    - exp_gone)      * 25.0
    sc -= len(appeared - exp_appeared) * 25.0
    if exp_gone == gone and exp_appeared == appeared:
        sc += 15.0
    if chess.square_name(mv.from_square) not in gone:
        sc -= 30.0
    return sc

def infer_move(board: chess.Board,
               gone: Set[str], appeared: Set[str],
               occ_after: Set[str]) -> Tuple[Optional[chess.Move], float]:
    if not gone and not appeared:
        return None, -9999.0
    scored = [(score_move(board, mv, gone, appeared, occ_after), mv)
              for mv in board.legal_moves]
    if not scored:
        return None, -9999.0
    scored.sort(key=lambda x: -x[0])
    best_sc, best_mv = scored[0]
    if best_sc < MIN_SCORE:
        return best_mv, best_sc
    if len(scored) >= 2:
        gap = best_sc - scored[1][0]
        if gap < GAP_REQUIRED:
            return best_mv, best_sc - (GAP_REQUIRED - gap) * 3.0
    return best_mv, best_sc

# ─────────────────────────────────────────────────────── GRID MODEL ──────────
@dataclass
class GridModel:
    margin_pct:  float = 6.0
    scale:       float = 1.00
    dx:          int   = 0
    dy:          int   = 0
    orientation: str   = "WHITE_RIGHT"
    camera_side: str   = "A_NEAR"

    def rect(self, fs: int):
        M = fs * self.margin_pct / 100.0
        bw = (fs - 2*M) * self.scale
        cx = fs/2 + self.dx; cy = fs/2 + self.dy
        c = lambda v: max(0.0, min(float(fs-1), v))
        return c(cx-bw/2), c(cy-bw/2), c(cx+bw/2), c(cy+bw/2)

    def _row_img_to_file_index(self, row_img: int) -> int:
        return 7 - row_img if self.camera_side == "A_NEAR" else row_img

    def _file_index_to_row_img(self, file_index: int) -> int:
        return 7 - file_index if self.camera_side == "A_NEAR" else file_index

    def _col_img_to_rank(self, col_img: int) -> int:
        return 8 - col_img if self.orientation == "WHITE_RIGHT" else col_img + 1

    def _rank_to_col_img(self, rank: int) -> int:
        return 8 - rank if self.orientation == "WHITE_RIGHT" else rank - 1

    def pt_to_square(self, cx: float, cy: float, fs: int) -> Optional[str]:
        x0, y0, x1, y1 = self.rect(fs)
        w, h = x1 - x0, y1 - y0
        if w < 5 or h < 5: return None
        col_img = int((cx - x0) / (w / 8))
        row_img = int((cy - y0) / (h / 8))
        if not (0 <= col_img <= 7 and 0 <= row_img <= 7): return None
        rank       = self._col_img_to_rank(col_img)
        file_index = self._row_img_to_file_index(row_img)
        if not (1 <= rank <= 8 and 0 <= file_index <= 7): return None
        return FILES[file_index] + str(rank)

    def square_to_img_coords(self, square_name: str, fs: int) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.rect(fs)
        w, h = x1 - x0, y1 - y0
        file_index = FILES.index(square_name[0])
        rank = int(square_name[1])
        col_img = self._rank_to_col_img(rank)
        row_img = self._file_index_to_row_img(file_index)
        return x0 + (col_img + 0.5) * (w / 8), y0 + (row_img + 0.5) * (h / 8)

    def draw_overlay(self, img: np.ndarray, fs: int) -> np.ndarray:
        out = img.copy(); ov = out.copy()
        x0, y0, x1, y1 = self.rect(fs)
        xi0, yi0, xi1, yi1 = map(int, map(round, (x0, y0, x1, y1)))
        cv2.rectangle(ov, (xi0, yi0), (xi1, yi1), (255, 80, 0), 2)
        sx, sy = (x1 - x0) / 8, (y1 - y0) / 8
        for i in range(9):
            cv2.line(ov, (int(x0+i*sx), yi0), (int(x0+i*sx), yi1), (255, 200, 0), 1)
            cv2.line(ov, (xi0, int(y0+i*sy)), (xi1, int(y0+i*sy)), (255, 200, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for col in range(8):
            cv2.putText(ov, str(self._col_img_to_rank(col)),
                        (int(x0+(col+0.5)*sx-5), yi0-6), font, 0.45, (200,200,200), 1)
        for row in range(8):
            fi = self._row_img_to_file_index(row)
            lbl = FILES[fi] if 0 <= fi <= 7 else "?"
            cv2.putText(ov, lbl, (xi0-16, int(y0+(row+0.5)*sy+5)), font, 0.45, (200,200,200), 1)
        return cv2.addWeighted(ov, 0.80, out, 0.20, 0)

# ──────────────────────────────────────────────────────── WIDGETS ────────────
class AspectLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._pm = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
        self.setStyleSheet(f"background:{BG}; border:1px solid {BORDER}; border-radius:8px;")
        self.setAlignment(Qt.AlignCenter)
    def setPixmap(self, pm: QPixmap): self._pm = pm; self.update()
    def paintEvent(self, e):
        super().paintEvent(e)
        if not self._pm: return
        s = self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p = QPainter(self)
        p.drawPixmap((self.width()-s.width())//2, (self.height()-s.height())//2, s)

class VideoLabel(AspectLabel):
    clicked = Signal(int, int)
    def __init__(self): super().__init__(); self._iw = self._ih = 1
    def set_image(self, img: np.ndarray):
        self._ih, self._iw = img.shape[:2]; self.setPixmap(bgr_to_qpixmap(img))
    def mousePressEvent(self, e):
        if not self._pm: return
        s  = self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (self.width()-s.width())/2; oy = (self.height()-s.height())/2
        rx = (e.position().x()-ox)/s.width()*self._iw
        ry = (e.position().y()-oy)/s.height()*self._ih
        if 0<=rx<self._iw and 0<=ry<self._ih: self.clicked.emit(int(rx), int(ry))

class StatusWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._state="idle"; self._text="AGUARDANDO"; self._sub=""
        self.setMinimumHeight(72); self.setMaximumHeight(80); self._blink=False
        t=QTimer(self); t.timeout.connect(self._tog); t.start(500)
    def set_state(self, s, t, sub=""): self._state=s; self._text=t; self._sub=sub; self.update()
    def _tog(self): self._blink=not self._blink; self.update()
    def _ac(self):
        return QColor({"human":GREEN,"engine":CYAN,"baseline":AMBER,
                       "warn":RED,"idle":MUTED,"setup":CYAN}.get(self._state, WHITE))
    def paintEvent(self, e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ac=self._ac()
        p.setBrush(QColor("#0A0C12")); p.setPen(QPen(ac.darker(160), 1))
        p.drawRoundedRect(0, 0, self.width()-1, self.height()-1, 8, 8)
        p.setBrush(ac.darker(200) if self._state=="human" and self._blink else ac)
        p.setPen(Qt.NoPen); p.drawRoundedRect(0, 0, 5, self.height(), 2, 2)
        p.setPen(ac); p.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        p.drawText(16, 0, self.width()-20, self.height()//2+10,
                   Qt.AlignLeft|Qt.AlignVCenter, self._text)
        if self._sub:
            p.setPen(QColor(MUTED)); p.setFont(QFont("Consolas", 10))
            p.drawText(18, self.height()//2+2, self.width()-20, self.height()//2-4,
                       Qt.AlignLeft|Qt.AlignVCenter, self._sub)

class PromotionDialog(QDialog):
    """Blocking dialog shown when a black pawn reaches rank 1."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Promoção de Peão")
        self.setModal(True)
        self.setFixedSize(420, 160)
        self.chosen = chess.QUEEN
        self.setStyleSheet(
            f"background:{BG}; color:{WHITE}; font-family:Consolas;")

        l = QVBoxLayout(self); l.setSpacing(14); l.setContentsMargins(20,16,20,16)

        lbl = QLabel("♟  Peão promovido — escolhe a peça:")
        lbl.setStyleSheet(f"color:{AMBER}; font-size:13px; font-weight:bold;")
        l.addWidget(lbl)

        row = QHBoxLayout(); row.setSpacing(8)
        for piece_type, symbol, name in [
            (chess.QUEEN,  "♛", "DAMA"),
            (chess.ROOK,   "♜", "TORRE"),
            (chess.BISHOP, "♝", "BISPO"),
            (chess.KNIGHT, "♞", "CAVALO"),
        ]:
            btn = QPushButton(f"{symbol}\n{name}")
            btn.setFixedSize(86, 60)
            btn.setStyleSheet(
                f"font-size:18px; border:1px solid {BORDER}; border-radius:6px;"
                f" background:#252B40; color:{WHITE};"
                f" QPushButton:hover {{ border-color:{AMBER}; color:{AMBER}; }}")
            btn.clicked.connect(lambda _, pt=piece_type: self._pick(pt))
            row.addWidget(btn)
        l.addLayout(row)

    def _pick(self, piece_type: int):
        self.chosen = piece_type
        self.accept()

# ─────────────────────────────────────────────────────── WORKERS ─────────────
_GAMMA_TABLE = np.array(
    [((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)], dtype=np.uint8
)

class CameraThread(QThread):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__()
        self._cap = cap
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._run = False

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    @staticmethod
    def _enhance(frame: np.ndarray) -> np.ndarray:
        # gamma correction (brightens dark frames)
        frame = cv2.LUT(frame, _GAMMA_TABLE)
        # brightness / contrast:  out = alpha*in + beta
        alpha = 1.0 + CONTRAST / 127.0
        beta  = float(BRIGHTNESS) - 100.0
        if alpha != 1.0 or beta != 0.0:
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return frame

    def run(self):
        self._run = True
        while self._run:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                frame = self._enhance(frame)
                with self._lock:
                    self._latest = frame
            else:
                self.msleep(5)

    def stop(self):
        self._run = False
        self.wait(2000)
        self._cap.release()

@dataclass
class DetResult:
    boxes: List[List[float]]
    confs: List[float]

class YoloWorker(QThread):
    result_ready = Signal(DetResult)
    def __init__(self, model: YOLO):
        super().__init__()
        self.model=model; self._lock=threading.Lock(); self._frame=None; self._run=False
    def submit(self, frame: np.ndarray):
        with self._lock: self._frame=frame.copy()
    def run(self):
        self._run=True
        while self._run:
            with self._lock: f, self._frame = self._frame, None
            if f is None: self.msleep(4); continue
            try:
                res=self.model(f, conf=CONFIDENCE, verbose=False, imgsz=YOLO_SIZE); r=res[0]
                if r.boxes is not None and len(r.boxes):
                    self.result_ready.emit(DetResult(
                        r.boxes.xyxy.cpu().numpy().tolist(),
                        r.boxes.conf.cpu().numpy().tolist()))
                else: self.result_ready.emit(DetResult([], []))
            except Exception as ex: print(f"[YOLO] {ex}")
    def stop(self): self._run=False; self.wait()

class StockfishWorker(QThread):
    move_ready     = Signal(chess.Move)
    error_occurred = Signal(str)
    engine_ready   = Signal(bool)
    def __init__(self, engine_path: str, think_time: float):
        super().__init__()
        self._path=engine_path; self._think_time=think_time
        self._engine=None; self._board=None
        self._lock=threading.Lock(); self._event=threading.Event()
        self._running=False; self._engine_ok=False
    def request_move(self, board: chess.Board):
        with self._lock: self._board = board.copy()
        self._event.set()
    def run(self):
        self._running = True
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
            self._engine_ok = True; self.engine_ready.emit(True)
        except Exception as ex:
            self.error_occurred.emit(f"Engine open failed: {ex}")
            self._engine_ok = False; self.engine_ready.emit(False); return
        while self._running:
            self._event.wait(timeout=0.2); self._event.clear()
            with self._lock: board = self._board; self._board = None
            if board is None: continue
            try:
                result = self._engine.play(board, chess.engine.Limit(time=self._think_time))
                if result.move: self.move_ready.emit(result.move)
                else: self.error_occurred.emit("No move returned")
            except Exception as ex: self.error_occurred.emit(str(ex))
        if self._engine:
            try: self._engine.quit()
            except: pass
            self._engine = None; self._engine_ok = False
    def stop(self):
        self._running = False; self._event.set(); self.wait(4000)

# ─────────────────────────────────────────────────────── MAIN WINDOW ─────────
class ChessBotWindow(QMainWindow):
    _sig_update_ui = Signal()
    _sig_stockfish = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KASPAROV V2 - INFINITE GAMES")
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(1400, 860)

        self.board = chess.Board(); self.state = GameState.SETUP
        self.board_lock = threading.Lock()

        self.pipeline    : Optional[cv2.VideoCapture] = None
        self._cam_thread : Optional[CameraThread]     = None
        self.yolo        : Optional[YOLO]            = None
        self.yolo_worker : Optional[YoloWorker]      = None
        self.sf_worker   : Optional[StockfishWorker] = None
        self._sf_thinking: bool = False

        self.grid = GridModel(orientation=BOARD_ORIENTATION, camera_side=CAMERA_SIDE)
        self.piece_imgs = load_piece_images(SQUARE_PX)
        self.H = None; self.calib_pts: List[Tuple[int,int]] = []
        self.calibrated = False

        self._last_det   : Optional[DetResult] = None
        self._raw_occ    : Set[str]             = set()
        self._yolo_count : int                  = 0
        self._last_submit: float                = 0.0

        # ── tracker ──────────────────────────────────────────────────────────
        self._sq_tracker : dict     = {}
        self._tracked_occ: Set[str] = set()

        # ── baseline ──────────────────────────────────────────────────────────
        self._occ_base   : Optional[Set[str]] = None
        self._base_cand  : Optional[Set[str]] = None
        self._base_stable: int                 = 0

        # ── deteção de jogada ─────────────────────────────────────────────────
        self._occ_history: List[Set[str]] = []

        self._cand_gone       : Optional[Set[str]]  = None
        self._cand_appeared   : Optional[Set[str]]  = None
        self._cand_occ        : Set[str]             = set()
        self._cand_move       : Optional[chess.Move] = None
        self._cand_stable     : int                  = 0
        self._vote_window     : list                 = []
        self._last_confirm    : float                = 0.0
        self._cand_is_castling: bool                 = False

        self._occ_before: Optional[Set[str]] = None

        self._engine_instruction : Optional[str]        = None
        self._engine_instr_since : float                 = 0.0
        self._engine_instr_move  : Optional[chess.Move] = None
        self._white_confirm_frames: int                  = 0

        self._illegal_active  : bool     = False
        self._illegal_since   : float    = 0.0
        self._illegal_gone    : Set[str] = set()
        self._illegal_appeared: Set[str] = set()
        self._illegal_reason  : str      = ""

        self._human_turn_announced: bool = False
        self._last_activity: float = time.time()

        self._debug_dirty  = False; self._debug_lines: List[str] = []
        dt = QTimer(self); dt.setInterval(400)
        dt.timeout.connect(self._flush_debug); dt.start()

        wt = QTimer(self); wt.setInterval(5000)
        wt.timeout.connect(self._watchdog_check); wt.start()

        self._pending_lock       = threading.Lock()
        self._pending_stockfish  = False
        self._pending_ui_refresh = False

        self._sig_update_ui.connect(self._consume_pending_ui)
        self._sig_stockfish.connect(self._consume_pending_stockfish)

        try:
            pygame.mixer.pre_init(44100, -16, 2, 1024)
            pygame.mixer.init()
        except Exception as e:
            print(f"[SPEAK] Falha mixer: {e}")

        self._speak_lock   = threading.Lock()
        self._speak_active = False
        self._last_spoken  = 0.0
        self._tts_lang: str = TTS_LANG

        self._init_ui(); self._init_backend()
        tt = QTimer(self); tt.timeout.connect(self._tick); tt.start(16)

    # ══════════════════════════════════════════════════════════════ WATCHDOG ══
    def _watchdog_check(self):
        if self.state == GameState.WAITING_OPERATOR:
            elapsed = time.time() - self._last_activity
            if elapsed > WATCHDOG_SEC:
                self.statusbar.showMessage(
                    f"[WATCHDOG] WAITING_OPERATOR preso {elapsed:.0f}s — re-baseline", 4000)
                self._white_confirm_frames = 0
                self._reset_baseline()
                return

        if self.state == GameState.WAITING_MOVE:
            if self._occ_base is None:
                return
            elapsed = time.time() - self._last_activity
            if elapsed > WATCHDOG_SEC:
                self.statusbar.showMessage(
                    f"[WATCHDOG] Sem actividade há {elapsed:.0f}s — re-baseline automático", 4000)
                self._reset_baseline()

    # ═══════════════════════════════════════════════════════════ UI BUILD ════
    def _init_ui(self):
        root_w = QWidget(); root_w.setObjectName("root")
        self.setCentralWidget(root_w)
        root = QVBoxLayout(root_w)
        root.setContentsMargins(10, 8, 10, 4); root.setSpacing(5)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout(); hdr.setSpacing(8)
        title = QLabel("KASPAROV")
        title.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{AMBER}; letter-spacing:3px;")
        v2lbl = QLabel("V2")
        v2lbl.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        v2lbl.setStyleSheet(f"color:{CYAN}; letter-spacing:2px;")
        dot = QLabel(" · ")
        dot.setStyleSheet(f"color:{BORDER}; font-size:16px;")
        sub = QLabel("INFINITE GAMES")
        sub.setFont(QFont("Consolas", 10))
        sub.setStyleSheet(f"color:{MUTED}; letter-spacing:2px; padding-top:5px;")
        hdr.addWidget(title); hdr.addWidget(v2lbl)
        hdr.addWidget(dot);   hdr.addWidget(sub); hdr.addStretch()
        self._pill_cam = self._make_pill("◉  CÂMERA", False)
        self._pill_sf  = self._make_pill("◉  ENGINE", False)
        hdr.addWidget(self._pill_cam); hdr.addWidget(self._pill_sf)
        root.addLayout(hdr)

        div = QFrame(); div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background:{BORDER}; max-height:1px;")
        root.addWidget(div)

        # ── Main splitter ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3); splitter.setChildrenCollapsible(False)

        self.vid_lbl = VideoLabel()
        self.vid_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.vid_lbl.setMinimumSize(380, 380)
        self.vid_lbl.clicked.connect(self._on_vid_click)
        splitter.addWidget(self.vid_lbl)

        right_w = QWidget()
        right_w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_w.setMinimumWidth(360)
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(4, 0, 0, 0); right_l.setSpacing(5)

        self.board_lbl = AspectLabel()
        self.board_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.board_lbl.setMinimumSize(240, 240)
        right_l.addWidget(self.board_lbl, 1)

        self.status_widget = StatusWidget()
        right_l.addWidget(self.status_widget)

        right_l.addWidget(self._build_progress_strip())

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tabs.addTab(self._build_tab_jogo(),  " JOGO ")
        tabs.addTab(self._build_tab_setup(), " SETUP ")
        tabs.addTab(self._build_tab_debug(), " DEBUG ")
        tabs.setFixedHeight(360)
        right_l.addWidget(tabs)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 55); splitter.setStretchFactor(1, 45)
        root.addWidget(splitter, 1)

        self.statusbar = QStatusBar(); self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("A inicializar...")

    # ── Pill helpers ─────────────────────────────────────────────────────────
    def _make_pill(self, text: str, active: bool) -> QLabel:
        lbl = QLabel(text)
        c = GREEN if active else MUTED
        lbl.setStyleSheet(
            f"color:{c}; font-size:10px; font-family:Consolas; letter-spacing:1px;"
            f" padding:3px 10px; border:1px solid {c if active else BORDER}; border-radius:10px;")
        return lbl

    def _update_pill(self, lbl: QLabel, text: str, active: bool):
        c = GREEN if active else RED
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color:{c}; font-size:10px; font-family:Consolas; letter-spacing:1px;"
            f" padding:3px 10px; border:1px solid {c}; border-radius:10px;")

    # ── Progress strip (always visible above tabs) ───────────────────────────
    def _build_progress_strip(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(
            f"background:{PANEL}; border:1px solid {BORDER}; border-radius:6px;")
        outer = QVBoxLayout(w)
        outer.setContentsMargins(12, 8, 12, 8); outer.setSpacing(6)

        row1 = QHBoxLayout(); row1.setSpacing(16)

        self.lbl_phase = QLabel("--")
        self.lbl_phase.setStyleSheet(
            f"color:{AMBER}; font-weight:bold; font-size:12px;")
        self.lbl_phase.setMinimumWidth(180)
        row1.addWidget(self.lbl_phase, 1)

        for attr, name, obj_name, maxv, fmtv in [
            ("bar_baseline", "BASELINE", "baseline", BASELINE_NEED, BASELINE_NEED),
            ("bar_stable",   "DETEÇÃO",  "stable",   STABLE_NEEDED, STABLE_NEEDED),
        ]:
            col = QVBoxLayout(); col.setSpacing(2)
            lbl = QLabel(name); lbl.setStyleSheet(f"color:{MUTED}; font-size:10px; letter-spacing:1px;")
            bar = QProgressBar()
            bar.setMaximum(maxv); bar.setObjectName(obj_name)
            bar.setFormat(f"%v/{fmtv}"); bar.setFixedHeight(18)
            setattr(self, attr, bar)
            col.addWidget(lbl); col.addWidget(bar)
            row1.addLayout(col, 1)

        self.lbl_candidate = QLabel("candidate: --")
        self.lbl_candidate.setStyleSheet(f"color:{CYAN}; font-size:11px; font-family:Consolas;")
        self.lbl_candidate.setWordWrap(True)

        outer.addLayout(row1)
        outer.addWidget(self.lbl_candidate)
        return w

    # ── Tab: JOGO ────────────────────────────────────────────────────────────
    def _build_tab_jogo(self) -> QWidget:
        outer = QWidget()
        outer_l = QVBoxLayout(outer); outer_l.setContentsMargins(0,0,0,0); outer_l.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8, 6, 8, 6); l.setSpacing(6)

        # Top row: history + action buttons
        top = QHBoxLayout(); top.setSpacing(8)

        hist_gb = QGroupBox("HISTORIAL")
        hist_l = QVBoxLayout(hist_gb); hist_l.setContentsMargins(5, 3, 5, 3)
        self.moves_list = QListWidget()
        self.moves_list.setFixedHeight(110)
        hist_l.addWidget(self.moves_list)

        btns_w = QWidget(); btns_w.setFixedWidth(148)
        btns_l = QVBoxLayout(btns_w); btns_l.setContentsMargins(0, 0, 0, 0); btns_l.setSpacing(4)

        self.btn_start = QPushButton("INICIAR  [↵]")
        self.btn_start.setObjectName("start")
        self.btn_start.setFixedHeight(36)
        self.btn_start.clicked.connect(self._on_start_game)
        self.btn_start.setEnabled(False)

        self.btn_capture = QPushButton("CAPTURAR  [␣]")
        self.btn_capture.setObjectName("primary"); self.btn_capture.setFixedHeight(30)
        self.btn_capture.clicked.connect(self._on_capture)

        self.btn_confirm = QPushButton("CONFIRMAR  [␣]")
        self.btn_confirm.setEnabled(False); self.btn_confirm.setFixedHeight(30)
        self.btn_confirm.clicked.connect(self._on_confirm)

        btn_rb = QPushButton("BASELINE  [B]"); btn_rb.setFixedHeight(26)
        btn_rb.clicked.connect(self._reset_baseline)

        self.btn_reset = QPushButton("RESET  [R]")
        self.btn_reset.setObjectName("danger"); self.btn_reset.setFixedHeight(26)
        self.btn_reset.clicked.connect(self._on_reset)

        btns_l.addWidget(self.btn_start)
        btns_l.addWidget(self.btn_capture)
        btns_l.addWidget(self.btn_confirm)
        btns_l.addStretch()
        btns_l.addWidget(btn_rb)
        btns_l.addWidget(self.btn_reset)

        top.addWidget(hist_gb, 1); top.addWidget(btns_w)
        l.addLayout(top)

        # Divider
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{BORDER};"); l.addWidget(sep)

        # Bottom row: overlays | TTS | orientation+camera
        bot = QHBoxLayout(); bot.setSpacing(18)

        # Overlays + auto (2×2 grid to save vertical space)
        ovl = QVBoxLayout(); ovl.setSpacing(2)
        ovl.addWidget(self._sec_lbl("OVERLAYS"))
        self.chk_grid  = QCheckBox("Grid"); self.chk_grid.setChecked(True)
        self.chk_boxes = QCheckBox("YOLO"); self.chk_boxes.setChecked(True)
        self.chk_mask  = QCheckBox("Mask");  self.chk_mask.setChecked(True)
        self.chk_auto  = QCheckBox("Auto"); self.chk_auto.setChecked(True)
        self.chk_auto.stateChanged.connect(lambda s: self._on_auto_toggle(bool(s)))
        ck_row1 = QHBoxLayout(); ck_row1.setSpacing(6)
        ck_row1.addWidget(self.chk_grid); ck_row1.addWidget(self.chk_boxes)
        ck_row2 = QHBoxLayout(); ck_row2.setSpacing(6)
        ck_row2.addWidget(self.chk_mask); ck_row2.addWidget(self.chk_auto)
        ovl.addLayout(ck_row1); ovl.addLayout(ck_row2)
        bot.addLayout(ovl)

        # TTS
        tts = QVBoxLayout(); tts.setSpacing(3)
        tts.addWidget(self._sec_lbl("VOZ TTS"))
        lang_row = QHBoxLayout(); lang_row.setSpacing(3)
        self.btn_lang_pt = QPushButton("PT"); self.btn_lang_en = QPushButton("EN")
        self.btn_lang_fr = QPushButton("FR")
        for b in [self.btn_lang_pt, self.btn_lang_en, self.btn_lang_fr]:
            b.setCheckable(True); b.setFixedWidth(38); b.setFixedHeight(26)
        self.btn_lang_pt.setChecked(TTS_LANG=="pt")
        self.btn_lang_en.setChecked(TTS_LANG=="en")
        self.btn_lang_fr.setChecked(TTS_LANG=="fr")
        self.btn_lang_pt.clicked.connect(lambda: self._set_tts_lang("pt"))
        self.btn_lang_en.clicked.connect(lambda: self._set_tts_lang("en"))
        self.btn_lang_fr.clicked.connect(lambda: self._set_tts_lang("fr"))
        lang_row.addWidget(self.btn_lang_pt); lang_row.addWidget(self.btn_lang_en)
        lang_row.addWidget(self.btn_lang_fr); lang_row.addStretch()
        self._lbl_lang_voice = QLabel(TTS_VOICES[TTS_LANG])
        self._lbl_lang_voice.setStyleSheet(f"color:{CYAN}; font-size:9px;")
        self._lbl_lang_voice.setWordWrap(True)
        tts.addLayout(lang_row); tts.addWidget(self._lbl_lang_voice); tts.addStretch()
        bot.addLayout(tts)

        # Orientation + Camera
        ori = QVBoxLayout(); ori.setSpacing(3)
        ori.addWidget(self._sec_lbl("ORIENTAÇÃO"))
        ori_row = QHBoxLayout(); ori_row.setSpacing(3)
        self.btn_orient_right = QPushButton("▶ DIR")
        self.btn_orient_left  = QPushButton("◀ ESQ")
        for b in [self.btn_orient_right, self.btn_orient_left]:
            b.setCheckable(True); b.setFixedHeight(26)
        self.btn_orient_right.setChecked(BOARD_ORIENTATION=="WHITE_RIGHT")
        self.btn_orient_left.setChecked(BOARD_ORIENTATION=="WHITE_LEFT")
        self.btn_orient_right.clicked.connect(lambda: self._set_orientation("WHITE_RIGHT"))
        self.btn_orient_left.clicked.connect(lambda:  self._set_orientation("WHITE_LEFT"))
        ori_row.addWidget(self.btn_orient_right); ori_row.addWidget(self.btn_orient_left)
        ori.addLayout(ori_row)

        ori.addWidget(self._sec_lbl("CÂMERA"))
        cam_row = QHBoxLayout(); cam_row.setSpacing(3)
        self.btn_cam_a = QPushButton("Lado A")
        self.btn_cam_h = QPushButton("Lado H")
        for b in [self.btn_cam_a, self.btn_cam_h]:
            b.setCheckable(True); b.setFixedHeight(26)
        self.btn_cam_a.setChecked(CAMERA_SIDE=="A_NEAR")
        self.btn_cam_h.setChecked(CAMERA_SIDE=="H_NEAR")
        self.btn_cam_a.clicked.connect(lambda: self._set_camera_side("A_NEAR"))
        self.btn_cam_h.clicked.connect(lambda: self._set_camera_side("H_NEAR"))
        cam_row.addWidget(self.btn_cam_a); cam_row.addWidget(self.btn_cam_h)
        ori.addLayout(cam_row)
        self._lbl_cam_info = QLabel(self._camera_side_desc(CAMERA_SIDE))
        self._lbl_cam_info.setStyleSheet(f"color:{CYAN}; font-size:9px;")
        ori.addWidget(self._lbl_cam_info); ori.addStretch()
        bot.addLayout(ori)

        bot.addStretch()
        l.addLayout(bot)
        scroll.setWidget(w)
        outer_l.addWidget(scroll)
        return outer

    # ── Tab: SETUP ───────────────────────────────────────────────────────────
    def _build_tab_setup(self) -> QWidget:
        outer = QWidget()
        outer_l = QVBoxLayout(outer); outer_l.setContentsMargins(0,0,0,0); outer_l.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8, 6, 8, 6); l.setSpacing(8)

        # Calibration
        calib_gb = QGroupBox("CALIBRAÇÃO")
        cl = QVBoxLayout(calib_gb); cl.setSpacing(5)
        info = QLabel(
            "Clica 4 cantos do tabuleiro em ordem (WHITE_RIGHT):\n"
            "  1 → sup-esq (h8)    2 → sup-dir (h1)\n"
            "  3 → inf-dir (a1)    4 → inf-esq (a8)")
        info.setStyleSheet(f"color:{WHITE}; font-size:11px;")
        self.lbl_calib_pts = QLabel("Pontos: 0/4")
        self.lbl_calib_pts.setStyleSheet(f"color:{AMBER}; font-weight:bold; font-size:11px;")
        cb_row = QHBoxLayout(); cb_row.setSpacing(6)
        b_calc  = QPushButton("Calcular Homografia"); b_calc.setObjectName("primary"); b_calc.setFixedHeight(30)
        b_clear = QPushButton("Limpar");              b_clear.setFixedHeight(30)
        b_calc.clicked.connect(self._on_calib_calc)
        b_clear.clicked.connect(self._on_calib_clear)
        cb_row.addWidget(b_calc, 2); cb_row.addWidget(b_clear, 1)
        cl.addWidget(info); cl.addWidget(self.lbl_calib_pts); cl.addLayout(cb_row)

        # Grid
        grid_gb = QGroupBox("GRID")
        gl = QVBoxLayout(grid_gb); gl.setSpacing(4)

        def mk_row(name, lo, hi, init):
            row = QHBoxLayout(); row.setSpacing(4)
            lbl = QLabel(name); lbl.setFixedWidth(66)
            lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")
            val = QLabel(); val.setFixedWidth(54)
            val.setStyleSheet(f"color:{AMBER}; font-family:Consolas; font-size:10px;")
            sld = QSlider(Qt.Horizontal); sld.setRange(lo, hi); sld.setValue(int(init))
            row.addWidget(lbl); row.addWidget(sld, 1); row.addWidget(val)
            return row, val, sld

        r1,self.v_margin,self.s_margin = mk_row("MARGIN %",  0,   200, self.grid.margin_pct*10)
        r2,self.v_scale, self.s_scale  = mk_row("SCALE",     50,  150, self.grid.scale*100)
        r3,self.v_dx,    self.s_dx     = mk_row("SHIFT X",  -240, 240, self.grid.dx)
        r4,self.v_dy,    self.s_dy     = mk_row("SHIFT Y",  -240, 240, self.grid.dy)
        for sld in [self.s_margin, self.s_scale, self.s_dx, self.s_dy]:
            sld.valueChanged.connect(self._on_grid_changed)
        for r in [r1, r2, r3, r4]: gl.addLayout(r)

        nudge = QHBoxLayout(); nudge.setSpacing(2)
        for lbl_n, fn in [
            ("RST",  self._grid_reset),
            ("←",    lambda: self._nudge(dx=-2)),
            ("→",    lambda: self._nudge(dx=+2)),
            ("↑",    lambda: self._nudge(dy=-2)),
            ("↓",    lambda: self._nudge(dy=+2)),
            ("S−",   lambda: self._nudge(scale=-0.01)),
            ("S+",   lambda: self._nudge(scale=+0.01)),
        ]:
            b = QPushButton(lbl_n); b.clicked.connect(fn); b.setFixedHeight(24)
            nudge.addWidget(b)
        gl.addLayout(nudge)

        l.addWidget(calib_gb); l.addWidget(grid_gb); l.addStretch()
        self._update_grid_labels()
        scroll.setWidget(w)
        outer_l.addWidget(scroll)
        return outer

    # ── Tab: DEBUG ───────────────────────────────────────────────────────────
    def _build_tab_debug(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(6, 4, 6, 4); l.setSpacing(4)
        params = QLabel(
            f"Baseline:{BASELINE_NEED}  Votes:{STABLE_NEEDED}  Conf:{CONFIDENCE:.0%}  "
            f"Gap:{GAP_REQUIRED}  Score:{MIN_SCORE}  WD:{WATCHDOG_SEC:.0f}s  "
            f"Castle:{CASTLE_STABLE}  WConfirm:{WHITE_MOVE_CONFIRM}  Cooldown:{COOLDOWN_SEC}s"
        )
        params.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        params.setWordWrap(True)
        self.txt_debug = QTextEdit(); self.txt_debug.setReadOnly(True)
        self.txt_debug.setStyleSheet(
            f"font-family:Consolas; font-size:10px; color:{GREEN}; background:#050709;")
        l.addWidget(params)
        l.addWidget(self.txt_debug, 1)
        return w

    # ── Small helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _sec_lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{MUTED}; font-size:9px; letter-spacing:1px;")
        return lbl

    # ════════════════════════════════════════════════════ BACKEND ════════════
    def _init_backend(self):
        if not os.path.exists(MODEL_PATH):
            QMessageBox.critical(self,"Error",f"YOLO model not found:\n{MODEL_PATH}"); sys.exit(1)
        try:
            self.yolo = YOLO(MODEL_PATH); self.yolo_worker = YoloWorker(self.yolo)
            self.yolo_worker.result_ready.connect(self._on_yolo_result)
            self.yolo_worker.start()
            self.statusbar.showMessage("YOLO loaded", 3000)
        except Exception as ex:
            QMessageBox.critical(self,"YOLO",str(ex)); sys.exit(1)
        if os.path.exists(STOCKFISH_PATH):
            self.sf_worker = StockfishWorker(STOCKFISH_PATH, STOCKFISH_TIME)
            self.sf_worker.move_ready.connect(self._on_sf_move)
            self.sf_worker.error_occurred.connect(self._on_sf_error)
            self.sf_worker.engine_ready.connect(self._on_engine_ready)
            self.sf_worker.start()
        else:
            self._update_pill(self._pill_sf, "◉  ENGINE", False)
            QMessageBox.warning(self,"Stockfish",f"Not found:\n{STOCKFISH_PATH}")
        self._start_camera(); self._refresh_status(); self._draw_board()

    def _start_camera(self):
        for idx in CAM_CANDIDATES:
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = auto on most backends
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.pipeline = cap
                self._cam_thread = CameraThread(cap)
                self._cam_thread.start()
                self._update_pill(self._pill_cam, f"◉  {w}×{h}", True)
                self.statusbar.showMessage(f"Camera {w}x{h} (idx={idx})", 3000)
                return
            cap.release()
        self._update_pill(self._pill_cam, "◉  SEM CÂMERA", False)
        QMessageBox.warning(self, "Camera", "Nenhuma câmera detetada.")

    def _on_engine_ready(self, ok: bool):
        if ok:
            self._update_pill(self._pill_sf, "◉  STOCKFISH", True)
            self.statusbar.showMessage("Stockfish ready", 4000)
        else:
            self._update_pill(self._pill_sf, "◉  ENGINE ERR", False)
            self.statusbar.showMessage("Stockfish FAILED", 4000)

    @staticmethod
    def _camera_side_desc(side: str) -> str:
        return "▼ file a  (a1–a8 na base)" if side=="A_NEAR" else "▼ file h  (h1–h8 na base)"

    def _set_camera_side(self, side: str):
        self.grid.camera_side = side
        self.btn_cam_a.setChecked(side=="A_NEAR"); self.btn_cam_h.setChecked(side=="H_NEAR")
        self._lbl_cam_info.setText(self._camera_side_desc(side))
        self.statusbar.showMessage(f"Câmera: {side}", 2000); self._reset_baseline()

    def _set_tts_lang(self, lang: str):
        self._tts_lang = lang
        self.btn_lang_pt.setChecked(lang=="pt"); self.btn_lang_en.setChecked(lang=="en")
        self.btn_lang_fr.setChecked(lang=="fr")
        self._lbl_lang_voice.setText(TTS_VOICES[lang])
        self.statusbar.showMessage(f"TTS: {lang.upper()} — {TTS_VOICES[lang]}", 3000)

    def _set_orientation(self, orientation: str):
        self.grid.orientation = orientation
        self.btn_orient_right.setChecked(orientation=="WHITE_RIGHT")
        self.btn_orient_left.setChecked(orientation=="WHITE_LEFT")
        self.statusbar.showMessage(f"Orientação: {orientation}", 2000)
        self._reset_baseline()

    def _on_start_game(self):
        if not self.calibrated or self.H is None:
            self.statusbar.showMessage("Calibra primeiro!", 3000); return
        self.state = GameState.IDLE; self._reset_baseline(); self.btn_start.setEnabled(False)
        self.statusbar.showMessage("Jogo iniciado! A capturar baseline...", 5000)
        self._refresh_status()

    # ════════════════════════════════════════════════════ TICK ═══════════════
    def _tick(self):
        if not self._cam_thread: return
        frame = self._cam_thread.get_latest()
        if frame is None: return
        if self.H is None:
            vis = frame.copy()
            for i,(x,y) in enumerate(self.calib_pts):
                cv2.drawMarker(vis,(x,y),(0,200,100),cv2.MARKER_CROSS,24,2)
                cv2.circle(vis,(x,y),5,(255,80,0),-1)
                cv2.putText(vis,str(i+1),(x+10,y-12),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.putText(vis,"CALIBRA: clica 4 cantos (ver tab CALIB para ordem)",
                        (16,36),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2,cv2.LINE_AA)
            self.vid_lbl.set_image(vis); return
        warped = cv2.warpPerspective(frame, self.H, (FRAME_SIZE,FRAME_SIZE))
        vis = warped.copy()
        if self.chk_grid.isChecked(): vis = self.grid.draw_overlay(vis, FRAME_SIZE)
        if self.state == GameState.SETUP:
            self._draw_setup_banner(vis); self.vid_lbl.set_image(vis)
            now = time.time()
            if self.yolo_worker and now-self._last_submit >= 1.0/20.0:
                self._last_submit = now; self.yolo_worker.submit(warped)
            return
        if self.chk_boxes.isChecked() and self._last_det:
            for box,conf in zip(self._last_det.boxes, self._last_det.confs):
                x1,y1,x2,y2 = box
                sq = self.grid.pt_to_square((x1+x2)/2,(y1+y2)/2,FRAME_SIZE)
                if sq:
                    cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,220,80),1)
                    cv2.putText(vis,f"{sq} {conf:.2f}",(int(x1),int(y1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,220,80),1)
        if self._occ_base is None and self.calibrated:
            self._draw_hud_bar(vis, self._base_stable, BASELINE_NEED, "BASELINE", (255,180,0))
        elif self.state in (GameState.WAITING_MOVE, GameState.WAITING_OPERATOR) \
                and self._cand_stable > 0 and self.chk_auto.isChecked():
            threshold = CASTLE_STABLE if self._cand_is_castling else STABLE_NEEDED
            lbl = (f"-{sorted(self._cand_gone)} +{sorted(self._cand_appeared)}"
                   if self._cand_gone else "")
            color = (0,200,255) if self.state==GameState.WAITING_OPERATOR else (0,255,120)
            self._draw_hud_bar(vis, self._cand_stable, threshold, f"MOVE? {lbl}", color)
        if self._engine_instruction: vis = self._draw_engine_instruction(vis)
        vis = self._draw_tracker_overlay(vis)
        if self._illegal_active: vis = self._draw_illegal_overlay(vis)
        self.vid_lbl.set_image(vis)
        now = time.time()
        if self.yolo_worker and now-self._last_submit >= 1.0/20.0:
            self._last_submit = now; self.yolo_worker.submit(warped)

    # ═══════════════════════════════════════════════════════════ SPEECH ══════
    _SPEECH_STRINGS = {
        "pt": {
            "your_turn": "É a sua vez",
            "from_to":   "De {f} para {t}",
            "illegal":   "Lance ilegal! Volta atrás",
            "file": {'a':'á','b':'b','c':'c','d':'d',
                     'e':'é','f':'éfe','g':'g','h':'agá'},
        },
        "en": {
            "your_turn": "Your turn",
            "from_to":   "Move from {f} to {t}",
            "illegal":   "Illegal move! Take it back",
            "file": {'a':'ay','b':'bee','c':'see','d':'dee',
                     'e':'ee','f':'eff','g':'gee','h':'aitch'},
        },
        "fr": {
            "your_turn": "C'est votre tour",
            "from_to":   "De {f} vers {t}",
            "illegal":   "Coup illégal! Revenez en arrière",
            "file": {'a':'a','b':'bé','c':'cé','d':'dé',
                     'e':'e','f':'effe','g':'gé','h':'ache'},
        },
    }

    def _sq_to_speech(self, sq: str) -> str:
        file_map = self._SPEECH_STRINGS[self._tts_lang]["file"]
        return f"{file_map.get(sq[0].lower(), sq[0])} {sq[1]}"

    def _speak_now(self, text: str):
        def _worker():
            with self._speak_lock:
                self._speak_active = True; self._last_spoken = time.time(); tmp = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp: tmp = fp.name
                    voice = TTS_VOICES.get(self._tts_lang, TTS_VOICES["pt"])
                    asyncio.run(edge_tts.Communicate(text, voice, rate="-30%").save(tmp))
                    pygame.mixer.music.stop(); pygame.mixer.music.load(tmp)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy(): time.sleep(0.05)
                except Exception as e: print(f"[SPEAK] {e}")
                finally:
                    self._speak_active = False
                    if tmp:
                        try: pygame.mixer.music.stop(); time.sleep(0.05); os.unlink(tmp)
                        except: pass
        threading.Thread(target=_worker, daemon=True).start()

    def _announce_white_move(self, move: chess.Move, board_before: chess.Board):
        fs = self._sq_to_speech(chess.square_name(move.from_square))
        ts = self._sq_to_speech(chess.square_name(move.to_square))
        tmpl = self._SPEECH_STRINGS[self._tts_lang]["from_to"]
        self._speak_now(tmpl.format(f=fs, t=ts))

    def _announce_human_turn(self):
        if not self._human_turn_announced:
            self._human_turn_announced = True
            self._speak_now(self._SPEECH_STRINGS[self._tts_lang]["your_turn"])

    # ════════════════════════════════════════════════ ENGINE INSTRUCTION ═════
    def _sq_rect_px(self, sq_name: str) -> Tuple[int,int,int,int]:
        cx,cy = self.grid.square_to_img_coords(sq_name, FRAME_SIZE)
        x0,y0,x1,y1 = self.grid.rect(FRAME_SIZE)
        hw = (x1-x0)/8/2; hh = (y1-y0)/8/2
        return (max(0,int(cx-hw)), max(0,int(cy-hh)),
                min(FRAME_SIZE-1,int(cx+hw)), min(FRAME_SIZE-1,int(cy+hh)))

    def _draw_engine_instruction(self, vis: np.ndarray) -> np.ndarray:
        out = vis.copy()
        if self.state == GameState.WAITING_OPERATOR:
            cv2.putText(out,"AGUARDANDO MOVIMENTO DAS BRANCAS",
                        (16,FRAME_SIZE-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
        elapsed = time.time()-self._engine_instr_since
        pulse = 0.5+0.5*abs(np.sin(elapsed*np.pi*1.2))
        bh=72; by=0; ov=out.copy()
        cv2.rectangle(ov,(0,by),(FRAME_SIZE,by+bh),(6,14,26),-1)
        cv2.addWeighted(ov,0.82,out,0.18,0,out)
        cv2.line(out,(0,by+bh),(FRAME_SIZE,by+bh),
                 (int(200*pulse),int(160+95*pulse),255),2)
        cv2.putText(out,"MOVER BRANCAS",(16,by+22),cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,(100,180,255),1,cv2.LINE_AA)
        instr = self._engine_instruction or ""
        font=cv2.FONT_HERSHEY_DUPLEX; scale=1.10; thick=2
        (tw,th),_ = cv2.getTextSize(instr,font,scale,thick)
        tx=(FRAME_SIZE-tw)//2; ty=by+bh-10
        cv2.putText(out,instr,(tx+2,ty+2),font,scale,(0,0,0),thick+2,cv2.LINE_AA)
        cv2.putText(out,instr,(tx,ty),font,scale,
                    (int(180+75*pulse),int(230+25*pulse),255),thick,cv2.LINE_AA)
        if self._engine_instr_move:
            mv = self._engine_instr_move
            for sq_name,sq_color in [(chess.square_name(mv.from_square),(0,140,255)),
                                     (chess.square_name(mv.to_square),(0,220,120))]:
                r = self._sq_rect_px(sq_name); sq_ov = out.copy()
                cv2.rectangle(sq_ov,(r[0],r[1]),(r[2],r[3]),sq_color,-1)
                cv2.addWeighted(sq_ov,0.25+0.20*pulse,out,1-(0.25+0.20*pulse),0,out)
                cv2.rectangle(out,(r[0],r[1]),(r[2],r[3]),sq_color,max(2,int(3*pulse)))
                cv2.putText(out,sq_name.upper(),(r[0]+4,r[1]+18),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,sq_color,2,cv2.LINE_AA)
        return out

    def _draw_setup_banner(self, img):
        bh=46; by=10; ov=img.copy()
        cv2.rectangle(ov,(0,by),(FRAME_SIZE,by+bh),(10,18,30),-1)
        cv2.addWeighted(ov,0.82,img,0.18,0,img)
        cv2.line(img,(0,by),(FRAME_SIZE,by),(0,180,255),2)
        cv2.line(img,(0,by+bh),(FRAME_SIZE,by+bh),(0,180,255),1)
        txt = "MODO SETUP  --  Ajusta Grid e Calibracao, depois prime INICIAR JOGO"
        font=cv2.FONT_HERSHEY_SIMPLEX; (tw,th),_=cv2.getTextSize(txt,font,0.50,1)
        tx=max(8,(FRAME_SIZE-tw)//2); ty=by+bh//2+th//2
        cv2.putText(img,txt,(tx+1,ty+1),font,0.50,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,txt,(tx,ty),font,0.50,(0,200,255),1,cv2.LINE_AA)

    def _raw_sq_occupied(self, sq_name: str, det: DetResult) -> bool:
        for box,conf in zip(det.boxes, det.confs):
            if conf < CONFIDENCE*0.8: continue
            x1,y1,x2,y2 = box
            sq = self.grid.pt_to_square((x1+x2)/2,(y1+y2)/2,FRAME_SIZE)
            if sq == sq_name: return True
        return False

    def _draw_tracker_overlay(self, vis):
        if not self._sq_tracker or not self.chk_boxes.isChecked(): return vis
        out = vis.copy()
        for sq,val in self._sq_tracker.items():
            if val <= 0: continue
            r = self._sq_rect_px(sq)
            if val >= TRACK_OCC_THR: color,thick = (0,200,0),2
            elif val >= TRACK_FREE_THR: color,thick = (0,180,255),1
            else: continue
            bh = max(3,int((r[3]-r[1])*0.12)); bw = int((r[2]-r[0])*val/TRACK_MAX)
            cv2.rectangle(out,(r[0],r[3]-bh),(r[0]+bw,r[3]),color,-1)
            cv2.putText(out,f"{sq}:{val}",(r[0]+2,r[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,color,1,cv2.LINE_AA)
            cv2.rectangle(out,(r[0],r[1]),(r[2],r[3]),color,thick)
        return out

    def _draw_hud_bar(self, img, val, max_val, label, color):
        bx=16; by=FRAME_SIZE-44; bw=int(FRAME_SIZE*0.55); bh=18
        pct = min(val/max_val, 1.0)
        cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(30,30,30),-1)
        cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(60,60,60),1)
        if pct > 0: cv2.rectangle(img,(bx,by),(bx+int(bw*pct),by+bh),color,-1)
        cv2.putText(img,f"{label} {val}/{max_val}",(bx,by-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,color,1,cv2.LINE_AA)

    def _draw_illegal_overlay(self, vis):
        elapsed = time.time()-self._illegal_since; SHOW_SEC=3.5
        if elapsed > SHOW_SEC: self._illegal_active = False; return vis
        out = vis.copy(); pulse=abs(np.sin(elapsed*np.pi*4.0)); inten=0.25+0.55*pulse
        for sq_name in self._illegal_gone|self._illegal_appeared:
            r = self._sq_rect_px(sq_name); ov = out.copy()
            color = (0,0,220) if sq_name in self._illegal_gone else (0,80,255)
            cv2.rectangle(ov,(r[0],r[1]),(r[2],r[3]),color,-1)
            cv2.addWeighted(ov,inten*0.7,out,1-inten*0.7,0,out)
            cv2.rectangle(out,(r[0],r[1]),(r[2],r[3]),(0,0,255),max(2,int(4*pulse)))
            cx=(r[0]+r[2])//2; cy=(r[1]+r[3])//2; sz=(r[2]-r[0])//3
            cv2.line(out,(cx-sz,cy-sz),(cx+sz,cy+sz),(0,0,255),3,cv2.LINE_AA)
            cv2.line(out,(cx+sz,cy-sz),(cx-sz,cy+sz),(0,0,255),3,cv2.LINE_AA)
        bh=52; by=FRAME_SIZE//2-bh//2; ba=min(inten+0.3,0.85)
        bov=out.copy(); cv2.rectangle(bov,(0,by),(FRAME_SIZE,by+bh),(0,0,0),-1)
        cv2.addWeighted(bov,ba,out,1-ba,0,out)
        cv2.line(out,(0,by),(FRAME_SIZE,by),(0,0,220),2)
        cv2.line(out,(0,by+bh),(FRAME_SIZE,by+bh),(0,0,220),2)
        mt="LANCE ILEGAL  --  VOLTA ATRAS!"; font=cv2.FONT_HERSHEY_DUPLEX
        (tw,th),_=cv2.getTextSize(mt,font,0.72,2)
        tx=(FRAME_SIZE-tw)//2; ty=by+bh//2+th//2-2
        cv2.putText(out,mt,(tx+2,ty+2),font,0.72,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,mt,(tx,ty),font,0.72,
                    (80,80,255) if pulse>0.5 else (200,200,255),2,cv2.LINE_AA)
        (sw,sh),_=cv2.getTextSize(self._illegal_reason,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
        sx=(FRAME_SIZE-sw)//2; sy=by+bh+18
        cv2.putText(out,self._illegal_reason,(sx+1,sy+1),cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(out,self._illegal_reason,(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,(160,160,255),1,cv2.LINE_AA)
        remain=max(0.0,1.0-elapsed/SHOW_SEC); bw2=int(FRAME_SIZE*0.6*remain)
        bx2=(FRAME_SIZE-int(FRAME_SIZE*0.6))//2; bby=by+bh+30
        cv2.rectangle(out,(bx2,bby),(bx2+int(FRAME_SIZE*0.6),bby+6),(40,40,40),-1)
        cv2.rectangle(out,(bx2,bby),(bx2+bw2,bby+6),(0,0,200),-1)
        return out

    # ═══════════════════════════════════════════════════ OCCUPANCY HELPERS ═══

    def _boxes_to_occ(self, det: DetResult,
                      white_piece_squares: Optional[Set[str]] = None,
                      baseline: Optional[Set[str]] = None,
                      capture_candidates: Optional[Set[str]] = None) -> Set[str]:
        occ = set()
        gx0,gy0,gx1,gy1 = self.grid.rect(FRAME_SIZE)
        sq_w = max(10.0,(gx1-gx0)/8.0); sq_h = max(10.0,(gy1-gy0)/8.0)
        min_w,max_w = sq_w*0.18, sq_w*1.90
        min_h,max_h = sq_h*0.18, sq_h*1.90
        _cap = capture_candidates or set()
        _sw  = (white_piece_squares or set()) & (baseline or self._occ_base or set())
        for box,conf in zip(det.boxes, det.confs):
            x1,y1,x2,y2 = box; w,h = x2-x1,y2-y1
            if not (min_w<=w<=max_w and min_h<=h<=max_h): continue
            if conf < CONFIDENCE: continue
            ix,iy = w*0.25,h*0.25
            pts = [((x1+x2)/2,(y1+y2)/2),(x1+ix,y1+iy),(x2-ix,y1+iy),
                   (x1+ix,y2-iy),(x2-ix,y2-iy)]
            votes = {}
            for px,py in pts:
                sq = self.grid.pt_to_square(px,py,FRAME_SIZE)
                if sq: votes[sq] = votes.get(sq,0)+1
            if not votes: continue
            best = max(votes, key=lambda s: votes[s])
            if votes[best] < 3: continue
            if best in _cap: occ.add(best); continue
            if self.chk_mask.isChecked() and white_piece_squares and best in white_piece_squares:
                if best not in _sw and conf < WHITE_SQ_CONF: continue
            occ.add(best)
        return occ

    def _seed_tracker(self, occ: Set[str]):
        for sq in occ: self._sq_tracker[sq] = TRACK_MAX
        self._tracked_occ = set(occ)

    def _update_tracker(self, raw_occ: Set[str]) -> Set[str]:
        all_sq = set(self._sq_tracker.keys()) | raw_occ
        for sq in all_sq:
            cur = self._sq_tracker.get(sq,0)
            cur = min(cur+TRACK_UP,TRACK_MAX) if sq in raw_occ else max(cur-TRACK_DOWN,0)
            self._sq_tracker[sq] = cur
        new_occ = set(self._tracked_occ)
        for sq,val in self._sq_tracker.items():
            if val >= TRACK_OCC_THR: new_occ.add(sq)
            elif val <= TRACK_FREE_THR: new_occ.discard(sq)
        self._tracked_occ = new_occ
        for sq in [s for s,v in self._sq_tracker.items() if v==0]:
            del self._sq_tracker[sq]
        return new_occ

    def _stable_occ(self, smooth_occ: Set[str]) -> Set[str]:
        self._occ_history.append(set(smooth_occ))
        if len(self._occ_history) > OCC_HISTORY_LEN:
            self._occ_history.pop(0)
        N = len(self._occ_history)
        thresh = (N // 2) + 1
        all_sqs = set().union(*self._occ_history)
        return {sq for sq in all_sqs
                if sum(1 for h in self._occ_history if sq in h) >= thresh}

    # ════════════════════════════════════════════════════ YOLO HANDLER ═══════
    def _on_yolo_result(self, det: DetResult):
        self._last_det = det
        with self.board_lock: board_snap = self.board.copy()
        capture_cands: Set[str] = set()
        if self.state not in (GameState.SETUP, GameState.ENGINE_THINK):
            capture_cands = {
                chess.square_name(mv.to_square)
                for mv in board_snap.legal_moves
                if board_snap.is_capture(mv)
                   and board_snap.piece_at(mv.from_square)
                   and board_snap.piece_at(mv.from_square).color == board_snap.turn
            }
        white_sqs: Optional[Set[str]] = None
        if self.chk_mask.isChecked() and board_snap.turn == HUMAN_COLOR:
            white_sqs = {chess.square_name(sq) for sq,p in board_snap.piece_map().items()
                         if p.color == chess.WHITE}

        raw_occ    = self._boxes_to_occ(det, white_sqs, baseline=self._occ_base,
                                         capture_candidates=capture_cands)
        self._raw_occ = raw_occ

        smooth_occ = self._update_tracker(raw_occ)
        stable_occ = self._stable_occ(smooth_occ)

        self._debug_dirty = True
        top = sorted([(v,s) for s,v in self._sq_tracker.items() if v>0],reverse=True)[:8]
        self._debug_lines = [
            f"state        : {self.state.value}",
            f"turn         : {'WHITE' if board_snap.turn==chess.WHITE else 'BLACK'}",
            f"frames       : {self._yolo_count}",
            f"raw occ ({len(raw_occ):2d}): {sorted(raw_occ)}",
            f"smooth  ({len(smooth_occ):2d}): {sorted(smooth_occ)}",
            f"stable  ({len(stable_occ):2d}): {sorted(stable_occ)}",
            f"top counters : {[(s,v) for v,s in top]}",
            f"baseline ({len(self._occ_base) if self._occ_base else 0:2d}): "
            f"{'set' if self._occ_base is not None else '--'}",
            f"cand votes   : {self._cand_stable}/{CASTLE_STABLE if self._cand_is_castling else STABLE_NEEDED}",
            f"cand move    : {self._cand_move.uci() if self._cand_move else '--'}",
            f"cand gone    : {sorted(self._cand_gone) if self._cand_gone else '--'}",
            f"cand appeared: {sorted(self._cand_appeared) if self._cand_appeared else '--'}",
            f"engine instr : {self._engine_instruction or '--'}",
            f"white confirm: {self._white_confirm_frames}/{WHITE_MOVE_CONFIRM}",
            f"watchdog     : {time.time()-self._last_activity:.0f}s ago",
            f"orientation  : {self.grid.orientation}",
            f"camera_side  : {self.grid.camera_side}",
        ]

        if self.state == GameState.SETUP: return
        if not self.chk_auto.isChecked() or not self.calibrated: return
        self._yolo_count += 1

        # ── BASELINE ──────────────────────────────────────────────────────────
        if self._occ_base is None:
            self._process_baseline(smooth_occ); return

        # ── CONFIRMAÇÃO DO MOVIMENTO DAS BRANCAS ─────────────────────────────
        if self.state == GameState.WAITING_OPERATOR and self._engine_instr_move is not None:
            mv = self._engine_instr_move
            from_name  = chess.square_name(mv.from_square)
            to_name    = chess.square_name(mv.to_square)
            from_empty  = not self._raw_sq_occupied(from_name, det)
            to_occupied =     self._raw_sq_occupied(to_name,   det)
            self.bar_baseline.setMaximum(WHITE_MOVE_CONFIRM)
            self.bar_baseline.setFormat(f"%v/{WHITE_MOVE_CONFIRM}")

            if from_empty and to_occupied:
                self._white_confirm_frames += 1
                self.bar_baseline.setValue(min(self._white_confirm_frames, WHITE_MOVE_CONFIRM))
                self.lbl_phase.setText(
                    f"Movendo BRANCA: {from_name.upper()}→{to_name.upper()} "
                    f"({self._white_confirm_frames}/{WHITE_MOVE_CONFIRM})")
                if self._white_confirm_frames >= WHITE_MOVE_CONFIRM:
                    self._confirm_white_move_done()
            else:
                if self._white_confirm_frames > 0:
                    self._white_confirm_frames = 0
                    self.bar_baseline.setValue(0)
                self.lbl_phase.setText(
                    f"MOVER BRANCAS {from_name.upper()}→{to_name.upper()} "
                    f"(origem vazia={from_empty}, destino ocupado={to_occupied})")
            return

        # WAITING_OPERATOR sem instrução → desbloqueia
        if self.state == GameState.WAITING_OPERATOR and self._engine_instr_move is None:
            self.state = GameState.IDLE
            self._reset_baseline()
            return

        with self.board_lock:
            thinking       = self.state == GameState.ENGINE_THINK
            is_idle_engine = (self.state == GameState.IDLE
                              and self.board.turn == ENGINE_COLOR)

        active_detection = (self.state == GameState.WAITING_MOVE
                            and board_snap.turn == HUMAN_COLOR
                            and self._engine_instruction is None)

        if not active_detection:
            if not thinking and self._occ_base is not None:
                diff = len(smooth_occ.symmetric_difference(self._occ_base))
                if diff <= NOISE_TOL:
                    self._occ_base = smooth_occ
            if is_idle_engine and not self._sf_thinking:
                QTimer.singleShot(200, self._stockfish_play)
            return

        self._last_activity = time.time()
        self._process_detection(stable_occ, capture_cands, board_snap=board_snap)

    def _confirm_white_move_done(self):
        self._engine_instruction   = None
        self._engine_instr_move    = None
        self._white_confirm_frames = 0
        self._human_turn_announced = False
        self._last_activity        = time.time()

        self._sq_tracker.clear()
        self._tracked_occ.clear()
        self._occ_history.clear()
        self._vote_window.clear()
        self._cand_stable       = 0
        self._cand_move         = None
        self._cand_gone         = None
        self._cand_appeared     = None
        self._cand_is_castling  = False

        self._occ_base    = None
        self._base_cand   = None
        self._base_stable = 0
        self.bar_baseline.setMaximum(BASELINE_NEED)
        self.bar_baseline.setFormat(f"%v/{BASELINE_NEED}")
        self.bar_baseline.setValue(0)
        self.bar_stable.setMaximum(STABLE_NEEDED)
        self.bar_stable.setFormat(f"%v/{STABLE_NEEDED}")
        self.bar_stable.setValue(0)

        self.state = GameState.IDLE
        self.lbl_phase.setText("Branca confirmada! A re-capturar baseline...")
        self.statusbar.showMessage("Brancas moveram! A re-capturar baseline...", 5000)
        self._refresh_status()

    # ── Baseline ──────────────────────────────────────────────────────────────
    def _process_baseline(self, smooth_occ: Set[str]):
        if self._base_cand is None:
            self._base_cand = smooth_occ; self._base_stable = 1
        else:
            diff = len(smooth_occ.symmetric_difference(self._base_cand))
            if diff <= BASELINE_NOISE_TOL:
                self._base_stable += 1; self._base_cand = smooth_occ
            else:
                self._base_cand = smooth_occ; self._base_stable = 1
        self.bar_baseline.setMaximum(BASELINE_NEED)
        self.bar_baseline.setFormat(f"%v/{BASELINE_NEED}")
        self.bar_baseline.setValue(min(self._base_stable, BASELINE_NEED))
        self.lbl_phase.setText(f"A capturar baseline... {self._base_stable}/{BASELINE_NEED}")
        self._refresh_status()

        if self._base_stable < BASELINE_NEED:
            return

        self._occ_base    = set(self._base_cand)
        self._base_stable = BASELINE_NEED
        self._seed_tracker(self._occ_base)
        self._occ_history.clear()
        self._last_activity = time.time()

        with self.board_lock: turn = self.board.turn

        if self.state == GameState.IDLE and turn == ENGINE_COLOR:
            self.state = GameState.ENGINE_THINK
            self.lbl_phase.setText("Stockfish a pensar...")
            QTimer.singleShot(100, self._stockfish_play)

        elif self.state == GameState.ENGINE_THINK:
            self.lbl_phase.setText("Baseline OK — Stockfish a calcular...")

        elif self.state == GameState.WAITING_OPERATOR:
            instr = self._engine_instruction
            self.lbl_phase.setText(f"Baseline OK — MOVER BRANCAS: {instr}")
            self.statusbar.showMessage(f"Baseline OK! MOVER BRANCAS: {instr}", 8000)

        elif turn == HUMAN_COLOR:
            self.state = GameState.WAITING_MOVE
            self.lbl_phase.setText("Aguardando tua jogada (Pretas)...")
            self.statusbar.showMessage("Baseline OK! Faz a tua jogada (Pretas).", 6000)
            self._announce_human_turn()

        else:
            self.state = GameState.WAITING_MOVE
            self.lbl_phase.setText("Aguardando jogada...")
            self.statusbar.showMessage(f"Baseline: {len(self._occ_base)} casas.", 6000)

        self._refresh_status()

    # ── Castling ──────────────────────────────────────────────────────────────
    def _match_castling(self, board_snap: chess.Board,
                        gone: Set[str], appeared: Set[str]) -> Tuple[bool, Optional[chess.Move]]:
        for mv in board_snap.legal_moves:
            if not board_snap.is_castling(mv): continue
            exp_gone, exp_appeared = _expected_delta(board_snap, mv)
            if (len(gone.symmetric_difference(exp_gone)) <= 1
                    and len(appeared.symmetric_difference(exp_appeared)) <= 1):
                return True, mv
        return False, None

    # ── Move detection ────────────────────────────────────────────────────────
    def _process_detection(self, stable_occ: Set[str],
                           capture_cands: Optional[Set[str]] = None,
                           board_snap: Optional[chess.Board] = None):
        now = time.time()
        if now - self._last_confirm < COOLDOWN_SEC: return
        if board_snap is None:
            with self.board_lock: board_snap = self.board.copy()
        if board_snap.turn != HUMAN_COLOR or self._occ_base is None: return

        if capture_cands is None: capture_cands = set()
        # During a capture, the destination square may briefly drop from stable_occ
        # while YOLO transitions from detecting the white piece to the arriving black piece.
        # Remove capture destinations from "gone" so this transient blind-spot doesn't
        # penalise the capture move and block its score from reaching MIN_SCORE.
        gone     = (self._occ_base - stable_occ) - capture_cands
        # Same filter for appeared: if a capture destination appears as "new" it means
        # the white piece wasn't in the baseline (undetected), but the capture is valid.
        appeared = (stable_occ - self._occ_base) - capture_cands

        if len(gone) + len(appeared) == 0:
            self._reset_candidate(); return

        is_castling, _ = self._match_castling(board_snap, gone, appeared)
        move, score    = infer_move(board_snap, gone, appeared, stable_occ)

        if move is None or score < MIN_SCORE:
            self._reset_candidate()
            self.lbl_candidate.setText(
                f"candidate: -- (score {score:.0f} < {MIN_SCORE}) "
                f"gone={sorted(gone)} app={sorted(appeared)}")
            return

        self._vote_window.append(move.uci())
        if len(self._vote_window) > VOTE_WINDOW: self._vote_window.pop(0)
        votes = self._vote_window.count(move.uci())

        self._cand_move        = move
        self._cand_gone        = gone
        self._cand_appeared    = appeared
        self._cand_occ         = stable_occ
        self._cand_is_castling = is_castling
        self._cand_stable      = votes
        threshold = CASTLE_STABLE if is_castling else STABLE_NEEDED

        self.lbl_candidate.setText(
            f"candidate: {move.uci()}  score={score:.0f}  votes={votes}/{threshold}  "
            f"gone={sorted(gone)} appeared={sorted(appeared)}")
        self.bar_stable.setMaximum(threshold)
        self.bar_stable.setValue(min(votes, threshold))

        if votes < threshold: return

        # ── Promoção de peão ──────────────────────────────────────────────────
        if move.promotion is not None:
            dlg = PromotionDialog(self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                move = chess.Move(move.from_square, move.to_square,
                                  promotion=dlg.chosen)
            else:
                move = chess.Move(move.from_square, move.to_square,
                                  promotion=chess.QUEEN)
            piece_names = {
                chess.QUEEN: "Dama", chess.ROOK: "Torre",
                chess.BISHOP: "Bispo", chess.KNIGHT: "Cavalo",
            }
            self.statusbar.showMessage(
                f"Promoção: {piece_names.get(move.promotion, '?')}", 4000)

        # ── Confirmação final ─────────────────────────────────────────────────
        with self.board_lock:
            if move not in self.board.legal_moves:
                self._illegal_active   = True
                self._illegal_since    = time.time()
                self._illegal_gone     = gone
                self._illegal_appeared = appeared
                self._illegal_reason   = f"Lance ilegal: {move.uci()}"
                self.statusbar.showMessage(f"Lance PRETO ilegal: {move.uci()}", 6000)
                self._speak_now(self._SPEECH_STRINGS[self._tts_lang]["illegal"])
                self._reset_candidate(); return
            board_before = self.board.copy()
            self.board.push(move)

        self._last_confirm  = now
        self._last_activity = now
        san = board_before.san(move)
        self.moves_list.addItem(san); self.moves_list.scrollToBottom()
        self.statusbar.showMessage(f"Jogada PRETA confirmada: {san}", 6000)
        self.lbl_candidate.setText(f"confirmado: {move.uci()} ({san})")

        self._sq_tracker.clear()
        self._tracked_occ.clear()
        self._occ_history.clear()
        self._vote_window.clear()
        self._cand_stable      = 0
        self._cand_move        = None
        self._cand_gone        = None
        self._cand_appeared    = None
        self._cand_is_castling = False
        self.bar_stable.setMaximum(STABLE_NEEDED)
        self.bar_stable.setValue(0)

        self._occ_base    = None
        self._base_cand   = None
        self._base_stable = 0
        self.bar_baseline.setMaximum(BASELINE_NEED)
        self.bar_baseline.setFormat(f"%v/{BASELINE_NEED}")
        self.bar_baseline.setValue(0)
        self._human_turn_announced = False

        self.state = GameState.IDLE
        self._refresh_status(); self._draw_board()

        if self.sf_worker and not self._sf_thinking:
            QTimer.singleShot(250, self._stockfish_play)

    # ════════════════════════════════════════════════════ STOCKFISH ══════════
    def _stockfish_play(self):
        if not self.sf_worker:
            self.statusbar.showMessage("Stockfish não configurado",4000); return
        if not self.sf_worker._engine_ok:
            self.statusbar.showMessage("Stockfish a iniciar...",2000)
            QTimer.singleShot(1000, self._stockfish_play); return
        with self.board_lock:
            if self.board.turn != ENGINE_COLOR or self._sf_thinking: return
            bc = self.board.copy()
        self._sf_thinking = True; self.state = GameState.ENGINE_THINK
        self._engine_instruction = None; self._engine_instr_move = None
        self.lbl_phase.setText("Stockfish a calcular...")
        self._refresh_status(); self.sf_worker.request_move(bc)

    def _on_sf_move(self, move: chess.Move):
        if not self._sf_thinking: return
        self._sf_thinking = False
        with self.board_lock:
            if move not in self.board.legal_moves:
                self.statusbar.showMessage(f"Stockfish illegal: {move.uci()}",5000)
                self.state = GameState.ERROR; self._refresh_status(); return
            board_before = self.board.copy(); self.board.push(move)
        from_sq = chess.square_name(move.from_square).upper()
        to_sq   = chess.square_name(move.to_square).upper()
        self._engine_instruction  = f"{from_sq}  ->  {to_sq}"
        self._engine_instr_move   = move
        self._engine_instr_since  = time.time()
        self.state = GameState.WAITING_OPERATOR

        self._sq_tracker.clear()
        self._tracked_occ.clear()
        self._occ_history.clear()
        self._vote_window.clear()
        self._cand_stable      = 0
        self._cand_move        = None
        self._cand_gone        = None
        self._cand_appeared    = None
        self._cand_is_castling = False

        self._occ_base    = None
        self._base_cand   = None
        self._base_stable = 0
        self.bar_baseline.setMaximum(BASELINE_NEED)
        self.bar_baseline.setFormat(f"%v/{BASELINE_NEED}")
        self.bar_baseline.setValue(0)
        self.bar_stable.setMaximum(STABLE_NEEDED)
        self.bar_stable.setFormat(f"%v/{STABLE_NEEDED}")
        self.bar_stable.setValue(0)
        self._white_confirm_frames = 0
        self._last_activity = time.time()
        self.lbl_phase.setText("MOVER BRANCAS conforme instrução...")
        self._announce_white_move(move, board_before)
        self.statusbar.showMessage(
            f"Stockfish: {move.uci()}  |  Mova BRANCA {from_sq} -> {to_sq}", 9000)
        self._refresh_status(); self._draw_board()

    def _on_sf_error(self, msg: str):
        self._sf_thinking = False
        self.statusbar.showMessage(f"Stockfish error: {msg}",5000)
        self.state = GameState.ERROR; self.lbl_phase.setText(f"ERROR: {msg}")
        self._refresh_status()

    # ════════════════════════════════════════════════════ MANUAL FALLBACK ════
    def _on_capture(self):
        if not self.calibrated:
            self.statusbar.showMessage("Calibra primeiro!",2000); return
        if self.state == GameState.SETUP:
            self.statusbar.showMessage("Prime INICIAR JOGO primeiro!",2000); return
        if self.state == GameState.ENGINE_THINK:
            self.statusbar.showMessage("Aguarda o Stockfish...",2000); return
        self._occ_before = set(self._raw_occ)
        self.btn_capture.setEnabled(False); self.btn_confirm.setEnabled(True)
        self.statusbar.showMessage("ANTES guardado. Faz a jogada e prime SPACE.", 5000)

    def _on_confirm(self):
        if self._occ_before is None:
            self.statusbar.showMessage("Captura primeiro",2000); return
        occ_after = set(self._raw_occ)
        gone      = self._occ_before - occ_after
        appeared  = occ_after - self._occ_before
        self._occ_before = None
        self.btn_capture.setEnabled(True); self.btn_confirm.setEnabled(False)
        with self.board_lock:
            mv,sc = infer_move(self.board, gone, appeared, occ_after)
            if mv is None or sc <= 0:
                self.statusbar.showMessage(f"Jogada nao reconhecida (score={sc:.0f})",4000)
                return
            was_operator = (self.state == GameState.WAITING_OPERATOR)
            try: self.board.push(mv)
            except Exception:
                self.statusbar.showMessage("Lance ilegal!",2500); return
        self._sq_tracker.clear(); self._tracked_occ.clear()
        self._occ_history.clear()
        self._vote_window.clear()
        self._cand_stable      = 0
        self._cand_move        = None
        self._cand_gone        = None
        self._cand_appeared    = None
        self._cand_is_castling = False
        self._occ_base    = None; self._base_cand = None; self._base_stable = 0
        self.bar_baseline.setMaximum(BASELINE_NEED); self.bar_baseline.setFormat(f"%v/{BASELINE_NEED}")
        self.bar_baseline.setValue(0)
        self.bar_stable.setMaximum(STABLE_NEEDED); self.bar_stable.setFormat(f"%v/{STABLE_NEEDED}")
        self.bar_stable.setValue(0)
        self._last_activity = time.time()
        if was_operator:
            self.state = GameState.IDLE
            self._engine_instruction = None; self._engine_instr_move = None
            self.statusbar.showMessage(f"[MANUAL] Brancas: {mv.uci()} — baseline...",7000)
        else:
            self.state = GameState.IDLE
            self.statusbar.showMessage(f"[MANUAL] Tu: {mv.uci()}  |  Stockfish a pensar...",7000)
            self._refresh_status(); self._draw_board()
            QTimer.singleShot(50, self._stockfish_play)
        self._human_turn_announced = False
        self._refresh_status(); self._draw_board()

    # ════════════════════════════════════════════════════ CALIBRATION ════════
    def _on_vid_click(self, x, y):
        if self.calibrated or len(self.calib_pts) >= 4: return
        self.calib_pts.append((x,y))
        self.lbl_calib_pts.setText(f"Points: {len(self.calib_pts)}/4")
        if len(self.calib_pts) < 4:
            self.statusbar.showMessage(f"Ponto {len(self.calib_pts)}/4",1500)
        else:
            self.statusbar.showMessage("4 pontos prontos. Calcula a homografia.",3000)

    def _on_calib_calc(self):
        if len(self.calib_pts) != 4:
            self.statusbar.showMessage(f"Preciso de 4 pontos ({len(self.calib_pts)})",2000)
            return
        M   = FRAME_SIZE*0.06; src = np.array(self.calib_pts, dtype=np.float32)
        dst = np.array([[M,M],[FRAME_SIZE-M,M],
                        [FRAME_SIZE-M,FRAME_SIZE-M],[M,FRAME_SIZE-M]],dtype=np.float32)
        try:
            self.H = cv2.getPerspectiveTransform(src,dst); self.calibrated = True
            self.state = GameState.SETUP; self.btn_start.setEnabled(True)
            self.statusbar.showMessage("Homografia calculada! Ajusta grid e prime INICIAR.",8000)
            self._refresh_status()
        except Exception as ex:
            self.statusbar.showMessage(f"Homografia: {ex}",4000)

    def _on_calib_clear(self):
        self.calib_pts = []; self.calibrated = False; self.H = None
        self.state = GameState.SETUP; self.lbl_calib_pts.setText("Points: 0/4")
        self.btn_start.setEnabled(False); self._reset_baseline()
        self.statusbar.showMessage("Calibracao limpa.",2000); self._refresh_status()

    # ════════════════════════════════════════════════════ GRID ═══════════════
    def _on_grid_changed(self,*_):
        self.grid.margin_pct = self.s_margin.value()/10.0
        self.grid.scale      = self.s_scale.value()/100.0
        self.grid.dx         = int(self.s_dx.value())
        self.grid.dy         = int(self.s_dy.value())
        self._update_grid_labels()

    def _grid_reset(self):
        o = self.grid.orientation; cs = self.grid.camera_side
        self.grid = GridModel(orientation=o, camera_side=cs)
        self.s_margin.setValue(int(self.grid.margin_pct*10))
        self.s_scale.setValue(int(self.grid.scale*100))
        self.s_dx.setValue(self.grid.dx); self.s_dy.setValue(self.grid.dy)
        self._update_grid_labels()

    def _nudge(self, dx=0, dy=0, scale=0.0):
        self.grid.dx += int(dx); self.grid.dy += int(dy)
        self.grid.scale = max(0.5, min(1.5, self.grid.scale+scale))
        self.s_dx.setValue(self.grid.dx); self.s_dy.setValue(self.grid.dy)
        self.s_scale.setValue(int(self.grid.scale*100)); self._update_grid_labels()

    def _update_grid_labels(self):
        self.v_margin.setText(f"{self.grid.margin_pct:.1f}%")
        self.v_scale.setText(f"{self.grid.scale:.2f}x")
        self.v_dx.setText(f"{self.grid.dx}px"); self.v_dy.setText(f"{self.grid.dy}px")

    # ════════════════════════════════════════════════════ HELPERS ════════════
    def _reset_candidate(self):
        self._cand_stable      = 0
        self._cand_move        = None
        self._cand_gone        = None
        self._cand_appeared    = None
        self._cand_is_castling = False
        self._vote_window = []
        self.bar_stable.setMaximum(STABLE_NEEDED)
        self.bar_stable.setFormat(f"%v/{STABLE_NEEDED}")
        self.bar_stable.setValue(0)

    def _reset_baseline(self):
        self._occ_base         = None
        self._base_cand        = None
        self._base_stable      = 0
        self._white_confirm_frames = 0
        self._vote_window      = []
        self._cand_stable      = 0
        self._cand_move        = None
        self._cand_gone        = None
        self._cand_appeared    = None
        self._cand_is_castling = False
        self._sq_tracker       = {}
        self._tracked_occ      = set()
        self._occ_history.clear()
        self.bar_baseline.setMaximum(BASELINE_NEED)
        self.bar_baseline.setFormat(f"%v/{BASELINE_NEED}")
        self.bar_baseline.setValue(0)
        self.bar_stable.setMaximum(STABLE_NEEDED)
        self.bar_stable.setFormat(f"%v/{STABLE_NEEDED}")
        self.bar_stable.setValue(0)
        self.lbl_phase.setText("Re-capturing baseline...")
        self.lbl_candidate.setText("candidate: --")
        self._last_activity = time.time()

    def _consume_pending_ui(self):
        with self._pending_lock: self._pending_ui_refresh = False
        self._refresh_status(); self._draw_board()

    def _consume_pending_stockfish(self):
        with self._pending_lock: self._pending_stockfish = False
        self._stockfish_play()

    def _on_reset(self):
        self._sf_thinking = False
        with self.board_lock: self.board = chess.Board()
        self.state = GameState.IDLE if self.calibrated else GameState.SETUP
        self._occ_before = None; self._last_det = None; self._last_confirm = 0.0
        self._engine_instruction = None; self._engine_instr_move = None
        self._white_confirm_frames = 0; self._human_turn_announced = False
        self._illegal_active = False
        self.btn_capture.setEnabled(True); self.btn_confirm.setEnabled(False)
        if self.calibrated: self.btn_start.setEnabled(False)
        self._reset_baseline(); self._refresh_status(); self._draw_board()
        self.statusbar.showMessage("Reset completo.", 3000)

    def _on_auto_toggle(self, enabled):
        self.statusbar.showMessage("Auto ON" if enabled else "Auto OFF", 2000)
        if enabled and self.state != GameState.SETUP: self._reset_baseline()

    def _flush_debug(self):
        if self._debug_dirty:
            self.txt_debug.setPlainText("\n".join(self._debug_lines))
            self._debug_dirty = False

    def _refresh_status(self):
        with self.board_lock:
            turn    = self.board.turn; is_chk = self.board.is_check()
            is_mate = self.board.is_checkmate()
            is_draw = self.board.is_stalemate() or self.board.is_insufficient_material()
            mvs     = [m.uci() for m in self.board.move_stack[-20:][::-1]]
        self.moves_list.clear(); self.moves_list.addItems(mvs)
        if is_mate:
            winner = "WHITE" if turn == chess.BLACK else "BLACK"
            self.status_widget.set_state("warn",f"CHECKMATE -- {winner} WINS","Game over")
            self.state = GameState.GAME_OVER; return
        if is_draw:
            self.status_widget.set_state("warn","DRAW","Stalemate/insufficient material")
            self.state = GameState.GAME_OVER; return
        if self.state == GameState.SETUP:
            self.status_widget.set_state("setup","MODO SETUP",
                                         "Calibra, ajusta grid e prime INICIAR JOGO")
        elif self._occ_base is None and self.state not in (GameState.SETUP,
                                                            GameState.ENGINE_THINK):
            self.status_widget.set_state("baseline","CALIBRATING...",
                                         "Capturing initial board state")
        elif self.state == GameState.ENGINE_THINK:
            self.status_widget.set_state("engine","STOCKFISH THINKING",
                                         "Calculating best move...")
        elif self.state == GameState.WAITING_OPERATOR:
            instr = self._engine_instruction or ""
            self.status_widget.set_state("engine","MOVER BRANCAS",
                                         f"{instr} -- depois joga pretas")
        elif self.state == GameState.ERROR:
            self.status_widget.set_state("warn","ERROR","Check debug tab")
        elif turn == HUMAN_COLOR and self.state == GameState.WAITING_MOVE:
            chk = "  CHECK" if is_chk else ""
            self.status_widget.set_state("human",f"YOUR TURN{chk}",
                                         "Blacks -- auto-detection active")
        else:
            self.status_widget.set_state("idle","AGUARDANDO","Board idle")

    # ════════════════════════════════════════════════════ BOARD DRAW ═════════
    def _draw_board(self):
        px = BOARD_BORDER*2 + SQUARE_PX*8
        img = np.full((px,px,3), C_BORDER, dtype=np.uint8)
        for rank in range(8):
            for file in range(8):
                light = (file+rank)%2 == 1
                col_canvas = 7-file; row_canvas = 7-rank
                x1 = BOARD_BORDER+col_canvas*SQUARE_PX
                y1 = BOARD_BORDER+row_canvas*SQUARE_PX
                cv2.rectangle(img,(x1,y1),(x1+SQUARE_PX,y1+SQUARE_PX),
                              C_LIGHT if light else C_DARK,-1)
                if file == 0:
                    cv2.putText(img,str(8-rank),(x1+2,y1+12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.35,
                                C_DARK if light else C_LIGHT,1)
                if rank == 7:
                    cv2.putText(img,FILES[7-file],(x1+SQUARE_PX-10,y1+SQUARE_PX-3),
                                cv2.FONT_HERSHEY_SIMPLEX,0.35,
                                C_DARK if light else C_LIGHT,1)
        with self.board_lock:
            pm      = dict(self.board.piece_map())
            last_mv = self.board.peek() if self.board.move_stack else None
        if last_mv:
            for sq in [last_mv.from_square, last_mv.to_square]:
                fi = chess.square_file(sq); ri = chess.square_rank(sq)
                col_canvas = 7-fi; row_canvas = 7-ri
                x1 = BOARD_BORDER+col_canvas*SQUARE_PX
                y1 = BOARD_BORDER+row_canvas*SQUARE_PX
                ov = img[y1:y1+SQUARE_PX, x1:x1+SQUARE_PX].copy()
                cv2.addWeighted(ov,0.55,np.full_like(ov,(255,200,50)),0.45,0,
                                img[y1:y1+SQUARE_PX, x1:x1+SQUARE_PX])
        for sq,piece in pm.items():
            im = self.piece_imgs.get(piece.symbol())
            if im is None: continue
            fi = chess.square_file(sq); ri = chess.square_rank(sq)
            col_canvas = 7-fi; row_canvas = 7-ri
            overlay_png(img,im,BOARD_BORDER+col_canvas*SQUARE_PX,
                        BOARD_BORDER+row_canvas*SQUARE_PX)
        self.board_lbl.setPixmap(bgr_to_qpixmap(img))

    # ═══════════════════════════════════════════════════════ KEYBOARD ════════
    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            if self.btn_start.isEnabled(): self._on_start_game()
        elif k == Qt.Key_Space:
            if self.btn_capture.isEnabled():   self._on_capture()
            elif self.btn_confirm.isEnabled(): self._on_confirm()
        elif k == Qt.Key_R: self._on_reset()
        elif k == Qt.Key_B: self._reset_baseline()
        else: super().keyPressEvent(e)

    # ═══════════════════════════════════════════════════════ CLOSE ═══════════
    def closeEvent(self, e):
        for fn in [
            lambda: self.yolo_worker and self.yolo_worker.stop(),
            lambda: self.sf_worker and self.sf_worker.stop(),
            lambda: self._cam_thread and self._cam_thread.stop(),
            lambda: pygame.mixer.quit(),
        ]:
            try: fn()
            except: pass
        e.accept()

# ════════════════════════════════════════════════════════════════ MAIN ════════
if __name__ == "__main__":
    app = QApplication(sys.argv); apply_theme(app)
    win = ChessBotWindow(); win.show(); win.activateWindow(); win.raise_()
    sys.exit(app.exec())
