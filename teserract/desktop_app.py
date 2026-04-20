from __future__ import annotations

import json
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import END, BooleanVar, IntVar, Label, Listbox, StringVar, Tk, filedialog, messagebox
from tkinter import scrolledtext
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_FILES = None
    TkinterDnD = None
    DND_AVAILABLE = False

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    PROJECT_ROOT = Path(getattr(sys, "_MEIPASS"))
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import OUTPUTS_DIR  # noqa: E402
from pipelines import (  # noqa: E402
    analyze_pdf_pages,
    annotate_ocr_boxes,
    batch_process_folder,
    configure,
    detect_document_edges,
    detect_table,
    get_database_overview,
    pdf_file_to_images,
    query_database,
    run_specialized_pipeline,
    save_batch_to_database,
    save_excel_report,
    save_table_csv,
)


class OCRDesktopApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("OCR Desktop Studio Pro")
        self.root.geometry("1420x900")
        self.root.minsize(1200, 780)

        configure()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        self.lang_var = StringVar(value="tur")
        self.image_path_var = StringVar()
        self.image_mode_var = StringVar(value="otomatik")
        self.pdf_path_var = StringVar()
        self.batch_folder_var = StringVar()
        self.batch_recursive_var = BooleanVar(value=True)
        self.batch_db_var = BooleanVar(value=True)
        self.db_path_var = StringVar(value=str(OUTPUTS_DIR / "ocr_results.db"))
        self.db_search_var = StringVar()
        self.output_dir_var = StringVar(value=str(OUTPUTS_DIR))
        self.camera_index_var = IntVar(value=0)
        self.camera_status_var = StringVar(value="Kamera kapalı")
        self.theme_var = StringVar(value="dark")

        self.camera_capture = None
        self.camera_running = False
        self.current_camera_frame = None
        self.preview_images: dict[str, ImageTk.PhotoImage] = {}
        self.result_payloads: dict[str, dict] = {}
        self.result_artifacts: dict[str, list[Path]] = {}
        self.card_containers: dict[str, ttk.Frame] = {}
        self.table_views: dict[str, ttk.Treeview] = {}
        self.log_output: scrolledtext.ScrolledText | None = None
        self.log_entries: list[tuple[str, str]] = []
        self.log_filter_var = StringVar()
        self.drop_queue: list[Path] = []

        self._configure_theme()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_theme(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        bg = "#141a23"
        panel = "#1d2633"
        text = "#ecf0f6"
        accent = "#4ea1ff"
        muted = "#95a4b8"
        self.root.configure(bg=bg)
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=panel)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Muted.TLabel", background=bg, foreground=muted)
        style.configure("Header.TLabel", background=bg, foreground=text, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=7, relief="flat")
        style.map("TButton", background=[("active", accent)])
        style.configure("Accent.TButton", background=accent, foreground="white")
        style.configure("TCheckbutton", background=bg, foreground=text)
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), background=panel, foreground=text)
        style.map("TNotebook.Tab", background=[("selected", accent)], foreground=[("selected", "white")])
        style.configure("TLabelframe", background=bg, foreground=text)
        style.configure("TLabelframe.Label", background=bg, foreground=text)
        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="OCR Desktop Studio Pro", style="Header.TLabel").pack(side="left")
        ttk.Label(top, text="Dil:", style="Muted.TLabel").pack(side="left", padx=(20, 6))
        ttk.Combobox(top, textvariable=self.lang_var, values=["tur", "eng", "tur+eng"], width=12, state="readonly").pack(side="left")
        ttk.Label(top, text="Çıktı klasörü:", style="Muted.TLabel").pack(side="left", padx=(20, 6))
        ttk.Label(top, textvariable=self.output_dir_var).pack(side="left")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.image_tab = ttk.Frame(notebook, padding=12)
        self.pdf_tab = ttk.Frame(notebook, padding=12)
        self.camera_tab = ttk.Frame(notebook, padding=12)
        self.batch_tab = ttk.Frame(notebook, padding=12)
        self.db_tab = ttk.Frame(notebook, padding=12)
        self.settings_tab = ttk.Frame(notebook, padding=12)

        notebook.add(self.image_tab, text="Görsel OCR")
        notebook.add(self.pdf_tab, text="PDF OCR")
        notebook.add(self.camera_tab, text="Kamera")
        notebook.add(self.batch_tab, text="Toplu Klasör")
        notebook.add(self.db_tab, text="Veritabanı")
        notebook.add(self.settings_tab, text="Ayarlar")

        self._build_image_tab()
        self._build_pdf_tab()
        self._build_camera_tab()
        self._build_batch_tab()
        self._build_db_tab()
        self._build_settings_tab()

        log_frame = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        log_frame.pack(fill="both", padx=12, pady=(0, 12))
        log_header = ttk.Frame(log_frame, style="Card.TFrame")
        log_header.pack(fill="x")
        ttk.Label(log_header, text="Log / İşlem Geçmişi", style="Header.TLabel").pack(side="left")
        ttk.Label(log_header, text="Filtre:", style="Muted.TLabel").pack(side="left", padx=(18, 6))
        log_filter_entry = ttk.Entry(log_header, textvariable=self.log_filter_var, width=24)
        log_filter_entry.pack(side="left")
        log_filter_entry.bind("<KeyRelease>", lambda _event: self._refresh_log_view())
        ttk.Button(log_header, text="Temizle", command=self._clear_log).pack(side="right")
        self.log_output = scrolledtext.ScrolledText(log_frame, wrap="word", height=8, font=("Consolas", 9), bg="#0f141c", fg="#ecf0f6", insertbackground="#ecf0f6")
        self.log_output.pack(fill="both", expand=True, pady=(8, 0))
        self._log("Uygulama başlatıldı.")

    def _build_two_panel_tab(self, parent: ttk.Frame) -> tuple[ttk.Frame, ttk.Frame]:
        left = ttk.Frame(parent, style="Card.TFrame", padding=10)
        right = ttk.Frame(parent, style="Card.TFrame", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))
        return left, right

    def _build_image_tab(self) -> None:
        left, right = self._build_two_panel_tab(self.image_tab)
        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x")

        ttk.Label(controls, text="Görsel yolu:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.image_path_var, width=80).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Seç", command=self._pick_image).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Kuyruğu Çalıştır", command=lambda: self._run_thread(self._run_drop_queue)).grid(row=0, column=3, padx=4)
        ttk.Label(controls, text="Mod:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            controls,
            textvariable=self.image_mode_var,
            values=["otomatik", "bounding_boxes", "tablo", "form", "cmr"],
            state="readonly",
            width=24,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=(10, 0))
        ttk.Button(controls, text="Çalıştır", style="Accent.TButton", command=lambda: self._run_thread(self._run_image_tab)).grid(row=1, column=2, padx=4, pady=(10, 0))
        ttk.Button(controls, text="Kuyruğu Temizle", command=self._clear_drop_queue).grid(row=1, column=3, padx=4, pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(left, text="Sürükle-bırak kuyruğu", style="Header.TLabel").pack(anchor="w", pady=(12, 6))
        queue_actions = ttk.Frame(left, style="Card.TFrame")
        queue_actions.pack(fill="x", pady=(0, 6))
        ttk.Button(queue_actions, text="Seçileni Sil", command=self._remove_selected_queue_item).pack(side="left")
        ttk.Button(queue_actions, text="Yukarı", command=lambda: self._move_queue_item(-1)).pack(side="left", padx=(6, 0))
        ttk.Button(queue_actions, text="Aşağı", command=lambda: self._move_queue_item(1)).pack(side="left", padx=(6, 0))

        self.queue_listbox = Listbox(left, height=5, bg="#0f141c", fg="#ecf0f6", selectbackground="#4ea1ff")
        self.queue_listbox.pack(fill="x", pady=(0, 10))
        self._enable_drop_target(self.queue_listbox, self._handle_dropped_queue)

        ttk.Label(left, text="Görsel önizleme", style="Header.TLabel").pack(anchor="w", pady=(12, 6))
        self.image_preview = Label(left, bg="#0f141c", bd=0)
        self.image_preview.pack(fill="both", expand=True)

        self._enable_drop_target(self.image_preview, self._handle_dropped_image)

        self.image_output = self._build_output_panel(right, "image")

    def _build_pdf_tab(self) -> None:
        left, right = self._build_two_panel_tab(self.pdf_tab)
        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x")

        ttk.Label(controls, text="PDF yolu:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.pdf_path_var, width=80).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Seç", command=self._pick_pdf).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Çalıştır", style="Accent.TButton", command=lambda: self._run_thread(self._run_pdf_tab)).grid(row=1, column=2, padx=4, pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(left, text="PDF ilk sayfa önizleme", style="Header.TLabel").pack(anchor="w", pady=(12, 6))
        self.pdf_preview = Label(left, bg="#0f141c", bd=0)
        self.pdf_preview.pack(fill="both", expand=True)

        self._enable_drop_target(self.pdf_preview, self._handle_dropped_pdf)

        self.pdf_output = self._build_output_panel(right, "pdf")

    def _build_camera_tab(self) -> None:
        left, right = self._build_two_panel_tab(self.camera_tab)
        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x")

        ttk.Label(controls, text="Kamera index:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(controls, from_=0, to=10, textvariable=self.camera_index_var, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Button(controls, text="Başlat", command=self._start_camera).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Durdur", command=self._stop_camera).grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="Kareyi Analiz Et", style="Accent.TButton", command=lambda: self._run_thread(self._analyze_camera_frame)).grid(row=0, column=4, padx=4)
        ttk.Label(controls, textvariable=self.camera_status_var, style="Muted.TLabel").grid(row=1, column=0, columnspan=5, sticky="w", pady=(10, 0))

        ttk.Label(left, text="Canlı kamera önizleme", style="Header.TLabel").pack(anchor="w", pady=(12, 6))
        self.camera_preview = Label(left, bg="#0f141c", bd=0)
        self.camera_preview.pack(fill="both", expand=True)

        self.camera_output = self._build_output_panel(right, "camera")

    def _build_batch_tab(self) -> None:
        left, right = self._build_two_panel_tab(self.batch_tab)
        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x")

        ttk.Label(controls, text="Klasör yolu:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.batch_folder_var, width=80).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Seç", command=self._pick_folder).grid(row=0, column=2, padx=4)
        ttk.Checkbutton(controls, text="Alt klasörleri tara", variable=self.batch_recursive_var).grid(row=1, column=1, sticky="w", padx=6, pady=(10, 0))
        ttk.Checkbutton(controls, text="Veritabanına kaydet", variable=self.batch_db_var).grid(row=1, column=1, sticky="e", padx=6, pady=(10, 0))
        ttk.Button(controls, text="Çalıştır", style="Accent.TButton", command=lambda: self._run_thread(self._run_batch_tab)).grid(row=1, column=2, padx=4, pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(left, text="Toplu klasör çıktıları", style="Header.TLabel").pack(anchor="w", pady=(12, 6))
        self.batch_preview = Label(left, bg="#0f141c", bd=0, text="Toplu işlem sonrası özet burada listelenir.", fg="#95a4b8")
        self.batch_preview.pack(fill="both", expand=True)

        self._enable_drop_target(self.batch_preview, self._handle_dropped_folder)

        self.batch_output = self._build_output_panel(right, "batch")

    def _build_db_tab(self) -> None:
        left, right = self._build_two_panel_tab(self.db_tab)
        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x")

        ttk.Label(controls, text="DB yolu:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.db_path_var, width=76).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Yenile", command=lambda: self._run_thread(self._refresh_database_tab)).grid(row=0, column=2, padx=4)
        ttk.Label(controls, text="Ara:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.db_search_var, width=44).grid(row=1, column=1, sticky="w", padx=6, pady=(10, 0))
        ttk.Button(controls, text="Sorgula", style="Accent.TButton", command=lambda: self._run_thread(self._query_database_tab)).grid(row=1, column=2, padx=4, pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(left, text="Veritabanı özeti", style="Header.TLabel").pack(anchor="w", pady=(12, 6))
        self.db_preview = Label(left, bg="#0f141c", bd=0, text="Veritabanı özetleri burada görünür.", fg="#95a4b8")
        self.db_preview.pack(fill="both", expand=True)

        self.db_output = self._build_output_panel(right, "db", allow_export=False)

    def _build_settings_tab(self) -> None:
        card = ttk.Frame(self.settings_tab, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="Uygulama ayarları", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(card, text="Çıktı klasörü:").grid(row=1, column=0, sticky="w", pady=(14, 0))
        ttk.Entry(card, textvariable=self.output_dir_var, width=90).grid(row=1, column=1, sticky="ew", padx=8, pady=(14, 0))
        ttk.Button(card, text="Seç", command=self._pick_output_dir).grid(row=1, column=2, pady=(14, 0))

        ttk.Label(card, text="Varsayılan DB yolu:").grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(card, textvariable=self.db_path_var, width=90).grid(row=2, column=1, sticky="ew", padx=8, pady=(12, 0))

        ttk.Label(card, text="Tema:").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(card, textvariable=self.theme_var, values=["dark"], state="readonly", width=20).grid(row=3, column=1, sticky="w", padx=8, pady=(12, 0))

        ttk.Label(card, text="Kamera index:").grid(row=4, column=0, sticky="w", pady=(12, 0))
        ttk.Spinbox(card, from_=0, to=10, textvariable=self.camera_index_var, width=10).grid(row=4, column=1, sticky="w", padx=8, pady=(12, 0))

        ttk.Button(card, text="Ayarları uygula", style="Accent.TButton", command=self._apply_settings).grid(row=5, column=1, sticky="w", padx=8, pady=(18, 0))

        info = (
            "Bu masaüstü sürümünde görsel önizleme, kamera sekmesi, dışa aktarma butonları ve ayarlar paneli aktiftir.\n"
            "EXE içinde de aynı yapı çalışacak şekilde yollar paketlemeye uygun hazırlanmıştır."
        )
        ttk.Label(card, text=info, style="Muted.TLabel", justify="left").grid(row=6, column=0, columnspan=3, sticky="w", pady=(20, 0))
        card.columnconfigure(1, weight=1)

    def _build_output_panel(self, parent: ttk.Frame, key: str, allow_export: bool = True) -> scrolledtext.ScrolledText:
        header = ttk.Frame(parent, style="Card.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Sonuçlar", style="Header.TLabel").pack(side="left")
        if allow_export:
            ttk.Button(header, text="JSON kaydet", command=lambda: self._export_json_payload(key)).pack(side="right", padx=(6, 0))
            ttk.Button(header, text="Dosya dışa aktar", command=lambda: self._export_latest_artifact(key)).pack(side="right")

        cards = ttk.Frame(parent, style="Card.TFrame")
        cards.pack(fill="x", pady=(10, 6))
        self.card_containers[key] = cards

        table_frame = ttk.Frame(parent, style="Card.TFrame")
        table_frame.pack(fill="x", pady=(0, 8))
        table = ttk.Treeview(table_frame, show="headings", height=5)
        table.pack(fill="x", expand=True)
        self.table_views[key] = table
        table_frame.pack_forget()
        table._container = table_frame  # type: ignore[attr-defined]

        output = scrolledtext.ScrolledText(parent, wrap="word", font=("Consolas", 10), bg="#0f141c", fg="#ecf0f6", insertbackground="#ecf0f6")
        output.pack(fill="both", expand=True, pady=(10, 0))
        return output

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_entries.append((timestamp, message))
        self._refresh_log_view()

    def _refresh_log_view(self) -> None:
        if self.log_output is None:
            return

        filter_text = self.log_filter_var.get().strip().lower()

        def update() -> None:
            self.log_output.delete("1.0", "end")
            for timestamp, message in self.log_entries:
                rendered = f"[{timestamp}] {message}"
                if filter_text and filter_text not in rendered.lower():
                    continue
                self.log_output.insert("end", f"• {rendered}\n")
            self.log_output.see("end")

        self.root.after(0, update)

    def _clear_log(self) -> None:
        self.log_entries.clear()
        self._refresh_log_view()

    def _enable_drop_target(self, widget, callback) -> None:
        if not DND_AVAILABLE or not hasattr(widget, "drop_target_register"):
            return
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", lambda event: callback(event.data))

    def _extract_drop_paths(self, raw_data: str) -> list[Path]:
        try:
            items = list(self.root.tk.splitlist(raw_data))
        except Exception:
            items = [raw_data]

        paths: list[Path] = []
        for item in items:
            cleaned = str(item).strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                cleaned = cleaned[1:-1]
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            path = Path(cleaned)
            if path.exists():
                paths.append(path)
        return paths

    def _handle_dropped_image(self, raw_data: str) -> None:
        paths = self._extract_drop_paths(raw_data)
        image_paths = [path for path in paths if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        if not image_paths:
            return
        self._add_paths_to_queue(image_paths)
        self.image_path_var.set(str(image_paths[0]))
        image = cv2.imread(str(image_paths[0]))
        if image is not None:
            self._update_preview("image", self.image_preview, image)
        self._log(f"{len(image_paths)} görsel sürükle-bırak ile kuyruğa eklendi.")

    def _handle_dropped_pdf(self, raw_data: str) -> None:
        paths = self._extract_drop_paths(raw_data)
        pdf_paths = [path for path in paths if path.suffix.lower() == ".pdf"]
        if pdf_paths:
            self.pdf_path_var.set(str(pdf_paths[0]))
            self._add_paths_to_queue(pdf_paths)
            self._log(f"{len(pdf_paths)} PDF sürükle-bırak ile kuyruğa eklendi.")

    def _handle_dropped_folder(self, raw_data: str) -> None:
        paths = self._extract_drop_paths(raw_data)
        folders = [path for path in paths if path.is_dir()]
        if folders:
            self.batch_folder_var.set(str(folders[0]))
            self._log(f"Klasör sürükle-bırak ile seçildi: {folders[0].name}")

    def _handle_dropped_queue(self, raw_data: str) -> None:
        paths = [path for path in self._extract_drop_paths(raw_data) if path.is_file()]
        if not paths:
            return
        self._add_paths_to_queue(paths)
        self._log(f"Kuyruğa {len(paths)} dosya eklendi.")

    def _add_paths_to_queue(self, paths: list[Path]) -> None:
        existing = {str(path) for path in self.drop_queue}
        for path in paths:
            if str(path) not in existing:
                self.drop_queue.append(path)
        self._refresh_queue_list()

    def _refresh_queue_list(self) -> None:
        if not hasattr(self, "queue_listbox"):
            return
        self.queue_listbox.delete(0, END)
        for index, path in enumerate(self.drop_queue, start=1):
            self.queue_listbox.insert(END, f"{index:02d}. {path.name}")

    def _clear_drop_queue(self) -> None:
        self.drop_queue.clear()
        self._refresh_queue_list()
        self._log("Dosya kuyruğu temizlendi.")

    def _remove_selected_queue_item(self) -> None:
        if not hasattr(self, "queue_listbox"):
            return
        selection = self.queue_listbox.curselection()
        if not selection:
            messagebox.showinfo("Bilgi", "Silmek için kuyruktan bir dosya seçin.")
            return
        index = int(selection[0])
        removed = self.drop_queue.pop(index)
        self._refresh_queue_list()
        self._log(f"Kuyruktan silindi: {removed.name}")

    def _move_queue_item(self, direction: int) -> None:
        if not hasattr(self, "queue_listbox"):
            return
        selection = self.queue_listbox.curselection()
        if not selection:
            messagebox.showinfo("Bilgi", "Taşımak için kuyruktan bir dosya seçin.")
            return
        index = int(selection[0])
        target_index = index + direction
        if target_index < 0 or target_index >= len(self.drop_queue):
            return
        self.drop_queue[index], self.drop_queue[target_index] = self.drop_queue[target_index], self.drop_queue[index]
        self._refresh_queue_list()
        self.queue_listbox.selection_set(target_index)
        self._log(f"Kuyruk sırası değişti: {self.drop_queue[target_index].name}")

    def _run_drop_queue(self) -> None:
        if not self.drop_queue:
            messagebox.showinfo("Bilgi", "Kuyrukta dosya yok.")
            return

        processed_rows: list[list[str]] = []
        for path in list(self.drop_queue):
            if path.suffix.lower() == ".pdf":
                self.pdf_path_var.set(str(path))
                self._run_pdf_tab()
                processed_rows.append([path.name, "pdf", "tamamlandı"])
            else:
                self.image_path_var.set(str(path))
                self._run_image_tab()
                processed_rows.append([path.name, "image", "tamamlandı"])

        self._show_cards("image", [("Kuyruk", str(len(processed_rows))), ("Durum", "Tamamlandı")])
        self._show_table("image", processed_rows)
        self._log(f"Dosya kuyruğu işlendi. Adet: {len(processed_rows)}")

    def _clear_cards(self, key: str) -> None:
        container = self.card_containers.get(key)
        if container is None:
            return
        for child in container.winfo_children():
            child.destroy()

    def _show_cards(self, key: str, cards: list[tuple[str, str]]) -> None:
        self._clear_cards(key)
        container = self.card_containers.get(key)
        if container is None:
            return
        for index, (title, value) in enumerate(cards):
            palette = self._card_palette(title, value)
            card = ttk.Frame(container, style="Card.TFrame", padding=10)
            card.grid(row=0, column=index, padx=(0, 8), sticky="nsew")
            Label(card, text="", bg=palette["accent"], width=2, height=1).pack(anchor="w", fill="x")
            Label(card, text=f"{palette['icon']} {title}", bg=palette["background"], fg=palette["muted"], padx=10, pady=6, anchor="w").pack(anchor="w", fill="x")
            badge_color = self._status_color(title, value)
            if badge_color:
                Label(card, text=value, bg=badge_color, fg="white", padx=10, pady=4).pack(anchor="w", pady=(6, 0))
            else:
                Label(card, text=value, bg=palette["background"], fg=palette["text"], padx=10, pady=6, anchor="w", font=("Segoe UI", 10, "bold")).pack(anchor="w", fill="x", pady=(2, 0))
            container.columnconfigure(index, weight=1)

    def _card_palette(self, title: str, value: str) -> dict[str, str]:
        normalized_title = title.lower()
        normalized_value = value.lower()
        base = {
            "background": "#1d2633",
            "text": "#ecf0f6",
            "muted": "#a8b6c9",
            "accent": "#4ea1ff",
            "icon": "•",
        }

        if normalized_title in {"belge", "durum"}:
            mapping = {
                "kimlik": ("#1d4ed8", "🪪"),
                "fatura_veya_fis": ("#c2410c", "🧾"),
                "form": ("#7c3aed", "📋"),
                "tablo": ("#0f766e", "📊"),
                "cmr": ("#fbbf24", "🌍"),
                "genel_belge": ("#475569", "📄"),
                "tamamlandı": ("#16a34a", "✅"),
            }
            accent, icon = mapping.get(normalized_value, ("#4ea1ff", "📌"))
            base.update({"accent": accent, "icon": icon})
        elif normalized_title in {"kelime", "satır", "sayfa", "dosya", "sonuç"}:
            base.update({"accent": "#2563eb", "icon": "#️⃣"})
        elif normalized_title in {"pipeline", "mod"}:
            base.update({"accent": "#7c3aed", "icon": "⚙️"})
        elif normalized_title in {"excel", "json", "db"}:
            base.update({"accent": "#0f766e", "icon": "💾"})
        elif normalized_title == "kenar":
            base.update({"accent": "#16a34a" if normalized_value == "var" else "#dc2626", "icon": "📐"})
        else:
            base.update({"icon": "🔹"})
        return base

    def _status_color(self, title: str, value: str) -> str | None:
        normalized_title = title.lower()
        normalized_value = value.lower()
        if normalized_title in {"belge", "durum"}:
            mapping = {
                "kimlik": "#2563eb",
                "fatura_veya_fis": "#d97706",
                "form": "#7c3aed",
                "tablo": "#0f766e",
                "cmr": "#f59e0b",
                "genel_belge": "#475569",
                "tamamlandı": "#16a34a",
            }
            return mapping.get(normalized_value, "#4ea1ff")
        if normalized_title == "kenar":
            return "#16a34a" if normalized_value == "var" else "#dc2626"
        return None

    def _show_table(self, key: str, rows: list[list[str]] | None) -> None:
        table = self.table_views.get(key)
        if table is None:
            return
        container = getattr(table, "_container", None)
        if container is None:
            return
        for item in table.get_children():
            table.delete(item)
        if not rows:
            container.pack_forget()
            return
        max_cols = max((len(row) for row in rows), default=0)
        columns = [f"col_{index + 1}" for index in range(max_cols)]
        table.configure(columns=columns)
        for index, column in enumerate(columns, start=1):
            table.heading(column, text=f"Kolon {index}")
            table.column(column, width=130, anchor="w")
        for row in rows[:8]:
            padded = row + [""] * (max_cols - len(row))
            table.insert("", "end", values=padded)
        container.pack(fill="x", pady=(0, 8))

    def _output_dir(self) -> Path:
        output_dir = Path(self.output_dir_var.get().strip() or str(OUTPUTS_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _pick_image(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if file_path:
            self.image_path_var.set(file_path)
            image = cv2.imread(file_path)
            if image is not None:
                self._update_preview("image", self.image_preview, image)
                self._log(f"Görsel seçildi: {Path(file_path).name}")

    def _pick_pdf(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if file_path:
            self.pdf_path_var.set(file_path)
            self._log(f"PDF seçildi: {Path(file_path).name}")

    def _pick_folder(self) -> None:
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.batch_folder_var.set(folder_path)
            self._log(f"Klasör seçildi: {Path(folder_path).name}")

    def _pick_output_dir(self) -> None:
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_dir_var.set(folder_path)
            self.db_path_var.set(str(Path(folder_path) / "ocr_results.db"))

    def _apply_settings(self) -> None:
        self._output_dir()
        self.camera_status_var.set(f"Ayarlar uygulandı. Kamera index: {self.camera_index_var.get()}")
        self._log("Ayarlar güncellendi.")
        messagebox.showinfo("Ayarlar", "Ayarlar kaydedildi.")

    def _run_thread(self, target) -> None:
        thread = threading.Thread(target=self._safe_call, args=(target,), daemon=True)
        thread.start()

    def _safe_call(self, target) -> None:
        try:
            target()
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: messagebox.showerror("Hata", str(exc)))

    def _set_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        def update() -> None:
            widget.delete("1.0", "end")
            widget.insert("1.0", text)

        self.root.after(0, update)

    def _remember_result(self, key: str, payload: dict, artifacts: list[Path] | None = None) -> None:
        self.result_payloads[key] = payload
        self.result_artifacts[key] = artifacts or []

    def _export_json_payload(self, key: str) -> None:
        payload = self.result_payloads.get(key)
        if not payload:
            messagebox.showinfo("Bilgi", "Önce bir işlem çalıştırın.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if save_path:
            Path(save_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log(f"JSON dışa aktarıldı: {Path(save_path).name}")
            messagebox.showinfo("Kaydedildi", save_path)

    def _export_latest_artifact(self, key: str) -> None:
        artifacts = [path for path in self.result_artifacts.get(key, []) if path.exists()]
        if not artifacts:
            messagebox.showinfo("Bilgi", "Dışa aktarılacak dosya yok.")
            return
        source = artifacts[0]
        save_path = filedialog.asksaveasfilename(initialfile=source.name, defaultextension=source.suffix)
        if save_path:
            shutil.copy2(source, save_path)
            self._log(f"Dosya dışa aktarıldı: {Path(save_path).name}")
            messagebox.showinfo("Kaydedildi", save_path)

    def _update_preview(self, key: str, widget: Label, image_bgr, max_size: tuple[int, int] = (620, 520)) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(max_size)
        photo = ImageTk.PhotoImage(image=image)
        self.preview_images[key] = photo

        def update() -> None:
            widget.configure(image=photo, text="")
            widget.image = photo

        self.root.after(0, update)

    def _run_image_tab(self) -> None:
        path = Path(self.image_path_var.get().strip())
        if not path.exists():
            raise FileNotFoundError("Geçerli bir görsel seçin.")

        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError("Görsel okunamadı.")

        self._update_preview("image", self.image_preview, image)
        lang = self.lang_var.get()
        mode = self.image_mode_var.get()
        output_dir = self._output_dir()
        artifacts: list[Path] = []

        if mode == "bounding_boxes":
            result = annotate_ocr_boxes(image, lang=lang)
            output_image = output_dir / "desktop_bounding_boxes.png"
            cv2.imwrite(str(output_image), result["annotated"])
            self._update_preview("image_result", self.image_preview, result["annotated"])
            artifacts = [output_image]
            payload = {
                "mode": mode,
                "word_count": len(result["words"]),
                "output_image": str(output_image),
                "words_preview": result["words"][:30],
            }
            self._show_cards("image", [("Mod", "Bounding Boxes"), ("Kelime", str(len(result["words"])))])
            self._show_table("image", None)
        elif mode == "tablo":
            result = detect_table(image, lang=lang)
            output_image = output_dir / "desktop_table_annotated.png"
            output_csv = output_dir / "desktop_table_rows.csv"
            cv2.imwrite(str(output_image), result["annotated"])
            save_table_csv(result["rows"], output_csv)
            self._update_preview("image_result", self.image_preview, result["annotated"])
            artifacts = [output_csv, output_image]
            payload = {
                "mode": mode,
                "row_count": len(result["rows"]),
                "output_image": str(output_image),
                "output_csv": str(output_csv),
                "rows": result["rows"],
            }
            self._show_cards("image", [("Mod", "Tablo"), ("Satır", str(len(result["rows"])))])
            self._show_table("image", result["rows"])
        elif mode == "form":
            result = run_specialized_pipeline(image, lang=lang, forced_type="form")
            self._update_preview("image_result", self.image_preview, result["edge_result"]["annotated"])
            payload = {
                "mode": mode,
                "document_type": result["classification"]["type"],
                "pipeline": result["pipeline_name"],
                "fields": result["specialized"].get("form_fields"),
                "text": result["corrected_text"],
            }
            field_cards = [(key, str(value)) for key, value in (result["specialized"].get("form_fields") or {}).items()][:4]
            self._show_cards("image", [("Belge", result["classification"]["type"]), ("Pipeline", result["pipeline_name"]), *field_cards])
            self._show_table("image", None)
        elif mode == "cmr":
            result = run_specialized_pipeline(image, lang=lang, forced_type="cmr")
            self._update_preview("image_result", self.image_preview, result["edge_result"]["annotated"])
            payload = {
                "mode": mode,
                "document_type": result["classification"]["type"],
                "pipeline": result["pipeline_name"],
                "fields": result["specialized"].get("cmr_fields"),
                "text": result["corrected_text"],
            }
            cmr_fields = result["specialized"].get("cmr_fields") or {}
            field_cards = [(key, str(value)) for key, value in cmr_fields.items() if value][:6]
            self._show_cards("image", [("Belge", "cmr"), ("Pipeline", result["pipeline_name"]), *field_cards])
            self._show_table("image", None)
        else:
            result = run_specialized_pipeline(image, lang=lang)
            self._update_preview("image_result", self.image_preview, result["edge_result"]["annotated"])
            payload = {
                "mode": mode,
                "document_type": result["classification"]["type"],
                "pipeline": result["pipeline_name"],
                "fields": result["fields"],
                "insights": result["insights"],
                "specialized": result["specialized"],
                "text": result["corrected_text"],
            }
            field_cards = [(key, str(value)) for key, value in (result.get("fields") or {}).items() if value][:4]
            self._show_cards("image", [("Belge", result["classification"]["type"]), ("Pipeline", result["pipeline_name"]), *field_cards])
            self._show_table("image", result["specialized"].get("table_rows"))

        self._remember_result("image", payload, artifacts)
        self._log(f"Görsel işlendi. Mod: {mode}")
        self._set_text(self.image_output, json.dumps(payload, ensure_ascii=False, indent=2))

    def _run_pdf_tab(self) -> None:
        path = Path(self.pdf_path_var.get().strip())
        if not path.exists():
            raise FileNotFoundError("Geçerli bir PDF seçin.")

        output_dir = self._output_dir()
        images = pdf_file_to_images(path)
        if images:
            self._update_preview("pdf", self.pdf_preview, images[0])
        report = analyze_pdf_pages(images, lang=self.lang_var.get())
        excel_path = save_excel_report(report, output_dir / "desktop_pdf_report.xlsx")
        json_path = output_dir / "desktop_pdf_report.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        payload = {
            "page_count": report["page_count"],
            "document_type": report["document_type"],
            "excel_report": str(excel_path),
            "json_report": str(json_path),
            "pages": [
                {
                    "page": page["page"],
                    "classification": page["classification"]["type"],
                    "pipeline": page["pipeline_name"],
                }
                for page in report["pages"]
            ],
        }
        self._remember_result("pdf", payload, [excel_path, json_path])
        self._show_cards("pdf", [("Sayfa", str(report["page_count"])), ("Belge", report["document_type"]), ("Excel", excel_path.name), ("JSON", json_path.name)])
        first_table = report["tables"][0]["rows"] if report.get("tables") and report["tables"][0].get("rows") else None
        self._show_table("pdf", first_table)
        self._log(f"PDF işlendi: {path.name}")
        self._set_text(self.pdf_output, json.dumps(payload, ensure_ascii=False, indent=2))

    def _run_batch_tab(self) -> None:
        folder = Path(self.batch_folder_var.get().strip())
        if not folder.exists():
            raise FileNotFoundError("Geçerli bir klasör seçin.")

        output_dir = self._output_dir()
        report = batch_process_folder(
            folder,
            lang=self.lang_var.get(),
            include_images=True,
            include_pdfs=True,
            recursive=self.batch_recursive_var.get(),
        )
        excel_path = save_excel_report(report, output_dir / "desktop_batch_report.xlsx")
        db_result = None
        if self.batch_db_var.get():
            db_result = save_batch_to_database(report, self.db_path_var.get(), batch_name="desktop_batch")

        preview_text = [f"{item.get('file_name')} -> {item.get('document_type')}" for item in report["items"][:20]]
        self.root.after(0, lambda: self.batch_preview.configure(text="\n".join(preview_text) or "Kayıt yok.", image=""))
        payload = {
            "folder": report["folder"],
            "file_count": report["file_count"],
            "document_type_counts": report["document_type_counts"],
            "excel_report": str(excel_path),
            "db_result": db_result,
            "items": [
                {
                    "file_name": item.get("file_name"),
                    "file_type": item.get("file_type"),
                    "document_type": item.get("document_type"),
                    "pipeline_name": item.get("pipeline_name"),
                }
                for item in report["items"]
            ],
        }
        self._remember_result("batch", payload, [excel_path])
        self._show_cards("batch", [("Dosya", str(report["file_count"])), ("Tür", str(len(report["document_type_counts"]))), ("Excel", excel_path.name)])
        table_rows = [[item.get("file_name", ""), item.get("document_type", ""), item.get("pipeline_name", "")] for item in report["items"][:8]]
        self._show_table("batch", table_rows)
        self._log(f"Toplu klasör işlendi: {folder.name}")
        self._set_text(self.batch_output, json.dumps(payload, ensure_ascii=False, indent=2))

    def _refresh_database_tab(self) -> None:
        overview = get_database_overview(self.db_path_var.get())
        summary = f"Toplam kayıt: {overview['total_documents']}\n" + "\n".join(
            f"{item['document_type']}: {item['count']}" for item in overview["document_types"]
        )
        self.root.after(0, lambda: self.db_preview.configure(text=summary, image=""))
        self._show_cards("db", [("Toplam", str(overview["total_documents"])), ("Tablo", str(len(overview["specialized_counts"]))), ("DB", Path(overview["db_path"]).name)])
        self._show_table("db", [[row["document_type"], str(row["count"])] for row in overview["document_types"]])
        self._log("Veritabanı özeti yenilendi.")
        self._set_text(self.db_output, json.dumps(overview, ensure_ascii=False, indent=2))

    def _query_database_tab(self) -> None:
        results = query_database(
            self.db_path_var.get(),
            search_text=self.db_search_var.get().strip(),
            limit=100,
        )
        self._show_cards("db", [("Sonuç", str(len(results))), ("Arama", self.db_search_var.get().strip() or "-"), ("DB", Path(self.db_path_var.get()).name)])
        self._show_table("db", [[item.get("file_name", ""), item.get("document_type", ""), item.get("pipeline_name", "")] for item in results[:8]])
        self._log("Veritabanı sorgusu çalıştırıldı.")
        self._set_text(self.db_output, json.dumps(results, ensure_ascii=False, indent=2))

    def _start_camera(self) -> None:
        if self.camera_running:
            return
        capture = cv2.VideoCapture(self.camera_index_var.get())
        if not capture.isOpened():
            raise RuntimeError("Kamera açılamadı.")
        self.camera_capture = capture
        self.camera_running = True
        self.camera_status_var.set("Kamera açık")
        self._log(f"Kamera başlatıldı. Index: {self.camera_index_var.get()}")
        self._schedule_camera_frame()

    def _schedule_camera_frame(self) -> None:
        if not self.camera_running or self.camera_capture is None:
            return
        success, frame = self.camera_capture.read()
        if success:
            self.current_camera_frame = frame.copy()
            self._update_preview("camera", self.camera_preview, frame, max_size=(620, 420))
        self.root.after(50, self._schedule_camera_frame)

    def _stop_camera(self) -> None:
        self.camera_running = False
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        self.camera_status_var.set("Kamera kapalı")
        self._log("Kamera durduruldu.")

    def _analyze_camera_frame(self) -> None:
        if self.current_camera_frame is None:
            raise RuntimeError("Önce kamerayı başlatın ve bir kare alın.")

        frame = self.current_camera_frame.copy()
        output_dir = self._output_dir()
        edges = detect_document_edges(frame)
        source = edges["warped"] if edges["found"] else frame
        analysis = run_specialized_pipeline(source, lang=self.lang_var.get())
        overlay = annotate_ocr_boxes(source, lang=self.lang_var.get())
        output_image = output_dir / "desktop_camera_snapshot.png"
        output_json = output_dir / "desktop_camera_snapshot.json"
        cv2.imwrite(str(output_image), overlay["annotated"])
        payload = {
            "document_type": analysis["classification"]["type"],
            "pipeline": analysis["pipeline_name"],
            "edges_found": edges["found"],
            "fields": analysis["fields"],
            "insights": analysis["insights"],
            "text": analysis["corrected_text"],
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._update_preview("camera_result", self.camera_preview, overlay["annotated"], max_size=(620, 420))
        self._remember_result("camera", payload, [output_json, output_image])
        field_cards = [(key, str(value)) for key, value in (analysis.get("fields") or {}).items() if value][:4]
        self._show_cards("camera", [("Belge", analysis["classification"]["type"]), ("Kenar", "Var" if edges["found"] else "Yok"), *field_cards])
        self._show_table("camera", analysis["specialized"].get("table_rows"))
        self._log("Kamera karesi analiz edildi.")
        self._set_text(self.camera_output, json.dumps(payload, ensure_ascii=False, indent=2))

    def _on_close(self) -> None:
        self._stop_camera()
        self.root.destroy()


def main() -> None:
    root = TkinterDnD.Tk() if DND_AVAILABLE else Tk()
    OCRDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
