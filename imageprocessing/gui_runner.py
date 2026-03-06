#!/usr/bin/env python3
"""
CV Human Behavior Analytics — Test & Run GUI
=============================================
Tkinter GUI to run project modules & pytest tests one by one,
view real-time output, and see pass/fail status at a glance.

Usage:
    python gui_runner.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # use the same interpreter that launched the GUI
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"

# ── Color palette ───────────────────────────────────────────────────
BG_MAIN      = "#1e1e2e"
BG_SIDEBAR   = "#181825"
BG_CONSOLE   = "#11111b"
FG_TEXT       = "#cdd6f4"
FG_DIM        = "#6c7086"
FG_PASS       = "#a6e3a1"
FG_FAIL       = "#f38ba8"
FG_RUNNING    = "#f9e2af"
FG_PENDING    = "#585b70"
ACCENT        = "#89b4fa"
ACCENT_HOVER  = "#74c7ec"
BTN_BG        = "#313244"
BTN_ACTIVE    = "#45475a"
BORDER_COLOR  = "#45475a"


# ═══════════════════════════════════════════════════════════════════
#  Task definitions
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Task:
    name: str
    category: str          # "utils", "pipeline", "test", "main"
    command: List[str]
    status: str = "pending"  # pending | running | passed | failed
    output: str = ""
    duration: float = 0.0


def build_tasks() -> List[Task]:
    """Build the ordered list of tasks to run."""
    py = str(PYTHON)
    root = str(PROJECT_ROOT)

    tasks: List[Task] = []

    # ── Utils ───────────────────────────────────────────────────
    utils_modules = [
        ("utils/geometry.py",   "src.utils.geometry"),
        ("utils/fps.py",        "src.utils.fps"),
        ("utils/draw.py",       "src.utils.draw"),
        ("utils/time_utils.py", "src.utils.time_utils"),
    ]
    for display, mod in utils_modules:
        tasks.append(Task(
            name=display,
            category="utils",
            command=[py, "-c", f"import {mod}; print('[OK] {mod} imported successfully')"],
        ))

    # ── Pipeline ────────────────────────────────────────────────
    pipeline_modules = [
        ("pipeline/behavior.py",      "src.pipeline.behavior"),
        ("pipeline/detector_yolo.py",  "src.pipeline.detector_yolo"),
        ("pipeline/logger.py",         "src.pipeline.logger"),
        ("pipeline/overlay.py",        "src.pipeline.overlay"),
        ("pipeline/pose.py",           "src.pipeline.pose"),
        ("pipeline/segmentation.py",   "src.pipeline.segmentation"),
        ("pipeline/timer.py",          "src.pipeline.timer"),
        ("pipeline/tracker.py",        "src.pipeline.tracker"),
        ("pipeline/video_source.py",   "src.pipeline.video_source"),
    ]
    for display, mod in pipeline_modules:
        tasks.append(Task(
            name=display,
            category="pipeline",
            command=[py, "-c", f"import {mod}; print('[OK] {mod} imported successfully')"],
        ))

    # ── Main ────────────────────────────────────────────────────
    tasks.append(Task(
        name="main.py (import)",
        category="main",
        command=[py, "-c", "from src.main import main; print('[OK] src.main imported successfully')"],
    ))

    # ── Pytest tests ────────────────────────────────────────────
    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    for tf in test_files:
        rel = tf.relative_to(PROJECT_ROOT)
        tasks.append(Task(
            name=str(rel),
            category="test",
            command=[py, "-m", "pytest", str(tf), "-v", "--tb=short", "--no-header"],
        ))

    # ── Manual integration test ─────────────────────────────────
    manual_test = TESTS_DIR / "run_all_manual.py"
    if manual_test.exists():
        tasks.append(Task(
            name="tests/run_all_manual.py",
            category="test",
            command=[py, str(manual_test)],
        ))

    return tasks


# ═══════════════════════════════════════════════════════════════════
#  GUI Application
# ═══════════════════════════════════════════════════════════════════

class TestRunnerApp:
    """Main Tkinter application for the test/module runner."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CV Human Behavior Analytics — Runner")
        self.root.geometry("1100x720")
        self.root.minsize(900, 550)
        self.root.configure(bg=BG_MAIN)

        self.tasks = build_tasks()
        self._running = False
        self._stop_requested = False
        self._current_proc: Optional[subprocess.Popen] = None

        self._build_ui()
        self._populate_task_list()

    # ── UI construction ─────────────────────────────────────────

    def _build_ui(self) -> None:
        # Top bar
        top = tk.Frame(self.root, bg=BG_SIDEBAR, height=56)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        tk.Label(
            top, text="  CV Human Behavior Analytics", font=("SF Pro Display", 16, "bold"),
            fg=ACCENT, bg=BG_SIDEBAR, anchor="w",
        ).pack(side="left", padx=12, pady=10)

        tk.Label(
            top, text="Test & Module Runner", font=("SF Pro Display", 11),
            fg=FG_DIM, bg=BG_SIDEBAR, anchor="w",
        ).pack(side="left", padx=4, pady=10)

        # Status summary label (right side of top bar)
        self.lbl_summary = tk.Label(
            top, text="", font=("SF Mono", 11), fg=FG_TEXT, bg=BG_SIDEBAR,
        )
        self.lbl_summary.pack(side="right", padx=16, pady=10)

        # Main content: sidebar + console
        body = tk.PanedWindow(
            self.root, orient="horizontal", bg=BORDER_COLOR,
            sashwidth=3, sashrelief="flat",
        )
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Sidebar (task list) ─────────────────────────────────
        sidebar = tk.Frame(body, bg=BG_SIDEBAR, width=340)
        body.add(sidebar, minsize=280, stretch="never")

        # Button bar
        btn_bar = tk.Frame(sidebar, bg=BG_SIDEBAR)
        btn_bar.pack(fill="x", padx=8, pady=(10, 4))

        self.btn_run_all = self._make_button(btn_bar, "▶  Tümünü Çalıştır", self._on_run_all)
        self.btn_run_all.pack(side="left", padx=(0, 4))

        self.btn_run_sel = self._make_button(btn_bar, "▶  Seçili", self._on_run_selected)
        self.btn_run_sel.pack(side="left", padx=(0, 4))

        self.btn_stop = self._make_button(btn_bar, "■  Durdur", self._on_stop, fg=FG_FAIL)
        self.btn_stop.pack(side="left", padx=(0, 4))
        self.btn_stop.configure(state="disabled")

        self.btn_reset = self._make_button(btn_bar, "↻  Sıfırla", self._on_reset)
        self.btn_reset.pack(side="right")

        # Task treeview
        tree_frame = tk.Frame(sidebar, bg=BG_SIDEBAR)
        tree_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Task.Treeview",
                        background=BG_SIDEBAR, foreground=FG_TEXT,
                        fieldbackground=BG_SIDEBAR, borderwidth=0,
                        font=("SF Mono", 11), rowheight=28)
        style.configure("Task.Treeview.Heading",
                        background=BTN_BG, foreground=FG_DIM,
                        font=("SF Mono", 10, "bold"), borderwidth=0)
        style.map("Task.Treeview",
                  background=[("selected", BTN_ACTIVE)],
                  foreground=[("selected", ACCENT)])

        self.tree = ttk.Treeview(
            tree_frame, style="Task.Treeview",
            columns=("status", "time"), show="tree headings",
            selectmode="extended",
        )
        self.tree.heading("#0", text="Modül / Test", anchor="w")
        self.tree.heading("status", text="Durum", anchor="center")
        self.tree.heading("time", text="Süre", anchor="center")
        self.tree.column("#0", width=200, stretch=True)
        self.tree.column("status", width=65, stretch=False, anchor="center")
        self.tree.column("time", width=55, stretch=False, anchor="center")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # ── Console (right side) ────────────────────────────────
        console_frame = tk.Frame(body, bg=BG_CONSOLE)
        body.add(console_frame, minsize=400, stretch="always")

        console_top = tk.Frame(console_frame, bg=BG_CONSOLE)
        console_top.pack(fill="x", padx=10, pady=(8, 0))

        self.lbl_console_title = tk.Label(
            console_top, text="Konsol Çıktısı",
            font=("SF Pro Display", 12, "bold"), fg=FG_DIM, bg=BG_CONSOLE,
        )
        self.lbl_console_title.pack(side="left")

        self.btn_clear = self._make_button(console_top, "Temizle", self._on_clear_console)
        self.btn_clear.pack(side="right")

        self.console = scrolledtext.ScrolledText(
            console_frame, bg=BG_CONSOLE, fg=FG_TEXT,
            font=("SF Mono", 11), insertbackground=FG_TEXT,
            relief="flat", borderwidth=0, padx=10, pady=8,
            wrap="word", state="disabled",
        )
        self.console.pack(fill="both", expand=True, padx=4, pady=(4, 4))

        # Text tags for colors
        self.console.tag_configure("pass",    foreground=FG_PASS)
        self.console.tag_configure("fail",    foreground=FG_FAIL)
        self.console.tag_configure("running", foreground=FG_RUNNING)
        self.console.tag_configure("header",  foreground=ACCENT, font=("SF Mono", 11, "bold"))
        self.console.tag_configure("dim",     foreground=FG_DIM)

        # Progress bar at the very bottom
        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=len(self.tasks))
        style.configure("TProgressbar", troughcolor=BG_SIDEBAR, background=ACCENT,
                        bordercolor=BG_MAIN, lightcolor=ACCENT, darkcolor=ACCENT)
        self.progress.pack(fill="x", side="bottom", padx=0, pady=0)

    def _make_button(self, parent, text, command, fg=FG_TEXT):
        btn = tk.Button(
            parent, text=text, command=command,
            font=("SF Pro Display", 10), fg=fg, bg=BTN_BG,
            activebackground=BTN_ACTIVE, activeforeground=fg,
            relief="flat", padx=10, pady=4, cursor="hand2",
            highlightthickness=0, borderwidth=0,
        )
        btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=BTN_ACTIVE))
        btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=BTN_BG))
        return btn

    # ── Task list ───────────────────────────────────────────────

    def _populate_task_list(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._iid_map: dict[str, int] = {}  # iid → task index

        categories = {
            "utils":    "📦  Utils",
            "pipeline": "⚙️  Pipeline",
            "main":     "🚀  Main",
            "test":     "🧪  Tests",
        }
        cat_nodes: dict[str, str] = {}

        for cat_key, cat_label in categories.items():
            iid = self.tree.insert("", "end", text=cat_label, open=True, values=("", ""))
            cat_nodes[cat_key] = iid

        for idx, task in enumerate(self.tasks):
            parent = cat_nodes.get(task.category, "")
            status_icon = self._status_icon(task.status)
            iid = self.tree.insert(
                parent, "end",
                text=f"  {task.name}",
                values=(status_icon, ""),
            )
            self._iid_map[iid] = idx

    def _status_icon(self, status: str) -> str:
        return {
            "pending":  "⏳",
            "running":  "🔄",
            "passed":   "✅",
            "failed":   "❌",
        }.get(status, "")

    def _update_task_row(self, idx: int) -> None:
        task = self.tasks[idx]
        for iid, task_idx in self._iid_map.items():
            if task_idx == idx:
                duration_str = f"{task.duration:.1f}s" if task.duration > 0 else ""
                self.tree.item(iid, values=(self._status_icon(task.status), duration_str))
                break
        self._update_summary()

    def _update_summary(self) -> None:
        passed = sum(1 for t in self.tasks if t.status == "passed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        total = len(self.tasks)
        self.lbl_summary.config(text=f"✅ {passed}   ❌ {failed}   📋 {total}")

    # ── Console helpers ─────────────────────────────────────────

    def _console_write(self, text: str, tag: str = "") -> None:
        self.console.configure(state="normal")
        if tag:
            self.console.insert("end", text, tag)
        else:
            self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _console_clear(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    # ── Button callbacks ────────────────────────────────────────

    def _on_run_all(self) -> None:
        if self._running:
            return
        indices = list(range(len(self.tasks)))
        self._run_tasks(indices)

    def _on_run_selected(self) -> None:
        if self._running:
            return
        selected = self.tree.selection()
        indices = []
        for iid in selected:
            if iid in self._iid_map:
                indices.append(self._iid_map[iid])
            else:
                # category node → select all children
                for child in self.tree.get_children(iid):
                    if child in self._iid_map:
                        indices.append(self._iid_map[child])
        if not indices:
            messagebox.showinfo("Bilgi", "Lütfen çalıştırılacak modül veya testi seçin.")
            return
        self._run_tasks(sorted(set(indices)))

    def _on_stop(self) -> None:
        self._stop_requested = True
        if self._current_proc and self._current_proc.poll() is None:
            self._current_proc.terminate()

    def _on_reset(self) -> None:
        if self._running:
            return
        for task in self.tasks:
            task.status = "pending"
            task.output = ""
            task.duration = 0.0
        self._populate_task_list()
        self._console_clear()
        self._update_summary()
        self.progress["value"] = 0

    def _on_clear_console(self) -> None:
        self._console_clear()

    def _on_tree_select(self, event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        iid = selected[0]
        if iid in self._iid_map:
            idx = self._iid_map[iid]
            task = self.tasks[idx]
            if task.output:
                self._console_clear()
                self._console_write(f"── {task.name} ──\n", "header")
                tag = "pass" if task.status == "passed" else "fail" if task.status == "failed" else ""
                self._console_write(task.output, tag)

    # ── Task execution ──────────────────────────────────────────

    def _run_tasks(self, indices: List[int]) -> None:
        self._running = True
        self._stop_requested = False
        self.btn_run_all.configure(state="disabled")
        self.btn_run_sel.configure(state="disabled")
        self.btn_reset.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress["value"] = 0
        self.progress["maximum"] = len(indices)

        thread = threading.Thread(target=self._worker, args=(indices,), daemon=True)
        thread.start()

    def _worker(self, indices: List[int]) -> None:
        import time as _time

        for step, idx in enumerate(indices):
            if self._stop_requested:
                self.root.after(0, lambda: self._console_write("\n⛔ Durduruldu.\n", "fail"))
                break

            task = self.tasks[idx]
            task.status = "running"
            task.output = ""
            self.root.after(0, self._update_task_row, idx)
            self.root.after(0, lambda t=task: (
                self._console_write(f"\n{'─' * 50}\n", "dim"),
                self._console_write(f"▶  {t.name}\n", "header"),
            ))

            t0 = _time.perf_counter()
            try:
                self._current_proc = subprocess.Popen(
                    task.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                output_lines = []
                for line in self._current_proc.stdout:
                    output_lines.append(line)
                    self.root.after(0, self._console_write, line)

                self._current_proc.wait(timeout=120)
                task.output = "".join(output_lines)
                task.duration = _time.perf_counter() - t0

                if self._current_proc.returncode == 0:
                    task.status = "passed"
                    self.root.after(0, lambda t=task: self._console_write(
                        f"  ✅ PASSED ({t.duration:.1f}s)\n", "pass"
                    ))
                else:
                    task.status = "failed"
                    self.root.after(0, lambda t=task: self._console_write(
                        f"  ❌ FAILED (exit={self._current_proc.returncode}, {t.duration:.1f}s)\n", "fail"
                    ))

            except subprocess.TimeoutExpired:
                self._current_proc.kill()
                task.status = "failed"
                task.output += "\n[TIMEOUT after 120s]"
                task.duration = _time.perf_counter() - t0
                self.root.after(0, lambda: self._console_write(
                    "  ❌ TIMEOUT (120s)\n", "fail"
                ))
            except Exception as e:
                task.status = "failed"
                task.output += f"\n[ERROR] {e}"
                task.duration = _time.perf_counter() - t0
                self.root.after(0, lambda e=e: self._console_write(
                    f"  ❌ ERROR: {e}\n", "fail"
                ))

            self.root.after(0, self._update_task_row, idx)
            self.root.after(0, lambda s=step + 1: self.progress.configure(value=s))

        # Done
        self._running = False
        self._current_proc = None
        self.root.after(0, self._finish_run)

    def _finish_run(self) -> None:
        self.btn_run_all.configure(state="normal")
        self.btn_run_sel.configure(state="normal")
        self.btn_reset.configure(state="normal")
        self.btn_stop.configure(state="disabled")

        passed = sum(1 for t in self.tasks if t.status == "passed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        total = passed + failed

        self._console_write(f"\n{'═' * 50}\n", "dim")
        if failed == 0:
            self._console_write(f"  🎉  TÜMÜ BAŞARILI  —  {passed}/{total} passed\n", "pass")
        else:
            self._console_write(f"  ⚠️   {passed} passed  /  {failed} failed  (toplam {total})\n", "fail")
        self._console_write(f"{'═' * 50}\n", "dim")
        self._update_summary()


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    # macOS dark title bar hint
    try:
        root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "moveableModal", "")
    except tk.TclError:
        pass

    app = TestRunnerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
