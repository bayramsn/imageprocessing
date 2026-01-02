import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import sys

class ProjectLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Proje Başlatıcı")
        self.root.geometry("600x700")
        
        # Stil ayarları
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12), padding=10)
        style.configure('TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'), foreground='#333')

        # Başlık
        header = ttk.Label(root, text="🚀 CV Proje Merkezi", style='Header.TLabel')
        header.pack(pady=20)

        # Proje Listesi
        # Format: (Başlık, Açıklama, Dosya Yolu, Görüntü Gerektirir mi?)
        self.projects = [
            ("Webcam Paint", "Sanal çizim tahtası (Kamera)", 
             "03_opencv_giris/webcam_paint.py", False),
             
            ("Blur Karşılaştırma", "Farklı bulanıklaştırma yöntemleri", 
             "04_gaussian_blur_opencv/blur_comparison.py", True),
             
            ("Tilt-Shift Efekti", "Minyatür şehir efekti", 
             "04_gaussian_blur_opencv/tilt_shift_effect.py", True),
             
            ("Kernel Bahçesi", "Keskinleştirme, Kabartma filtreleri", 
             "05_gaussian_blur_manual/kernel_playground.py", True),
             
            ("Şekil Tespiti", "Kare, Üçgen, Daire bulma", 
             "06_traditional_image_processing/shape_detector.py", True),
             
            ("Panorama Yapıcı", "Resim birleştirme (Çoklu seçim)", 
             "07_keypoints_features/panorama_maker.py", "multi"),
             
            ("Data Augmentation", "Veri çoğaltma yöntemleri", 
             "08_cnn_intro/data_augmentation_demo.py", True),
             
            ("3D Renk Analizi", "RGB uzayında renk dağılımı", 
             "09_numpy_matplotlib/color_distribution_3d.py", True),
             
            ("Yüz Tespiti", "Haar Cascade ile yüz bulma", 
             "10_detection_segmentation/face_eye_detector.py", True),
        ]

        # Scrollable Frame oluştur
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1, padx=20, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Butonları ekle
        for title, desc, path, req_type in self.projects:
            self.create_project_row(scrollable_frame, title, desc, path, req_type)

        # Çıkış Butonu
        exit_btn = ttk.Button(root, text="Çıkış", command=root.quit)
        exit_btn.pack(pady=20)

    def create_project_row(self, parent, title, desc, path, req_type):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=10)

        # Bilgi Kısmı
        info_frame = ttk.Frame(frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        lbl_title = ttk.Label(info_frame, text=title, font=('Helvetica', 12, 'bold'))
        lbl_title.pack(anchor='w')
        
        lbl_desc = ttk.Label(info_frame, text=desc, font=('Helvetica', 10), foreground='#666')
        lbl_desc.pack(anchor='w')

        # Çalıştır Butonu
        btn = ttk.Button(frame, text="Çalıştır ▶", 
                         command=lambda: self.run_project(path, req_type))
        btn.pack(side=tk.RIGHT, padx=10)
        
    def run_project(self, script_path, req_type):
        if not os.path.exists(script_path):
            messagebox.showerror("Hata", f"Dosya bulunamadı:\n{script_path}")
            return

        cmd = [sys.executable, script_path]
        
        if req_type == True: # Tek resim
            file_path = filedialog.askopenfilename(
                title="Bir resim seçin",
                filetypes=[("Resimler", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not file_path: return
            cmd.append(file_path)
            
        elif req_type == "multi": # Çoklu resim (Panorama)
            file_paths = filedialog.askopenfilenames(
                title="Birleştirilecek resimleri seçin",
                filetypes=[("Resimler", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not file_paths or len(file_paths) < 2:
                messagebox.showwarning("Uyarı", "Panorama için en az 2 resim seçmelisiniz.")
                return
            cmd.extend(file_paths)

        # Komutu çalıştır
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Hata", f"Çalıştırma hatası:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectLauncherApp(root)
    root.mainloop()
