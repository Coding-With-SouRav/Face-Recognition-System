import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ctypes
import sys
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import traceback
from datetime import datetime
import cv2
from keras.models import load_model
import platform
import getpass
import urllib.request
import socket
from urllib.error import URLError, HTTPError

def get_appdata_dir():

    if platform.system() == "Windows":
        base_dir = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'FaceRecognitionPro')
    elif platform.system() == "Darwin":
        base_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'FaceRecognitionPro')
    else:
        base_dir = os.path.join(os.path.expanduser('~'), '.local', 'share', 'FaceRecognitionPro')
    return base_dir
APP_DATA_DIR = get_appdata_dir()

def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS

    except Exception:
        base_path = os.path.abspath(".")
    full_path = os.path.join(base_path, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Resource not found: {full_path}")
    return full_path
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['MALE', 'FEMALE']
Emotions = ["ANGRY", "DISGUST", "SCARED", "HAPPY", "SAD", "SURPRISED", "NORMAL"]
COLORS = {
    "primary": "#2C3E50",      # Dark blue
    "secondary": "#34495E",    # Slightly lighter blue
    "accent": "#1ABC9C",       # Teal accent
    "accent_dark": "#16A085",  # Darker teal
    "danger": "#E74C3C",       # Red
    "warning": "#F39C12",      # Orange
    "light": "#ECF0F1",        # Light gray
    "dark": "#2C3E50",         # Dark
    "success": "#27AE60",      # Green
    "bg": "#F8F9FA",           # Off-white background
    "card": "#FFFFFF",         # White cards
    "border": "#DEE2E6",       # Light border
}

def get_user_data_dir():
    return APP_DATA_DIR

def get_user_dataset_dir(name):
    return os.path.join(get_user_data_dir(), "user_datasets", name)

def get_classifiers_dir():
    return os.path.join(get_user_data_dir(), "classifiers")

def get_models_dir():
    return os.path.join(get_user_data_dir(), "models")

def create_data_directories():
    directories = [
        get_user_data_dir(),
        os.path.join(get_user_data_dir(), "user_datasets"),
        get_classifiers_dir(),
        get_models_dir()
    ]
    for directory in directories:

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
username = getpass.getuser()
BASE_DIR = rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models"
MODEL_FILES = {
    "emotion/_mini_XCEPTION.106-0.65.hdf5":
        "https://raw.githubusercontent.com/Coding-With-SouRav/MODELS/main/_mini_XCEPTION.106-0.65.hdf5",
    "age/age_deploy.prototxt":
        "https://raw.githubusercontent.com/Coding-With-SouRav/MODELS/main/age_deploy.prototxt",
    "age/age_net.caffemodel":
        "https://raw.githubusercontent.com/Coding-With-SouRav/MODELS/main/age_net.caffemodel",
    "gender/gender_deploy.prototxt":
        "https://raw.githubusercontent.com/Coding-With-SouRav/MODELS/main/gender_deploy.prototxt",
    "gender/gender_net.caffemodel":
        "https://raw.githubusercontent.com/Coding-With-SouRav/MODELS/main/gender_net.caffemodel",
    "haarcascade/haarcascade_frontalface_default.xml":
        "https://raw.githubusercontent.com/Coding-With-SouRav/MODELS/main/haarcascade_frontalface_default.xml",
}
face_cascade_path = rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models\haarcascade\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haarcascade classifier")

def train_classifier(name):
    path = get_user_dataset_dir(name)

    if not os.path.exists(path):
        messagebox.showerror("Error", f"Dataset folder not found: {path}")
        return False
    faces = []
    ids = []
    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not files:
        messagebox.showerror("Error", "No images found in dataset.")
        return False
    print(f"Training model for {name}...")
    print(f"Found {len(files)} images")
    for idx, pic in enumerate(files, 1):
        imgpath = os.path.join(path, pic)

        try:
            img = Image.open(imgpath).convert('L')
            imageNp = np.array(img, 'uint8')
            base_name = os.path.splitext(pic)[0]
            parts = base_name.split('_')

            if parts and parts[0].isdigit():
                id_ = int(parts[0])
            else:
                id_ = idx
            faces.append(imageNp)
            ids.append(id_)

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(files)} images")

        except Exception as e:
            print(f"Error processing {pic}: {e}")
            continue

    if len(faces) == 0:
        messagebox.showerror("Error", "No valid training images.")
        return False
    ids = np.array(ids)

    try:
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)

        classifier_file = os.path.join(get_classifiers_dir(), f"{name}_classifier.xml")
        clf.write(classifier_file)
        print(f"Training complete! Model saved to {classifier_file}")

    except Exception as e:
        messagebox.showerror("Training Error", str(e))
        traceback.print_exc()
        return False
    return True

def train_classifier_with_progress(name, progress_callback=None):
    path = get_user_dataset_dir(name)

    if not os.path.exists(path):
        return False, "Dataset folder not found"
    faces = []
    ids = []
    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not files:
        return False, "No images found in dataset"
    total_files = len(files)
    for idx, pic in enumerate(files, 1):
        imgpath = os.path.join(path, pic)

        try:
            img = Image.open(imgpath).convert('L')
            imageNp = np.array(img, 'uint8')
            base_name = os.path.splitext(pic)[0]
            parts = base_name.split('_')

            if parts and parts[0].isdigit():
                id_ = int(parts[0])
            else:
                id_ = idx
            faces.append(imageNp)
            ids.append(id_)

            if progress_callback:
                progress = int((idx / total_files) * 100)
                progress_callback(progress, f"Processing image {idx}/{total_files}")

        except Exception as e:
            print(f"Error processing {pic}: {e}")
            continue

    if len(faces) == 0:
        return False, "No valid training images"
    ids = np.array(ids)

    try:

        if progress_callback:
            progress_callback(95, "Training model...")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)

        classifier_file = os.path.join(get_classifiers_dir(), f"{name}_classifier.xml")
        clf.write(classifier_file)

        if progress_callback:
            progress_callback(100, "Training complete!")
        return True, "Training complete!"

    except Exception as e:
        return False, f"Training error: {str(e)}"

def get_all_trained_users():
    users = []

    classifiers_dir = get_classifiers_dir()

    if not os.path.exists(classifiers_dir):
        return users
    for classifier_file in os.listdir(classifiers_dir):

        if classifier_file.endswith("_classifier.xml"):
            user_name = classifier_file.replace("_classifier.xml", "")

            classifier_path = os.path.join(classifiers_dir, classifier_file)

            try:
                clf = cv2.face.LBPHFaceRecognizer_create()
                clf.read(classifier_path)
                dataset_path = get_user_dataset_dir(user_name)
                image_count = 0

                if os.path.exists(dataset_path):
                    image_count = len([f for f in os.listdir(dataset_path)

                                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                users.append({
                    'name': user_name,
                    'classifier_path': classifier_path,
                    'classifier': clf,
                    'image_count': image_count
                })

            except Exception as e:
                print(f"Error loading classifier for {user_name}: {e}")
    return users

def get_user_info():
    users = []
    user_datasets_dir = os.path.join(get_user_data_dir(), "user_datasets")

    if not os.path.exists(user_datasets_dir):
        return users
    for item in os.listdir(user_datasets_dir):
        item_path = os.path.join(user_datasets_dir, item)

        if os.path.isdir(item_path):

            try:
                creation_time = os.path.getctime(item_path)
                creation_date = datetime.fromtimestamp(creation_time)
                image_count = 0

                if os.path.exists(item_path):
                    for file in os.listdir(item_path):

                        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            image_count += 1
                model_exists = os.path.exists(os.path.join(get_classifiers_dir(), f"{item}_classifier.xml"))
                users.append({
                    'name': item,
                    'created': creation_date,
                    'image_count': image_count,
                    'model_exists': model_exists,
                    'model_path': os.path.join(get_classifiers_dir(), f"{item}_classifier.xml") if model_exists else None
                })

            except Exception as e:
                print(f"Error processing user {item}: {e}")
    users.sort(key=lambda x: x['created'], reverse=True)
    return users

def all_models_exist():
    for relative_path in MODEL_FILES:
        full_path = os.path.join(BASE_DIR, relative_path)

        if not os.path.exists(full_path):
            return False
    return True

class ModelDownloader:

    def __init__(self, parent, on_complete):
        self.parent = parent
        self.on_complete = on_complete
        self.stop_download = False
        self.current_file = ""
        self.current_temp_path = None
        self.window = tk.Toplevel(parent)
        self.window.title("Downloading AI Models")
        self.window.geometry("600x260")
        self.window.resizable(False, False)
        self.center_window(600, 260)

        if sys.platform == "win32":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "com.sourav.FaceRecognitionPro.ModelDownloader"
            )
        icon_img = Image.open(resource_path("Images/icon.png"))
        self.app_icon = ImageTk.PhotoImage(icon_img)
        self.window.iconphoto(True, self.app_icon)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        tk.Label(self.window, text= "Models are found in: ", font=("Segoe UI", 12, "bold")).pack(pady=10)
        tk.Label(self.window, text= rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro", font=("Segoe UI", 12, "bold")).pack(pady=10)
        self.label = tk.Label(self.window, text="Preparing downloads...", font=("Segoe UI", 10))
        self.label.pack(pady=10)
        self.progress = ttk.Progressbar(self.window, length=350, mode="determinate")
        self.progress.pack(pady=10)
        self.percent_label = tk.Label(self.window, text="0%")
        self.percent_label.pack()
        os.makedirs(BASE_DIR, exist_ok=True)
        threading.Thread(target=self.start_download, daemon=True).start()

    def center_window(self, width, height):
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def internet_available(self):

        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True

        except:
            return False

    def on_close(self):
        self.stop_download = True

        if self.current_temp_path and os.path.exists(self.current_temp_path):

            try:
                os.remove(self.current_temp_path)

            except:
                pass
        self.window.destroy()
        self.parent.destroy()

    def download_file(self, url, final_path):
        temp_path = final_path + ".part"
        self.current_temp_path = temp_path
        downloaded = 0

        if os.path.exists(temp_path):
            downloaded = os.path.getsize(temp_path)

        while not self.stop_download:

            if not self.internet_available():
                self.window.after(0, lambda: self.label.config(
                    text="📡 Waiting for internet connection..."
                ))
                time.sleep(2)
                continue
            else:
                self.window.after(0, lambda: self.label.config(
                    text=f"Downloading:\n{self.current_file}"
                ))

            try:
                req = urllib.request.Request(url)

                if downloaded > 0:
                    req.add_header("Range", f"bytes={downloaded}-")
                response = urllib.request.urlopen(req, timeout=10)
                total_size = int(response.getheader("Content-Length", 0)) + downloaded

                with open(temp_path, "ab") as f:

                    while not self.stop_download:
                        chunk = response.read(8192)

                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = int(downloaded / total_size * 100)
                        self.progress["value"] = percent
                        self.percent_label.config(text=f"{percent}%")
                        self.window.update_idletasks()

                if downloaded >= total_size:
                    break

            except (URLError, HTTPError, socket.gaierror, TimeoutError):
                time.sleep(2)
                continue

        if self.stop_download:
            return False
        os.replace(temp_path, final_path)
        self.current_temp_path = None
        return True

    def start_download(self):
        for relative_path, url in MODEL_FILES.items():

            if self.stop_download:
                return
            final_path = os.path.join(BASE_DIR, relative_path)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            temp_path = final_path + ".part"

            if os.path.exists(temp_path):
                os.remove(temp_path)

            if os.path.exists(final_path):
                continue
            self.current_file = relative_path
            self.label.config(text=f"Downloading:\n{relative_path}")
            self.progress["value"] = 0
            self.percent_label.config(text="0%")

            if not self.download_file(url, final_path):
                return
        self.window.after(300, self.finish)

    def finish(self):
        self.window.destroy()
        self.on_complete()

class MainUI(tk.Tk):

    def __init__(self):
        super().__init__()
        create_data_directories()
        self.title("Face Recognition Pro - By SouRav Bhattacharya")
        self.geometry("1200x800")
        self.resizable(False, False)

        if sys.platform == "win32":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.example.FaceRecogApp")

        try:
            self.iconphoto(True, tk.PhotoImage(file=resource_path("Images/icon.png")))

        except:
            pass
        self.configure(bg=COLORS["bg"])
        self.title_font = ("Segoe UI", 24, "bold")
        self.subtitle_font = ("Segoe UI", 11)
        self.active_name = None
        self.camera_frame_ref = None
        self.back_icon = None

        try:
            img = Image.open(resource_path(r"Images/back.png")).resize((42, 42))
            self.back_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
        main_container = tk.Frame(self, bg=COLORS["primary"])
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        header_frame = tk.Frame(main_container, bg=COLORS["primary"])
        header_frame.pack(fill="x", padx=0, pady=(0, 20))
        title_frame = tk.Frame(header_frame, bg=COLORS["primary"])
        title_frame.pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(title_frame, fg="white", bg=COLORS["primary"], text="👤", font=("Segoe UI", 28)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(title_frame, text="FACE RECOGNIZER",
                 font=self.title_font,
                 fg="white", bg=COLORS["primary"]).pack(side=tk.LEFT)
        content_frame = tk.Frame(main_container, bg=COLORS["card"], relief="solid", bd=1)
        content_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self.frames = {}
        for F in (StartPage, SignupPage, SelectUserPage, CapturePage,
                RecognizePage, ShowUsersPage, FeatureRecognitionPage):
            page = F(parent=content_frame, controller=self)
            self.frames[F.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        self.show_frame("StartPage")
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def show_frame(self, name):

        if self.camera_frame_ref:

            try:
                self.camera_frame_ref.stop()

            except:
                pass
            self.camera_frame_ref = None
        frame = self.frames[name]
        frame.tkraise()

        if hasattr(frame, "on_show"):
            frame.on_show()

        if name == "ShowUsersPage":
            frame.refresh_users()

class FeatureRecognitionPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        self.emotion_model = None
        self.ageNet = None
        self.genderNet = None
        self.models_loaded = False
        self.camera_loading = False
        self.running = False
        self.cap = None
        self.recognizers = []
        self.main_container = tk.Frame(self, bg=COLORS["card"])
        self.main_container.pack(fill="both", expand=True)
        self.feature_selection_widgets = []
        self.create_setup_ui()

    def create_setup_ui(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()
        self.feature_selection_widgets = []
        header_frame = tk.Frame(self.main_container, bg=COLORS["card"])
        header_frame.pack(fill="x", padx=30, pady=(20, 10))
        self.feature_selection_widgets.append(header_frame)
        back_btn_text = "← Back"
        back_btn_image = None
        back_btn_font = ("Segoe UI", 10, "bold")
        back_btn_padx = 10
        back_btn_pady = 5

        if self.controller.back_icon:
            back_btn_text = ""
            back_btn_image = self.controller.back_icon
            back_btn_font = None
            back_btn_padx = 5
            back_btn_pady = 5
        back_btn = tk.Button(header_frame, text=back_btn_text,
                            image=back_btn_image,
                            command=lambda: self.controller.show_frame("StartPage"),
                            bg="white", fg="white",
                            font=back_btn_font,
                            bd=0,
                            activebackground="#3C98F4",
                            activeforeground="white",
                            relief="flat", padx=back_btn_padx, pady=back_btn_pady)

        if back_btn_image:
            back_btn.image = back_btn_image
        back_btn.pack(side=tk.LEFT)

        try:
            img = Image.open(resource_path(r"Images/recognition.png")).resize((42, 42))
            self.recognition_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.recognition_icon = None
        tk.Label(header_frame, text="  Advanced Recognition",
                 image=self.recognition_icon,
                 compound="left",
                font=("Segoe UI", 16, "bold"),
                fg=COLORS["primary"], bg=COLORS["card"]).pack(side=tk.LEFT, padx=20)
        self.status_label = tk.Label(header_frame,
                                    text="Ready",
                                    font=("Segoe UI", 10),
                                    fg=COLORS["success"], bg=COLORS["card"])
        self.status_label.pack(side=tk.RIGHT, padx=10)
        content_frame = tk.Frame(self.main_container, bg=COLORS["card"])
        content_frame.pack(fill="both", expand=True, padx=30, pady=20)
        self.feature_selection_widgets.append(content_frame)
        feature_frame = tk.Frame(content_frame, bg=COLORS["light"], relief="solid", bd=1)
        feature_frame.pack(fill="x", pady=(0, 20))
        self.feature_selection_widgets.append(feature_frame)
        tk.Label(feature_frame, text="  Select Features to Enable:  ",
                font=("Segoe UI", 15, "bold", "underline"),
                fg=COLORS["primary"], bg=COLORS["light"]).pack(pady=(15, 10))
        check_frame = tk.Frame(feature_frame, bg=COLORS["light"])
        check_frame.pack(pady=(0, 15))
        self.feature_selection_widgets.append(check_frame)
        style = ttk.Style()
        style.configure(
            "Custom.TCheckbutton",
            font=("Arial", 12, "bold"),
            foreground="black",
            background=COLORS["light"],
            padding=10
        )
        style.map(
            "Custom.TCheckbutton",
            foreground=[("selected", "#0CCB02")],
            background=[("active", "#D7F8D4")]
        )

        try:
            img = Image.open(resource_path(r"Images/gender.png")).resize((30, 30))
            self.gender_icon = ImageTk.PhotoImage(img)
            img = Image.open(resource_path(r"Images/emotion.png")).resize((30, 30))
            self.emotion_icon = ImageTk.PhotoImage(img)
            img = Image.open(resource_path(r"Images/recog_face.png")).resize((30, 30))
            self.recog_face_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.gender_icon = None
            self.emotion_icon = None
            self.recog_face_icon = None
        self.gender_var = tk.BooleanVar(value=True)
        self.gender_check = ttk.Checkbutton(check_frame, text="  Gender & Age Detection",
                                            image=self.gender_icon,
                                            compound="left",
                                          variable=self.gender_var,
                                          style="Custom.TCheckbutton"
                                          )
        self.gender_check.pack(pady=(10,0))
        self.feature_selection_widgets.append(self.gender_check)
        self.emotion_var = tk.BooleanVar(value=True)
        self.emotion_check = ttk.Checkbutton(check_frame,
                                            text="  Emotion Detection",
                                           image=self.emotion_icon,
                                           compound="left",
                                           variable=self.emotion_var,
                                           style="Custom.TCheckbutton")
        self.emotion_check.pack(pady=20, padx=(0,55))
        self.feature_selection_widgets.append(self.emotion_check)
        self.face_var = tk.BooleanVar(value=True)
        self.face_check = ttk.Checkbutton(check_frame, text="  Face Recognition (All Users)",
                                        image=self.recog_face_icon,
                                        compound="left",
                                        variable=self.face_var,
                                        style="Custom.TCheckbutton"
                                        )
        self.face_check.pack(pady=0, padx=(33,0))
        self.feature_selection_widgets.append(self.face_check)
        self.camera_container = tk.Frame(content_frame, bg="white")
        self.camera_container.pack(fill="both", expand=True, pady=(0, 20))
        self.feature_selection_widgets.append(self.camera_container)
        button_frame = tk.Frame(content_frame, bg=COLORS["card"])
        button_frame.pack(pady=10)
        self.feature_selection_widgets.append(button_frame)
        self.load_btn = tk.Button(button_frame, text=" Load Models",
                                 command=self.load_models,
                                 bg="#0606F3", fg="white",
                                 font=("Segoe UI", 10, "bold"),
                                 bd=0,
                                 activebackground="#9BA0FE",
                                 activeforeground="black",
                                 relief="flat", padx=15, pady=8)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        self.start_btn = tk.Button(button_frame, text="▶ Start Recognition",
                                  command=self.start_recognition,
                                  bg=COLORS["accent"], fg="white",
                                  font=("Segoe UI", 10, "bold"),
                                  bd=0,
                                  activebackground="#0B9560",
                                  activeforeground="black",
                                  relief="flat", padx=15, pady=8,
                                  state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.models_status = tk.Label(content_frame,
                                     text="⚠ Models not loaded. Click 'Load Models' first.",
                                     font=("Segoe UI", 9),
                                     fg=COLORS["warning"],
                                     bg=COLORS["card"])
        self.models_status.pack(pady=5)
        self.feature_selection_widgets.append(self.models_status)

    def create_camera_ui(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()
        header_frame = tk.Frame(self.main_container, bg=COLORS["card"])
        header_frame.pack(fill="x", padx=30, pady=(20, 10))
        back_btn_text = "← Back"
        back_btn_image = None
        back_btn_font = ("Segoe UI", 10, "bold")
        back_btn_padx = 10
        back_btn_pady = 5

        if self.controller.back_icon:
            back_btn_text = ""
            back_btn_image = self.controller.back_icon
            back_btn_font = None
            back_btn_padx = 5
            back_btn_pady = 5
        back_btn = tk.Button(header_frame, text=back_btn_text,
                            image=back_btn_image,
                            command=self.stop_recognition,
                            bg="white", fg="white",
                            font=back_btn_font,
                            bd=0,
                            activebackground="#3C98F4",
                            activeforeground="white",
                            relief="flat", padx=back_btn_padx, pady=back_btn_pady)

        if back_btn_image:
            back_btn.image = back_btn_image
        back_btn.pack(side=tk.LEFT)
        features = []

        if self.gender_var.get():
            features.append("Gender/Age")

        if self.emotion_var.get():
            features.append("Emotion")

        if self.face_var.get():
            features.append("Face Recognition")
        tk.Label(header_frame, text=f"  Advanced Recognition",
                 image=self.recognition_icon, compound="left",
                font=("Segoe UI", 16, "bold"),
                fg=COLORS["primary"], bg=COLORS["card"]).pack(side=tk.LEFT, padx=20)
        self.status_label = tk.Label(header_frame,
                                    text="Initializing camera...",
                                    font=("Segoe UI", 10),
                                    fg=COLORS["warning"], bg=COLORS["card"])
        self.status_label.pack(side=tk.RIGHT, padx=10)
        self.camera_container = tk.Frame(self.main_container, bg="white")
        self.camera_container.pack(fill="both", expand=True, padx=30, pady=20)
        self.camera_container.grid_columnconfigure(0, weight=3)
        self.camera_container.grid_columnconfigure(1, weight=1)
        self.camera_container.grid_rowconfigure(1, weight=1)
        self.spinner = LoadingSpinner(self.camera_container, size=80, speed=100)
        self.spinner.place(relx=0.5, rely=0.5, anchor="center")
        self.spinner.start()

    def load_models(self):

        try:
            self.status_label.config(text="Loading models...")
            self.load_btn.config(state=tk.DISABLED, text="⏳ Loading...")
            emotion_model_path = rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models\emotion\_mini_XCEPTION.106-0.65.hdf5"
            ageProto = rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models\age\age_deploy.prototxt"
            ageModel = rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models\age\age_net.caffemodel"
            genderProto = rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models\gender\gender_deploy.prototxt"
            genderModel = rf"C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models\gender\gender_net.caffemodel"
            missing_files = []
            for path, name in [
                (emotion_model_path, "Emotion Model"),
                (ageProto, "Age Prototxt"),
                (ageModel, "Age Model"),
                (genderProto, "Gender Prototxt"),
                (genderModel, "Gender Model")
            ]:

                if not os.path.exists(path):
                    missing_files.append(f"{name}: {path}")

            if missing_files:
                self.status_label.config(text="Error: Missing model files")
                self.models_status.config(
                    text=f"❌ Missing {len(missing_files)} model file(s).\n" +
                         "Please ensure all model files are in the 'data' folder.",
                    fg=COLORS["danger"]
                )
                self.load_btn.config(state=tk.NORMAL, text="📥 Load Models")
                return

            if self.emotion_var.get():
                self.emotion_model = load_model(emotion_model_path, compile=False)

            if self.gender_var.get():
                self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
                self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
            self.models_loaded = True
            self.start_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Models loaded successfully")
            self.models_status.config(
                text="✅ Models loaded successfully. Click 'Start Recognition' to begin.",
                fg=COLORS["success"]
            )
            self.load_btn.config(text="✅ Models Loaded", bg=COLORS["success"])

        except Exception as e:
            self.status_label.config(text="Error loading models")
            self.models_status.config(
                text=f"❌ Error loading models: {str(e)[:100]}...",
                fg=COLORS["danger"]
            )
            self.load_btn.config(state=tk.NORMAL, text="📥 Load Models", bg="#9B59B6")
            print(f"Error loading models: {e}")

    def start_recognition(self):

        if not self.models_loaded:
            messagebox.showwarning("Models Not Loaded",
                                 "Please load models first before starting recognition.")
            return

        if self.controller.camera_frame_ref:

            try:
                self.controller.camera_frame_ref.stop()

            except:
                pass
            self.controller.camera_frame_ref = None
        self.create_camera_ui()
        self.camera_loading = True
        self.controller.camera_frame_ref = self
        threading.Thread(target=self._initialize_camera_in_background, daemon=True).start()

    def _initialize_camera_in_background(self):

        try:
            self._create_camera_frame()

        except Exception as e:
            self.after(0, self._on_camera_error, str(e))

    def _create_camera_frame(self):

        if not self.camera_loading:
            return
        self.spinner.stop()
        self.spinner.place_forget()
        self.preview_container = tk.Frame(self.camera_container,
                                        bg=COLORS["light"],
                                        width=740,
                                        height=580
                                    )
        self.preview_container.grid(
            row=1, column=0,
            padx=(10, 10), pady=0,
            sticky="n"
        )
        self.preview_container.grid_propagate(False)
        self.preview_label = tk.Label(
            self.preview_container,
            bg=COLORS["light"], bd=0,
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        self.right_panel = tk.Frame(self.camera_container, bg=COLORS["light"])
        self.info_label = tk.Label(
            self.right_panel, text="Feature recognition active",
            font=("Segoe UI", 10),
            fg="#FB05C6", bg=COLORS["light"],
            wraplength=200, justify="left"
        )
        self.info_label.pack(anchor="w",padx=10, pady=(0, 10))
        features_frame = tk.Frame(self.right_panel, bg=COLORS["light"])
        features_frame.pack(anchor="w",padx=10, pady=(0, 15))
        tk.Label(features_frame, text="Active Features:",
                font=("Segoe UI", 9, "bold"),
                fg=COLORS["primary"], bg=COLORS["light"]).pack(anchor="w")

        if self.gender_var.get():
            tk.Label(features_frame, text="✓ Gender & Age Detection",
                    font=("Segoe UI", 8),
                    fg=COLORS["success"], bg=COLORS["light"]).pack(anchor="w")

        if self.emotion_var.get():
            tk.Label(features_frame, text="✓ Emotion Detection",
                    font=("Segoe UI", 8),
                    fg=COLORS["success"], bg=COLORS["light"]).pack(anchor="w")

        if self.face_var.get():
            tk.Label(features_frame, text="✓ Face Recognition",
                    font=("Segoe UI", 8),
                    fg=COLORS["success"], bg=COLORS["light"]).pack(anchor="w")
        button_frame = tk.Frame(self.right_panel, bg=COLORS["light"])
        button_frame.pack(fill="x", pady=10)
        self.stop_btn = tk.Button(
            button_frame, text="⏹ Stop",
            command=self.stop_recognition,
            bg=COLORS["danger"], fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            activebackground="#E88277",
            activeforeground="black"
        )
        self.stop_btn.pack(fill="x", pady=5, padx=10)

        if self.face_var.get():
            self.load_face_recognizers()
        threading.Thread(target=self._open_camera_async, daemon=True).start()

    def _open_camera_async(self):

        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

                if not self.cap.isOpened():
                    raise RuntimeError("Cannot open webcam")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.after(0, self._on_camera_opened)

        except Exception as e:
            self.after(0, self._on_camera_error, str(e))

    def _on_camera_opened(self):
        self.running = True
        self.status_label.config(text="Recognition Active", fg=COLORS["success"])
        self.right_panel.grid(
            row=1, column=1,
            padx=(0, 10),
            pady=15,
            sticky="new"
        )
        self.after(10, self._update_frame)

    def _on_camera_error(self, error_msg):
        self.camera_loading = False
        self.spinner.stop()
        self.spinner.place_forget()
        messagebox.showerror("Camera Error",
                           f"Failed to initialize camera:\n{error_msg}\n\nPlease check your camera connection.")
        self.create_setup_ui()

    def load_face_recognizers(self):

        classifiers_dir = get_classifiers_dir()

        if not os.path.exists(classifiers_dir):
            return
        self.recognizers = []

        classifier_files = [f for f in os.listdir(classifiers_dir)

                          if f.endswith("_classifier.xml")]
        for classifier_file in classifier_files:
            user_name = classifier_file.replace("_classifier.xml", "")

            classifier_path = os.path.join(classifiers_dir, classifier_file)

            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(classifier_path)
                self.recognizers.append((user_name, recognizer))

            except Exception as e:
                print(f"Error loading classifier for {user_name}: {e}")

    def _update_frame(self):

        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()

        if not ok:
            self.after(50, self._update_frame)
            return
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        FONT = cv2.FONT_HERSHEY_COMPLEX
        FONT_SCALE = 0.40
        FONT_THICKNESS = 1
        OUTLINE_THICKNESS = 2
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (26, 188, 156), 2)
            name_text = ""
            gender_age_text = ""
            emotion_text = ""
            emotion_color = (26, 188, 156)

            if self.face_var.get():
                face_roi = gray[y:y + h, x:x + w]
                best_match = None
                best_confidence = 100
                threshold = 70
                for user_name, recognizer in self.recognizers:

                    try:
                        _, confidence = recognizer.predict(face_roi)

                        if confidence < best_confidence and confidence < threshold:
                            best_confidence = confidence
                            best_match = user_name

                    except:
                        continue

                if best_match:
                    name_text = best_match
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    name_text = "Unknown"
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if self.gender_var.get() and self.genderNet and self.ageNet:

                try:
                    roi = frame[y:y + h, x:x + w]
                    blob = cv2.dnn.blobFromImage(
                        roi, 1.0, (227, 227),
                        MODEL_MEAN_VALUES, swapRB=False
                    )
                    self.genderNet.setInput(blob)
                    gender = genderList[self.genderNet.forward()[0].argmax()]
                    self.ageNet.setInput(blob)
                    age = ageList[self.ageNet.forward()[0].argmax()]
                    gender_age_text = f"{gender}, {age}"

                except Exception as e:
                    print("Gender/Age error:", e)

            if self.emotion_var.get() and self.emotion_model:

                try:
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = np.expand_dims(roi, axis=(0, -1))
                    preds = self.emotion_model.predict(roi, verbose=0)[0]
                    emotion = Emotions[preds.argmax()]
                    emotion_text = emotion

                    if emotion in ["happy", "surprised"]:
                        emotion_color = (0, 255, 0)
                    elif emotion in ["sad", "angry"]:
                        emotion_color = (0, 0, 255)
                    elif emotion in ["disgust", "scared"]:
                        emotion_color = (255, 165, 0)
                    else:
                        emotion_color = (255, 255, 0)

                except Exception as e:
                    print("Emotion error:", e)
            bar_height = 22
            bar_spacing = 2
            upper_bars = []
            upper_colors = []

            if name_text:
                upper_bars.append(name_text)
                upper_colors.append((0, 255, 0) if name_text != "Unknown" else (0, 0, 255))

            if gender_age_text:
                upper_bars.append(gender_age_text)
                upper_colors.append((255, 165, 0))
            lower_bars = []
            lower_colors = []

            if emotion_text:
                lower_bars.append(emotion_text)
                lower_colors.append(emotion_color)
            total_upper_height = len(upper_bars) * bar_height + max(0, len(upper_bars) - 1) * bar_spacing
            current_y = max(0, y - total_upper_height - bar_spacing)
            for i, (text, color) in enumerate(zip(upper_bars, upper_colors)):
                bar_y = current_y + i * (bar_height + bar_spacing)
                cv2.rectangle(display, (x, bar_y), (x + w, bar_y + bar_height), color, -1)
                text_size = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = bar_y + bar_height - 6
                cv2.putText(display, text, (text_x, text_y),
                            FONT, FONT_SCALE, (0, 0, 0),
                            OUTLINE_THICKNESS, cv2.LINE_AA)
                cv2.putText(display, text, (text_x, text_y),
                            FONT, FONT_SCALE, (255, 255, 255),
                            FONT_THICKNESS, cv2.LINE_AA)
            current_y = y + h + bar_spacing
            for i, (text, color) in enumerate(zip(lower_bars, lower_colors)):
                bar_y = current_y + i * (bar_height + bar_spacing)
                cv2.rectangle(display, (x, bar_y), (x + w, bar_y + bar_height), color, -1)
                text_size = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = bar_y + bar_height - 6
                cv2.putText(display, text, (text_x, text_y),
                            FONT, FONT_SCALE, (0, 0, 0),
                            OUTLINE_THICKNESS, cv2.LINE_AA)
                cv2.putText(display, text, (text_x, text_y),
                            FONT, FONT_SCALE, (255, 255, 255),
                            FONT_THICKNESS, cv2.LINE_AA)
        img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize(
            (self.preview_container.winfo_width(),
            self.preview_container.winfo_height())
        )
        imgtk = ImageTk.PhotoImage(img)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)

        if self.running:
            self.after(15, self._update_frame)

    def stop_recognition(self, event=None):
        self.camera_loading = False
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        if hasattr(self, 'spinner'):
            self.spinner.stop()
        self.create_setup_ui()

    def stop(self):
        self.stop_recognition()

class CameraFrame(tk.Frame):

    def __init__(self, parent, mode="capture", name=None,
                 video_source=0, max_images=100, auto_start=False,
                 multi_user=False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.mode = mode
        self.name = name
        self.video_source = video_source
        self.max_images = max_images
        self.auto_start = auto_start
        self.multi_user = multi_user
        self.cap = None
        self.running = False
        self.capturing = False
        self.captured_count = 0
        self.recognizer = None
        self.recognizers = []
        self.configure(bg=COLORS["card"], relief="raised", bd=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header_frame = tk.Frame(self, bg=COLORS["primary"])
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        if self.multi_user:
            mode_text = "FACE RECOGNITION (ALL USERS)"
            user_text = "Multiple Users"
        else:
            mode_text = "DATASET CAPTURE" if mode == "capture" else "FACE RECOGNITION"
            user_text = f"User: {name}" if name else ""
        tk.Label(
            header_frame, text=mode_text,
            font=("Segoe UI", 12, "bold"),
            fg="white", bg=COLORS["primary"]
        ).pack(side=tk.LEFT, padx=15, pady=10)

        if user_text:
            tk.Label(
                header_frame, text=user_text,
                font=("Segoe UI", 10),
                fg="white", bg=COLORS["primary"]
            ).pack(side=tk.RIGHT, padx=15, pady=10)
        preview_container = tk.Frame(self, bg=COLORS["border"])
        preview_container.grid(
            row=1, column=0,
            padx=(20, 10), pady=15,
            sticky="nsew"
        )
        self.preview_label = tk.Label(
            preview_container,
            bg=COLORS["light"], bd=0
        )
        self.preview_label.pack(padx=2, pady=2, fill="both", expand=True)
        right_panel = tk.Frame(self, bg=COLORS["light"])
        right_panel.grid(
            row=1, column=1,
            padx=(10, 20), pady=15,
            sticky="new"
        )
        self.info_label = tk.Label(
            right_panel, text="Ready to start",
            font=("Segoe UI", 9),
            fg=COLORS["secondary"], bg=COLORS["light"],
            wraplength=200, justify="left"
        )
        self.info_label.pack(anchor="w", pady=(0, 10))

        if self.mode == "capture":
            self.count_label = tk.Label(
                right_panel,
                text=f" 0 / {self.max_images} images",
                font=("Segoe UI", 9, "bold"),
                fg=COLORS["accent"], bg=COLORS["light"]
            )
            self.count_label.pack(anchor="w", pady=5)
            self.progress_canvas = tk.Canvas(
                right_panel, height=18,
                bg=COLORS["light"], highlightthickness=0
            )
            self.progress_canvas.pack(fill="x", pady=(5, 15))
            self.progress_bar = self.progress_canvas.create_rectangle(
                0, 0, 0, 18,
                fill=COLORS["accent"], outline=""
            )
        else:

            if self.multi_user:
                user_count = len(self.recognizers) if hasattr(self, 'recognizers') else 0
                self.status_label = tk.Label(
                    right_panel,
                    text=f"Status: Idle\nLoaded {user_count} user(s)",
                    font=("Segoe UI", 9),
                    fg=COLORS["secondary"], bg=COLORS["light"],
                    justify="left"
                )
            else:
                self.status_label = tk.Label(
                    right_panel,
                    text="Status: Idle",
                    font=("Segoe UI", 9),
                    fg=COLORS["secondary"], bg=COLORS["light"]
                )
            self.status_label.pack(anchor="w", pady=(0, 15))
        button_frame = tk.Frame(right_panel, bg=COLORS["light"])
        button_frame.pack(fill="x", pady=10)

        if self.mode == "capture":
            self.start_btn = tk.Button(
                button_frame, text="▶ Start Capture",
                command=self.toggle_capture,
                bg=COLORS["accent"], fg="white",
                font=("Segoe UI", 10, "bold"),
                relief="flat"
            )
            self.start_btn.pack(fill="x", pady=5)
            self.stop_btn = tk.Button(
                button_frame, text="⏹ Stop & Close",
                command=self.stop,
                bg=COLORS["danger"], fg="white",
                font=("Segoe UI", 10, "bold"),
                relief="flat"
            )
            self.stop_btn.pack(fill="x", pady=5)
        else:
            self.start_btn = tk.Button(
                button_frame, text="▶ Start Recognition",
                command=self.toggle_recognize,
                bg=COLORS["accent"], fg="white",
                font=("Segoe UI", 10, "bold"),
                relief="flat"
            )
            self.start_btn.pack(fill="x", pady=5)
            self.stop_btn = tk.Button(
                button_frame, text="⏹ Stop",
                command=self.stop,
                bg=COLORS["danger"], fg="white",
                font=("Segoe UI", 10, "bold"),
                relief="flat"
            )
            self.stop_btn.pack(fill="x", pady=5)
        tk.Button(
            button_frame, text="✕ Close",
            command=self.stop,
            bg=COLORS["secondary"], fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat"
        ).pack(fill="x", pady=(10, 0))

        if self.mode == "capture" and self.name:
            self.dataset_dir = get_user_dataset_dir(self.name)
            os.makedirs(self.dataset_dir, exist_ok=True)

        if self.mode == "recognize" and self.multi_user:
            self.load_all_recognizers()
        self.open_camera()

        if self.mode == "capture" and self.auto_start:
            self.after(200, self.toggle_capture)

    def load_all_recognizers(self):

        classifiers_dir = get_classifiers_dir()

        if not os.path.exists(classifiers_dir):
            self.info_label.config(text="No trained models found")
            return
        self.recognizers = []

        classifier_files = [f for f in os.listdir(classifiers_dir)

                          if f.endswith("_classifier.xml")]

        if not classifier_files:
            self.info_label.config(text="No trained models found")
            return
        for classifier_file in classifier_files:
            user_name = classifier_file.replace("_classifier.xml", "")

            classifier_path = os.path.join(classifiers_dir, classifier_file)

            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(classifier_path)
                self.recognizers.append((user_name, recognizer))

            except Exception as e:
                print(f"Error loading classifier for {user_name}: {e}")

        if self.recognizers:
            self.info_label.config(text=f"Loaded {len(self.recognizers)} user(s)")

            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Status: Idle\nLoaded {len(self.recognizers)} user(s)")
        else:
            self.info_label.config(text="Failed to load any classifiers")

    def open_camera(self):

        try:
            self.cap = cv2.VideoCapture(self.video_source)

            if not self.cap.isOpened():
                raise RuntimeError("Cannot open webcam")

        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.cap = None

    def toggle_capture(self):

        if not self.capturing:
            self.capturing = True
            self.running = True
            self.start_btn.config(text="⏸ Pause Capture")
            self.info_label.config(text="🎥 Capturing... Move your face slowly")
            self.after(10, self._update_frame)
        else:
            self.capturing = False
            self.start_btn.config(text="▶ Resume Capture")
            self.info_label.config(text=f"⏸ Paused ({self.captured_count})")

    def toggle_recognize(self):

        if not self.running:
            self.running = True
            self.start_btn.config(text="⏸ Pause Recognition")

            if self.multi_user:
                self.status_label.config(text="Status: 🔍 Recognizing all users...")
            else:
                self.status_label.config(text="Status: 🔍 Recognizing...")
            self.after(10, self._update_frame)
        else:
            self.running = False
            self.start_btn.config(text="▶ Resume Recognition")
            self.status_label.config(text="Status: ⏸ Paused")

    def _update_frame(self):

        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()

        if not ok:
            self.after(50, self._update_frame)
            return
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (26, 188, 156), 2)

            if self.mode == "capture" and self.capturing:
                crop = frame[y:y+h, x:x+w]
                path = os.path.join(self.dataset_dir, f"{self.captured_count}_{self.name}.jpg")
                cv2.imwrite(path, crop)
                self.captured_count += 1
                progress = (self.captured_count / self.max_images) * self.progress_canvas.winfo_width()
                self.progress_canvas.coords(self.progress_bar, 0, 0, progress, 18)
                self.count_label.config(
                    text=f"📷 {self.captured_count} / {self.max_images}"
                )

                if self.captured_count >= self.max_images:
                    self.running = False
                    self.capturing = False
                    self.start_btn.config(text="✅ Capture Complete")
                    self.info_label.config(text="✅ Capture finished")
                    break
            elif self.mode == "recognize" and self.running:
                face_roi = gray[y:y+h, x:x+w]

                if self.multi_user and self.recognizers:
                    best_match = None
                    best_confidence = 100
                    threshold = 70
                    for user_name, recognizer in self.recognizers:

                        try:
                            id_, confidence = recognizer.predict(face_roi)

                            if confidence < best_confidence and confidence < threshold:
                                best_confidence = confidence
                                best_match = user_name

                        except:
                            continue

                    if best_match:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(display, f"{best_match} ({best_confidence:.1f})",
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(display, "Unknown", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                elif self.recognizer and self.name:
                    id_, confidence = self.recognizer.predict(face_roi)

                    if confidence < 70:
                        cv2.putText(display, f"{self.name} ({confidence:.1f})",
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, "Unknown", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)

        if self.running:
            self.after(15, self._update_frame)

    def stop(self):
        self.running = False
        self.capturing = False

        if self.cap:
            self.cap.release()
            self.cap = None
        self.destroy()

class RecognizeAllPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        header_frame = tk.Frame(self, bg=COLORS["card"])
        header_frame.pack(fill="x", padx=30, pady=(20, 10))
        back_btn = tk.Button(header_frame, text="← Back",
                             command=lambda: controller.show_frame("StartPage"),
                             bg="white", fg="white",
                             font=("Segoe UI", 10, "bold"),
                             bd=0,
                             activebackground="#3C98F4",
                             activeforeground="white",
                             relief="flat", padx=10, pady=5)
        back_btn.pack(side=tk.LEFT)
        tk.Label(header_frame, text="🔍 Recognize All Users",
                 font=("Segoe UI", 16, "bold"),
                 fg=COLORS["primary"], bg=COLORS["card"]).pack(side=tk.LEFT, padx=20)
        self.status_label = tk.Label(header_frame,
                                   text="Ready",
                                   font=("Segoe UI", 10),
                                   fg=COLORS["success"], bg=COLORS["card"])
        self.status_label.pack(side=tk.RIGHT, padx=10)
        instructions = tk.Label(self,
                              text="• The system will recognize any trained user\n• Face will be highlighted with user name\n• Red box indicates unknown person",
                              font=("Segoe UI", 10),
                              fg=COLORS["secondary"],
                              bg=COLORS["card"],
                              justify=tk.LEFT)
        instructions.pack(pady=10)
        self.camera_container = tk.Frame(self, bg=COLORS["border"])
        self.camera_container.pack(padx=30, pady=10, fill="both", expand=True)
        self.trained_users = get_all_trained_users()

        if not self.trained_users:
            no_users_label = tk.Label(self.camera_container,
                                    text="No trained users found!\n\nPlease train at least one user first.",
                                    font=("Segoe UI", 12),
                                    fg=COLORS["danger"],
                                    bg=COLORS["card"],
                                    justify=tk.CENTER)
            no_users_label.pack(expand=True, fill="both")
            back_btn = tk.Button(self.camera_container,
                               text="Go Back",
                               command=lambda: controller.show_frame("StartPage"),
                               bg="white", fg="white",
                               font=("Segoe UI", 10, "bold"),
                               relief="flat", padx=15, pady=8)
            back_btn.pack(pady=20)
        else:
            users_count = len(self.trained_users)
            count_label = tk.Label(self,
                                 text=f"✓ Loaded {users_count} trained user(s)",
                                 font=("Segoe UI", 10, "bold"),
                                 fg=COLORS["success"],
                                 bg=COLORS["card"])
            count_label.pack(pady=5)
            button_frame = tk.Frame(self, bg=COLORS["card"])
            button_frame.pack(pady=20)
            tk.Button(button_frame, text="▶ Start Recognition",
                      command=self.open_camera,
                      bg=COLORS["accent"], fg="white",
                      font=("Segoe UI", 10, "bold"),
                      relief="flat", padx=15, pady=8).pack(side=tk.LEFT, padx=5)
            self.stop_btn = tk.Button(button_frame, text="⏹ Stop",
                                     command=self.stop_camera,
                                     bg=COLORS["danger"], fg="white",
                                     font=("Segoe UI", 10, "bold"),
                                     relief="flat", padx=15, pady=8,
                                     state=tk.DISABLED)
            self.stop_btn.pack(side=tk.LEFT, padx=5)

    def open_camera(self):

        if self.controller.camera_frame_ref:

            try:
                self.controller.camera_frame_ref.stop()

            except:
                pass
            self.controller.camera_frame_ref = None
        cam = CameraFrame(self.camera_container, mode="recognize",
                         multi_user=True, name=None)
        cam.pack(fill="both", expand=True)
        self.controller.camera_frame_ref = cam
        self.status_label.config(text="Recognition active")
        self.stop_btn.config(state=tk.NORMAL)

    def stop_camera(self):

        if self.controller.camera_frame_ref:

            try:
                self.controller.camera_frame_ref.stop()
                self.controller.camera_frame_ref = None
                self.status_label.config(text="Stopped")
                self.stop_btn.config(state=tk.DISABLED)

            except Exception as e:
                print(f"Error stopping camera: {e}")

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        center_frame = tk.Frame(self, bg=COLORS["card"], border=5, borderwidth=2, relief="solid")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(center_frame, text="👤",
                 font=("Segoe UI", 48),
                 fg=COLORS["accent"], bg=COLORS["card"]).pack(pady=(10, 10))
        tk.Label(center_frame, text="Face Recognition Pro",
                 font=controller.title_font,
                 fg=COLORS["primary"], bg=COLORS["card"]).pack(pady=(0, 20), padx=10)
        features_frame = tk.Frame(center_frame, bg=COLORS["card"])
        features_frame.pack(pady=(0, 10))
        features = [
            "✓ High-precision face detection",
            "✓ Real-time recognition",
            "✓ Multi-user support",
            "✓ Secure and local processing"
        ]
        for feature in features:
            tk.Label(features_frame, text=feature,
                     font=("Segoe UI", 10), bg=COLORS["card"],
                     fg=COLORS["secondary"]).pack(anchor="w",padx=3, pady=3)
        button_frame = tk.Frame(center_frame, bg=COLORS["card"])
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="👤👤Show Users",
                  command=lambda: controller.show_frame("ShowUsersPage"),
                  bg="#412AF3", fg="white",
                  font=("Arial", 11, "bold"),
                  activebackground="#ADA6E8",
                  activeforeground="black",
                  bd=0,
                  relief="flat", padx=20, pady=8, width=20).pack(pady=10)
        tk.Button(button_frame, text="➕ New User",
                  command=lambda: controller.show_frame("SignupPage"),
                  bg="#08A571", fg="white",
                  font=("Segoe UI", 11, "bold"),
                  bd=0,
                  activebackground="#287D6C",
                  activeforeground="white",
                  relief="flat", padx=20, pady=8, width=20).pack(pady=10)
        tk.Button(button_frame, text="🔍 Recognize",
                  command=lambda: controller.show_frame("FeatureRecognitionPage"),
                  bg="#469BC0", fg="white",
                  font=("Segoe UI", 11, "bold"),
                  bd=0,
                  activebackground="#8FDEE5",
                  activeforeground="black",
                  relief="flat", padx=20, pady=8, width=20).pack(pady=10)
        footer_frame = tk.Frame(self, bg=COLORS["card"])
        footer_frame.pack(side="bottom", fill="x", pady=20)

class SignupPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        back_btn_text = "← Back"
        back_btn_image = None
        back_btn_font = ("Segoe UI", 10, "bold")
        back_btn_padx = 10
        back_btn_pady = 5

        if controller.back_icon:
            back_btn_text = ""
            back_btn_image = controller.back_icon
            back_btn_font = None
            back_btn_padx = 5
            back_btn_pady = 5
        back_btn = tk.Button(self, text=back_btn_text,
                            image=back_btn_image,
                            command=lambda: controller.show_frame("StartPage"),
                            bg="white", fg="white",
                            font=back_btn_font,
                            bd=0,
                            activeforeground="white",
                            relief="flat", padx=back_btn_padx, pady=back_btn_pady,
                            activebackground="#3C98F4")

        if back_btn_image:
            back_btn.image = back_btn_image
        back_btn.place(x=20, y=20)
        content_frame = tk.Frame(self, bg=COLORS["card"])
        content_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(content_frame, text="👤 New User Registration",
                 font=("Segoe UI", 18, "bold"),
                 fg=COLORS["primary"], bg=COLORS["card"]).pack(pady=(0, 30))
        input_frame = tk.Frame(content_frame, bg=COLORS["card"])
        input_frame.pack(pady=10)

        try:
            img = Image.open(resource_path(r"Images/writing.png")).resize((42, 42))
            self.writing_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.writing_icon = None
        tk.Label(input_frame, image=self.writing_icon,
                 bg=COLORS["card"]).pack(side=tk.LEFT, padx=(0, 10))
        self.entry = tk.Entry(input_frame,
                              font=("Segoe UI", 12),
                              width=30, bd=1, relief="solid")
        self.entry.pack(side=tk.LEFT)
        self.entry.focus()
        tk.Label(content_frame,
                 text="Enter a unique username for the new user",
                 font=("Segoe UI", 10),
                 fg=COLORS["secondary"], bg=COLORS["card"]).pack(pady=(0, 30))
        next_btn = tk.Button(content_frame, text="Next →",
                             command=self.next_page,
                             bg=COLORS["accent"], fg="white",
                             font=("Segoe UI", 11, "bold"),
                             bd=0,
                             activebackground="#07A686",
                             activeforeground="white",
                             relief="flat", padx=20, pady=8, width=20)
        next_btn.pack(pady=10)
        self.entry.bind('<Return>', lambda e: self.next_page())

    def next_page(self):
        name = self.entry.get().strip()

        if not name:
            messagebox.showerror("Error", "Please enter a username.")
            return

        if name == "None":
            messagebox.showerror("Error", "Username cannot be 'None'")
            return
        user_folder = get_user_dataset_dir(name)

        if os.path.exists(user_folder):
            messagebox.showerror("Error", f"User '{name}' already exists!\nPlease choose a different username.")
            return
        self.controller.active_name = name
        self.controller.show_frame("CapturePage")

class SelectUserPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        back_btn_text = "← Back"
        back_btn_image = None
        back_btn_font = ("Segoe UI", 10, "bold")
        back_btn_padx = 10
        back_btn_pady = 5

        if controller.back_icon:
            back_btn_text = ""
            back_btn_image = controller.back_icon
            back_btn_font = None
            back_btn_padx = 5
            back_btn_pady = 5
        back_btn = tk.Button(self, text=back_btn_text,
                             image=back_btn_image,
                             command=lambda: controller.show_frame("StartPage"),
                             bg="white", fg="white",
                             font=back_btn_font,
                             relief="flat", padx=back_btn_padx, pady=back_btn_pady,
                             bd=0,
                                activebackground="#3C98F4",
                                activeforeground="white"
                                )
        back_btn.place(x=20, y=20)

        if back_btn_image:
            back_btn.image = back_btn_image
        content_frame = tk.Frame(self, bg=COLORS["card"])
        content_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(content_frame, text="🔍 Select User",
                 font=("Segoe UI", 18, "bold"),
                 fg=COLORS["primary"], bg=COLORS["card"]).pack(pady=(0, 30))
        dropdown_frame = tk.Frame(content_frame, bg=COLORS["card"])
        dropdown_frame.pack(pady=10)
        self.selected_user = tk.StringVar(self)
        self.user_names = []
        self.option_menu = tk.OptionMenu(dropdown_frame, self.selected_user, "")
        self.option_menu.config(font=("Segoe UI", 11), width=25)
        self.option_menu.pack(side=tk.LEFT, padx=(0, 10))
        refresh_btn = tk.Button(dropdown_frame, text="🔄",
                                command=self.refresh,
                                bg=COLORS["secondary"], fg="white",
                                font=("Segoe UI", 10),
                                bd=0,
                                activebackground="#3C98F4",
                                activeforeground="white",
                                relief="flat", width=3)
        refresh_btn.pack(side=tk.LEFT)
        tk.Label(content_frame, text="─ OR ─",
                 font=("Segoe UI", 10),
                 fg=COLORS["secondary"], bg=COLORS["card"]).pack(pady=15)
        tk.Label(content_frame, text="Enter username manually:",
                 bg=COLORS["card"]).pack()
        self.entry = tk.Entry(content_frame,
                              font=("Segoe UI", 11),
                              width=30, bd=1, relief="solid")
        self.entry.pack(pady=10)
        self.status_label = tk.Label(content_frame, text="",
                                     font=("Segoe UI", 9),
                                     fg=COLORS["secondary"], bg=COLORS["card"])
        self.status_label.pack(pady=5)
        next_btn = tk.Button(content_frame, text="Start Recognition →",
                             command=self.next_page,
                             bg=COLORS["accent"], fg="white",
                             font=("Segoe UI", 11, "bold"),
                             relief="flat", padx=20, pady=8, width=25)
        next_btn.pack(pady=20)
        self.refresh()
        self.entry.bind('<Return>', lambda e: self.next_page())

    def refresh(self):

        classifiers_dir = get_classifiers_dir()
        self.user_names = []

        if os.path.exists(classifiers_dir):
            for f in os.listdir(classifiers_dir):

                if f.endswith("_classifier.xml"):
                    self.user_names.append(f.replace("_classifier.xml",""))
        menu = self.option_menu["menu"]
        menu.delete(0, "end")
        for name in sorted(self.user_names):
            menu.add_command(label=name,
                            command=lambda value=name: self.selected_user.set(value))

        if self.user_names:
            self.selected_user.set(self.user_names[0])
            self.status_label.config(text=f"Found {len(self.user_names)} trained user(s)")
        else:
            self.selected_user.set("")
            self.status_label.config(text="No trained users found")

    def next_page(self):
        name = self.selected_user.get().strip()

        if not name:
            name = self.entry.get().strip()

        if not name:
            messagebox.showerror("Error", "Please select or enter a username.")
            return
        clf_path = os.path.join(get_classifiers_dir(), f"{name}_classifier.xml")

        if not os.path.exists(clf_path):
            messagebox.showerror("Error", f"No trained model found for '{name}'.\nPlease train this user first.")
            return
        self.controller.active_name = name
        self.controller.show_frame("RecognizePage")

class LoadingSpinner(tk.Label):

    def __init__(self, parent, size=100, speed=100, **kwargs):
        super().__init__(parent, **kwargs)
        self.size = size
        self.speed = speed
        self.angle = 0
        self.running = False

    def start(self):
        self.running = True
        self._animate()

    def stop(self):
        self.running = False
        self.place_forget()

    def _animate(self):

        if not self.running:
            return
        img = Image.new("RGBA", (self.size, self.size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.arc((5, 5, self.size - 5, self.size - 5),
                start=self.angle, end=self.angle + 100,
                fill="#1ABC9C", width=6)
        self.angle = (self.angle + 15) % 360
        tk_img = ImageTk.PhotoImage(img)
        self.img = tk_img
        self.config(image=tk_img, bg="#FFFFFF")  # Also set Label background
        self.after(self.speed, self._animate)

class CapturePage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        header_frame = tk.Frame(self, bg=COLORS["card"])
        header_frame.pack(fill="x", padx=30, pady=(20, 10))
        back_btn_text = "← Back"
        back_btn_image = None
        back_btn_font = ("Segoe UI", 10, "bold")
        back_btn_padx = 10
        back_btn_pady = 5

        if controller.back_icon:
            back_btn_text = ""
            back_btn_image = controller.back_icon
            back_btn_font = None
            back_btn_padx = 5
            back_btn_pady = 5
        back_btn = tk.Button(header_frame, text=back_btn_text,
                            image=back_btn_image,
                             command=lambda: controller.show_frame("SignupPage"),
                             bg="white", fg="white",
                             font=back_btn_font,
                             bd=0,
                             activebackground="#3C98F4",
                             activeforeground="white",
                             relief="flat", padx=back_btn_padx, pady=back_btn_pady,)

        if back_btn_image:
            back_btn.image = back_btn_image
        back_btn.pack(side=tk.LEFT)

        try:
            img = Image.open(resource_path(r"Images/camera.png")).resize((42, 42))
            self.capturing_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.capturing_icon = None
        tk.Label(header_frame, text="  Capture Training Data", image=self.capturing_icon,
                 compound=tk.LEFT,
                 font=("Segoe UI", 16, "bold"),
                 fg=COLORS["primary"], bg=COLORS["card"]).pack(side=tk.LEFT, padx=20)

        if controller.active_name:
            user_badge = tk.Label(header_frame,
                                  text=f"User: {controller.active_name}",
                                  font=("Segoe UI", 10, "bold"),
                                  fg="white",
                                  bg=COLORS["accent"],
                                  padx=10, pady=5)
            user_badge.pack(side=tk.RIGHT)
        instructions = tk.Label(self,
                                text="• Position your face in the frame\t\t\t• Ensure good lighting\n• Move slowly for varied angles\t\t\t• Capture 100+ images for best results",
                                font=("Segoe UI", 12),
                                fg="#022CFF",
                                bg=COLORS["card"],
                                justify=tk.LEFT)
        instructions.pack(pady=10)
        self.camera_main_frame = tk.Frame(self, bg=COLORS["card"])
        self.camera_main_frame.pack(fill="both", expand=True, padx=30, pady=10)
        self.preview_frame = tk.Frame(self.camera_main_frame, bg=COLORS["border"])
        self.preview_frame.config(width=640, height=480)
        self.preview_frame.pack(
            side=tk.LEFT,
            padx=(0, 20),
            pady=10
        )
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame, bg=COLORS["light"])
        self.preview_label.config(width=640, height=480)
        self.preview_label.pack()
        control_panel = tk.Frame(self.camera_main_frame, bg=COLORS["light"])
        control_panel.pack(side=tk.RIGHT, padx=(20, 0), pady=10, fill="y")

        try:
            img = Image.open(resource_path(r"Images/label_camera.png")).resize((25, 25))
            self.cam_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.cam_icon = None
        self.count_label = tk.Label(control_panel,
                                    text="  0 / 100 images",
                                    image=self.cam_icon,
                                    compound="left",
                                    font=("Segoe UI", 11, "bold"),
                                    fg=COLORS["accent"],
                                    bg=COLORS["light"])
        self.count_label.pack(anchor="w", pady=20, padx=10)
        self.progress_canvas = tk.Canvas(control_panel, height=20,
                                         bg=COLORS["light"], highlightthickness=0)
        self.progress_canvas.pack(fill="x", pady=(5, 20), padx=10)
        self.progress_bar = self.progress_canvas.create_rectangle(
            0, 0, 0, 20, fill=COLORS["accent"], outline=""
        )
        button_frame = tk.Frame(control_panel, bg=COLORS["light"])
        button_frame.pack(fill="x", pady=10)
        self.open_camera_btn = tk.Label(button_frame,
                                        image=self.capturing_icon,
                                        compound="left",
                                        text="▶ Start Capture",
                                        bg=COLORS["accent"], fg="white",
                                        font=("Segoe UI", 11, "bold"),
                                        relief="flat", padx=20, pady=12,
                                        cursor="hand2")
        self.open_camera_btn.pack(fill="x", padx=20, pady=5)
        self.train_model_btn = tk.Button(button_frame,
                                         text="⌛ Train Model",
                                         bg="#007DFB", fg="white",
                                         font=("Segoe UI", 11, "bold"),
                                         relief="flat", padx=20, pady=12,
                                         bd=0,
                                         activebackground="#05D9FF",
                                         activeforeground="black",
                                         command=self.start_training_from_capture)
        self.open_camera_btn.bind("<Enter>", lambda e: self.open_camera_btn.config(bg="#16A085"))
        self.open_camera_btn.bind("<Leave>", lambda e: self.open_camera_btn.config(bg=COLORS["accent"]))
        self.open_camera_btn.bind("<ButtonPress-1>", lambda e: self.open_camera_btn.config(bg="#138D75"))
        self.open_camera_btn.bind("<ButtonRelease-1>", lambda e: self.toggle_capture())
        tk.Button(button_frame, text="✕ Close & Back",
                  command=self.close_and_back,
                  bg="#FF0404", fg="white",
                  font=("Segoe UI", 11, "bold"),
                  bd=0,
                  activebackground="#F74545",
                  activeforeground="white",
                  relief="flat", padx=20, pady=12).pack(fill="x", padx=20, pady=(10, 0))
        self.cap = None
        self.running = False
        self.capturing = False
        self.captured_count = 0
        self.max_images = 100
        self.frame_job = None
        self.frame_count = 0

        if controller.active_name:
            self.dataset_dir = get_user_dataset_dir(controller.active_name)
            os.makedirs(self.dataset_dir, exist_ok=True)

    def toggle_capture(self):

        if not self.running:
            self.start_camera()

        if not self.capturing:
            self.capturing = True
            self.running = True
            self.open_camera_btn.config(text="⏸ Pause Capture", bg="#F39C12")
            self.schedule_frame_update()
        else:
            self.capturing = False
            self.open_camera_btn.config(text="▶ Resume Capture", bg=COLORS["accent"])

    def start_camera(self):

        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                raise RuntimeError("Cannot open webcam")
            self.running = True
            self.capturing = False
            self.open_camera_btn.config(text="▶ Start Capture", bg=COLORS["accent"])

        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.cap = None
            self.running = False

    def schedule_frame_update(self):

        if self.frame_job is None:
            self.frame_job = self.after(10, self.update_frame)

    def update_frame(self):
        self.frame_job = None

        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()

        if not ok:
            self.schedule_frame_update()
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if self.capturing and self.captured_count < self.max_images:
                self.frame_count += 1

                if self.frame_count % 3 == 0:
                    crop = frame[y:y+h, x:x+w]
                    path = os.path.join(
                        self.dataset_dir,
                        f"{self.captured_count}_{self.controller.active_name}.jpg"
                    )
                    cv2.imwrite(path, crop)
                    self.captured_count += 1
                    w = max(1, self.progress_canvas.winfo_width())
                    progress_width = (self.captured_count / self.max_images) * w
                    self.progress_canvas.coords(self.progress_bar, 0, 0, progress_width, 20)
                    self.count_label.config(text=f"  {self.captured_count} / {self.max_images} images")

                    if self.captured_count >= self.max_images:
                        self.capturing = False
                        self.running = False
                        self.open_camera_btn.config(text="✅ Capture Complete", bg=COLORS["success"])
                        self.after(50, self._show_train_button)
                        break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)

        if self.running:
            self.schedule_frame_update()

    def _show_train_button(self):
        self.stop_camera()

        try:
            self.open_camera_btn.pack_forget()

        except Exception:
            pass
        self.train_model_btn.pack(fill="x", padx=20, pady=5)
        self.count_label.config(text="  Ready to train model for: " + (self.controller.active_name or "unknown"))

    def start_training_from_capture(self):
        self.train_model_btn.config(state=tk.DISABLED, text="⏳ Training...")
        self.progress_canvas.coords(self.progress_bar, 0, 0, 0, 20)
        self.count_label.config(text="Training: 0%")
        username = self.controller.active_name

        if not username:
            messagebox.showerror("Error", "No username selected for training.")
            self._restore_open_button()
            return

        def progress_callback(progress, message):
            self.after(0, lambda: self._update_training_progress(progress, message))

        def background_train():

            try:
                success, msg = train_classifier_with_progress(username, progress_callback)
                self.after(0, lambda: self._on_training_finished(success, msg, username))

            except Exception as e:
                self.after(0, lambda: self._on_training_finished(False, f"Training error: {e}", username))
        threading.Thread(target=background_train, daemon=True).start()

    def _update_training_progress(self, progress, message=""):
        w = max(1, self.progress_canvas.winfo_width())
        width = (progress / 100.0) * w
        self.progress_canvas.coords(self.progress_bar, 0, 0, width, 20)
        self.count_label.config(text=f"Training: {progress}% — {message}")
        self.update_idletasks()

    def _on_training_finished(self, success, message, username):
        self.train_model_btn.config(state=tk.NORMAL, text="⏳ Train Model")

        try:
            self.train_model_btn.pack_forget()

        except Exception:
            pass

        if success:
            messagebox.showinfo("Training Complete", f"✅ Model trained successfully for {username}!")
        else:
            messagebox.showerror("Training Failed", f"Training failed: {message}")
        self._restore_open_button()
        self.controller.show_frame("StartPage")

    def _restore_open_button(self):

        try:
            self.train_model_btn.pack_forget()

        except Exception:
            pass

        try:
            self.open_camera_btn.pack(fill="x", padx=20, pady=5)
            self.open_camera_btn.config(text="▶ Start Capture", bg=COLORS["accent"])

        except Exception:
            pass
        self.progress_canvas.coords(self.progress_bar, 0, 0, 0, 20)
        self.count_label.config(text=f"  0 / {self.max_images} images")

    def stop_camera(self):
        self.running = False
        self.capturing = False

        if self.frame_job:
            self.after_cancel(self.frame_job)
            self.frame_job = None

        if self.cap:
            self.cap.release()
            self.cap = None
        self.preview_label.config(image="")
        self.preview_label.imgtk = None
        self.open_camera_btn.config(text="▶ Start Capture", bg=COLORS["accent"])

    def close_and_back(self):
        self.stop_camera()
        self.controller.show_frame("SignupPage")

    def on_show(self):
        self.stop_camera()
        self.captured_count = 0
        self.frame_count = 0

        if self.controller.active_name:
            self.dataset_dir = get_user_dataset_dir(self.controller.active_name)
            os.makedirs(self.dataset_dir, exist_ok=True)

            if os.path.exists(self.dataset_dir):
                existing = len([f for f in os.listdir(self.dataset_dir)

                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

                if existing > 0:
                    self.captured_count = existing
                    self.count_label.config(text=f"  {existing} / {self.max_images} images")
                    progress_width = (existing / self.max_images) * self.progress_canvas.winfo_width()
                    self.progress_canvas.coords(self.progress_bar, 0, 0, progress_width, 20)

    def capture_complete(self):
        self.stop_camera()
        messagebox.showinfo(
            "Capture Complete",
            f"Successfully captured {self.max_images} images for {self.controller.active_name}."
        )
        self.controller.show_frame("ShowUsersPage")

class RecognizePage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        header_frame = tk.Frame(self, bg=COLORS["card"])
        header_frame.pack(fill="x", padx=30, pady=(20, 10))
        back_btn_text = "← Back"
        back_btn_image = None
        back_btn_font = ("Segoe UI", 10, "bold")
        back_btn_padx = 10
        back_btn_pady = 5

        if controller.back_icon:
            back_btn_text = ""
            back_btn_image = controller.back_icon
            back_btn_font = None
            back_btn_padx = 5
            back_btn_pady = 5
        back_btn = tk.Button(header_frame, text=back_btn_text,
                             image=back_btn_image,
                             command=lambda: controller.show_frame("SelectUserPage"),
                             bg="white", fg="white",
                             font=back_btn_font,
                             bd=0,
                                activebackground="#3C98F4",
                                activeforeground="white",
                             relief="flat", padx=back_btn_padx, pady=back_btn_pady,)
        back_btn.pack(side=tk.LEFT)
        tk.Label(header_frame, text="🔍 Face Recognition",
                 font=("Segoe UI", 16, "bold"),
                 fg=COLORS["primary"], bg=COLORS["card"]).pack(side=tk.LEFT, padx=20)

        if controller.active_name:
            user_badge = tk.Label(header_frame,
                                  text=f"Recognizing: {controller.active_name}",
                                  font=("Segoe UI", 10, "bold"),
                                  fg="white",
                                  bg=COLORS["success"],
                                  padx=10, pady=5)
            user_badge.pack(side=tk.RIGHT)
        instructions = tk.Label(self,
                                text="• Ensure good lighting\n• Look directly at the camera\n• Recognition confidence will be displayed",
                                font=("Segoe UI", 10),
                                fg=COLORS["secondary"],
                                bg=COLORS["card"],
                                justify=tk.LEFT)
        instructions.pack(pady=10)
        self.camera_container = tk.Frame(self, bg=COLORS["border"])
        self.camera_container.pack(padx=30, pady=10, fill="both", expand=True)
        self.spinner = LoadingSpinner(self.camera_container)
        button_frame = tk.Frame(self, bg=COLORS["card"])
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="▶ Start Recognition",
                  command=self.open_camera,
                  bg=COLORS["accent"], fg="white",
                  font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=15, pady=8).pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(button_frame,
                                     text="Ready",
                                     font=("Segoe UI", 10),
                                     fg=COLORS["secondary"], bg=COLORS["card"])
        self.status_label.pack(side=tk.LEFT, padx=20)

    def open_camera(self):
        name = self.controller.active_name

        if not name:
            messagebox.showerror("Error", "No user selected.")
            return
        clf_path = os.path.join(get_classifiers_dir(), f"{name}_classifier.xml")

        if not os.path.exists(clf_path):
            messagebox.showerror("Error", f"No trained model found for '{name}'.")
            return

        if self.controller.camera_frame_ref:

            try:
                self.controller.camera_frame_ref.stop()

            except:
                pass
            self.controller.camera_frame_ref = None
        cam = CameraFrame(self.camera_container, mode="recognize", name=name)
        cam.pack(fill="both", expand=True)
        self.controller.camera_frame_ref = cam
        self.status_label.config(text="Recognition active")

class ShowUsersPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["card"])
        self.controller = controller
        self.is_capturing_mode = False
        self.current_capturing_user = None
        self.frame_job = None
        self.frame_count = 0
        self.main_container = tk.Frame(self, bg=COLORS["card"])
        self.main_container.pack(fill="both", expand=True, padx=30, pady=20)
        header_frame = tk.Frame(self.main_container, bg=COLORS["card"])
        header_frame.pack(fill="x", pady=(0, 20))
        back_btn_text = "← Back"
        back_btn_image = None
        back_btn_font = ("Segoe UI", 10, "bold")
        back_btn_padx = 10
        back_btn_pady = 5

        if controller.back_icon:
            back_btn_text = ""
            back_btn_image = controller.back_icon
            back_btn_font = None
            back_btn_padx = 5
            back_btn_pady = 5
        back_btn = tk.Button(header_frame, text=back_btn_text,
                            image=back_btn_image,
                            command=self.back_to_start,
                            bg="white", fg="white",
                            font=back_btn_font,
                            bd=0, activebackground="#3C98F4",
                            activeforeground="white", relief="flat", padx=back_btn_padx, pady=back_btn_pady)

        if back_btn_image:
            back_btn.image = back_btn_image
        back_btn.pack(side=tk.LEFT)
        self.back_btn = back_btn

        try:
            img = Image.open(resource_path(r"Images/registered.png")).resize((42, 42))
            self.register_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.register_icon = None
        self.title_label = tk.Label(header_frame, image=self.register_icon,
                                   compound=tk.LEFT, text=" Registered Users",
                                   font=("Segoe UI", 16, "bold"),
                                   fg=COLORS["primary"], bg=COLORS["card"])
        self.title_label.pack(side=tk.LEFT, padx=20)

        try:
            img = Image.open(resource_path(r"Images/refresh.png")).resize((42, 42))
            self.refresh_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.refresh_icon = None
        self.refresh_label = tk.Label(
            header_frame,
            text=" Refresh",
            image=self.refresh_icon,
            compound=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=COLORS["secondary"],
            padx=10,
            pady=5,
            cursor="hand2"
        )
        self.refresh_label.pack(side=tk.RIGHT)
        self.refresh_label.bind("<ButtonPress-1>", self._on_refresh_press)
        self.refresh_label.bind("<ButtonRelease-1>", self._on_refresh_release)
        self.refresh_label.bind("<Enter>", self._on_refresh_enter)
        self.refresh_label.bind("<Leave>", self._on_refresh_leave)
        self.status_frame = tk.Frame(self.main_container, bg=COLORS["light"])
        self.status_frame.pack(fill="x", pady=(0, 10))
        self.status_label = tk.Label(self.status_frame,
                                    text="Ready",
                                    font=("Segoe UI", 10),
                                    fg="#0066ff", bg=COLORS["light"])
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(self.status_frame, mode='determinate',
                                           length=200, style="Custom.Horizontal.TProgressbar")
        self.progress_label = tk.Label(self.status_frame, text="",
                                      font=("Segoe UI", 9),
                                      fg=COLORS["primary"], bg=COLORS["light"])
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Custom.Horizontal.TProgressbar",
                       thickness=10,
                       troughcolor=COLORS["light"],
                       bordercolor=COLORS["border"],
                       lightcolor=COLORS["accent"],
                       darkcolor=COLORS["accent"],
                       background=COLORS["accent"])
        self.hide_progress()
        self.stats_label = tk.Label(self.status_frame,
                                    text="Loading users...",
                                    font=("Segoe UI", 10),
                                    fg=COLORS["secondary"], bg=COLORS["light"])
        self.stats_label.pack(side=tk.RIGHT, padx=10, pady=5)
        self.capture_progress_frame = tk.Frame(self.status_frame, bg=COLORS["light"])
        self.capture_progress_frame.pack_forget()
        self.capture_progress_bar = ttk.Progressbar(self.capture_progress_frame,
                                                   mode='determinate',
                                                   length=100,
                                                   style="Capture.Horizontal.TProgressbar")
        self.capture_progress_bar.pack(side=tk.LEFT,padx=10)
        self.capture_progress_label = tk.Label(self.capture_progress_frame,
                                              text="0 / 100 images",
                                              font=("Segoe UI", 10, "bold"),
                                              fg=COLORS["accent"],
                                              bg=COLORS["light"])
        self.capture_progress_label.pack(side=tk.LEFT)
        style.configure("Capture.Horizontal.TProgressbar",
                       thickness=12,
                       troughcolor=COLORS["light"],
                       bordercolor=COLORS["accent"],
                       lightcolor=COLORS["accent"],
                       darkcolor=COLORS["accent_dark"],
                       background=COLORS["accent"])
        self.tree_container = tk.Frame(self.main_container, bg=COLORS["card"])
        self.camera_container = tk.Frame(self.main_container, bg=COLORS["card"])
        self.camera_main_frame = tk.Frame(self.camera_container, bg=COLORS["card"])
        self.camera_control_frame = tk.Frame(self.main_container, bg=COLORS["card"])

        try:
            img = Image.open(resource_path(r"Images/camera.png")).resize((42, 42))
            self.camera_icon = ImageTk.PhotoImage(img)

        except Exception as e:
            print("Image Load Error:", e)
            self.camera_icon = None
        self.open_camera_btn = tk.Label(self.camera_control_frame,
                                       image=self.camera_icon,
                                       text=" Open Camera",
                                       bg=COLORS["accent"], fg="white",
                                       compound=tk.LEFT,
                                       font=("Segoe UI", 10, "bold"),
                                       relief="flat", padx=15, pady=8,
                                       cursor="hand2")
        self.open_camera_btn.pack(side=tk.LEFT, padx=5)
        self.open_camera_btn.bind("<Enter>", self._on_camera_btn_enter)
        self.open_camera_btn.bind("<Leave>", self._on_camera_btn_leave)
        self.open_camera_btn.bind("<ButtonPress-1>", self._on_camera_btn_press)
        self.open_camera_btn.bind("<ButtonRelease-1>", lambda e: self.open_camera())
        self.cancel_capture_btn = tk.Button(self.camera_control_frame, text="Cancel",
                                           command=self.cancel_capture,
                                           bg=COLORS["danger"], fg="white",
                                           font=("Segoe UI", 10, "bold"),
                                           bd=0,
                                           activebackground="#D35400",
                                           activeforeground="white",
                                           relief="flat", padx=15, pady=8)
        self.cancel_capture_btn.pack(side=tk.LEFT, padx=5)
        self.camera_instructions = tk.Label(self.camera_container,
                                          text="• Position your face in the frame\t\t\t• Ensure good lighting\n• Move slowly for varied angles\t\t\t• Capture 100+ images for best results",
                                          font=("arial", 12),
                                          fg="#0338F9",
                                          bg=COLORS["card"],
                                          justify=tk.LEFT)
        self.preview_label = tk.Label(self.camera_container, bg="white")
        self.actions_frame = tk.Frame(self.main_container, bg=COLORS["card"])
        self.recapture_btn = tk.Button(self.actions_frame, text="Recapture",
                                       command=self.recapture_selected,
                                       bg="#0F05C8", fg="white",
                                       font=("Segoe UI", 10, "bold"),
                                       relief="flat", padx=15, pady=8,
                                       bd=0,
                                       activebackground="#3F66F4",
                                       activeforeground="white",
                                       state=tk.DISABLED)
        self.recapture_btn.pack(side=tk.LEFT, padx=5)
        self.train_btn = tk.Button(self.actions_frame, text="Train",
                                   command=self.train_selected,
                                   bg="#08F38D", fg="black",
                                   font=("Segoe UI", 10, "bold"),
                                   relief="flat", padx=15, pady=8,
                                   bd=0,
                                   activebackground="#0DCC13",
                                   activeforeground="black",
                                   state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        self.delete_btn = tk.Button(self.actions_frame, text="Delete",
                                    command=self.delete_selected,
                                    bg=COLORS["danger"], fg="white",
                                    font=("Segoe UI", 10, "bold"),
                                    relief="flat", padx=15, pady=8,
                                    bd=0,
                                    activebackground="#EE172D",
                                    activeforeground="white",
                                    state=tk.DISABLED)
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        self.cap = None
        self.running = False
        self.capturing = False
        self.captured_count = 0
        self.max_images = 100
        self.show_treeview()
        self.refresh_users()

    def _on_refresh_press(self, event):
        self.refresh_label.config(bg="#08529D")

    def _on_refresh_release(self, event):
        self.refresh_label.config(bg="#0761E8")
        self.refresh_users()

    def _on_refresh_enter(self, event):
        self.refresh_label.config(bg="#0761E8")

    def _on_refresh_leave(self, event):
        self.refresh_label.config(bg=COLORS["secondary"])

    def _on_camera_btn_enter(self, event):

        if self.cap is None or not self.capturing:
            self.open_camera_btn.config(bg="#1C8C72")  # Darker teal on hover

    def _on_camera_btn_leave(self, event):

        if self.cap is None or not self.capturing:
            self.open_camera_btn.config(bg=COLORS["accent"])

    def _on_camera_btn_press(self, event):

        if self.cap is None or not self.capturing:
            self.open_camera_btn.config(bg="#148F77")  # Even darker on press

    def show_treeview(self):
        self.show_refresh()
        self.is_capturing_mode = False
        self.camera_container.pack_forget()
        self.camera_control_frame.pack_forget()
        self.capture_progress_frame.pack_forget()
        self.tree_container.pack(fill="both", expand=True)
        self.actions_frame.pack(fill="x", pady=(10, 0))
        self.title_label.config(text=" Registered Users")
        self.back_btn.config(command=self.back_to_start)

        if not hasattr(self, 'tree'):
            self._create_treeview()

    def show_camera_mode(self, username):
        self.hide_refresh()
        self.is_capturing_mode = True
        self.current_capturing_user = username
        self.tree_container.pack_forget()
        self.actions_frame.pack_forget()
        self.title_label.config(text=f" Recapturing: {username}")
        self.back_btn.config(command=self.back_to_treeview)
        self.camera_container.pack(fill="both", expand=True, pady=(10, 20))
        self.camera_instructions.pack(anchor="w", pady=10)
        self.camera_main_frame.pack(fill="both", expand=True)
        self.preview_label.pack(
            in_=self.camera_main_frame,
            side=tk.LEFT,
            padx=20,
            pady=10
        )
        self.camera_control_frame.pack(
            in_=self.camera_main_frame,
            side=tk.RIGHT,
            padx=20,
            pady=10,
            anchor="n"
        )
        self.open_camera_btn.config(
            text=" Open Camera",
            bg=COLORS["accent"]
        )
        self.dataset_dir = get_user_dataset_dir(username)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.captured_count = 0

    def _create_treeview(self):
        tree_scroll = ttk.Scrollbar(self.tree_container, style="Vertical.TScrollbar")
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        columns = ("User Name", "Created Date", "Created Time", "Images", "Status")
        self.tree = ttk.Treeview(self.tree_container,
                                 yscrollcommand=tree_scroll.set,
                                 columns=columns,
                                 show="headings",
                                 selectmode="browse")
        tree_scroll.config(command=self.tree.yview)
        self.tree.tag_configure("evenrow", background="#d5ecf6")
        self.tree.tag_configure("oddrow", background="#fbf9e3")
        self.tree.heading("User Name", text="User Name")
        self.tree.heading("Created Date", text="Created Date")
        self.tree.heading("Created Time", text="Created Time")
        self.tree.heading("Images", text="Images")
        self.tree.heading("Status", text="Status")
        self.tree.column("User Name", width=200, minwidth=200, anchor=tk.CENTER)
        self.tree.column("Created Date", width=150, minwidth=200, anchor=tk.CENTER)
        self.tree.column("Created Time", width=120, minwidth=200, anchor=tk.CENTER)
        self.tree.column("Images", width=80, minwidth=200, anchor=tk.CENTER)
        self.tree.column("Status", width=100, minwidth=200, anchor=tk.CENTER)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background=COLORS["card"],
                        foreground=COLORS["primary"],
                        rowheight=40,
                        fieldbackground=COLORS["card"],
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading",
                        background=COLORS["primary"],
                        foreground="white",
                        font=("Segoe UI", 10, "bold"),
                        relief="flat",
                        padding=10)
        style.map("Treeview.Heading", background=[("active", "#2C7FFF")])
        style.map("Treeview", background=[("selected", COLORS["accent"])])
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def hide_progress(self):
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()
        self.status_label.config(text="Ready")

    def show_progress(self):
        self.progress_label.pack(side=tk.RIGHT, padx=(10, 0), pady=5)
        self.progress_bar.pack(side=tk.RIGHT, padx=(5, 10), pady=5)
        self.progress_bar["value"] = 0

    def update_progress(self, value, message=""):
        self.progress_bar["value"] = value
        self.progress_label.config(text=f"{value}%")
        self.status_label.config(text=message)
        self.update_idletasks()

    def update_capture_progress(self, count=None):

        if count is not None:
            self.captured_count = count
        progress = (self.captured_count / self.max_images) * 100
        self.capture_progress_bar["value"] = progress
        self.capture_progress_label.config(
            text=f"{self.captured_count} / {self.max_images} images"
        )
        self.update_idletasks()

    def open_camera(self):
        self.capture_progress_frame.pack(side=tk.RIGHT,fill="x", pady=(0, 10))

        if self.cap is not None:
            self.capturing = not self.capturing

            if self.capturing:
                self.open_camera_btn.config(
                    text=" ⏸ Pause Capture",
                    bg="#1C8C72"
                )
                self.status_label.config(text="Capturing... Move your face slowly")
            else:
                self.open_camera_btn.config(
                    text=" ▶ Resume Capture",
                    bg=COLORS["accent"]
                )
                self.status_label.config(text="Capture paused")
            return

        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                raise RuntimeError("Cannot open webcam")
            self.running = True
            self.capturing = True
            self.open_camera_btn.config(
                text=" ⏸ Pause Capture",
                bg="#1C8C72"
            )
            self.status_label.config(text="Capturing... Move your face slowly")
            self.schedule_frame_update()

        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.cap = None

    def schedule_frame_update(self):

        if self.frame_job is None:
            self.frame_job = self.after(10, self.update_frame)

    def stop_camera(self):
        self.running = False
        self.capturing = False

        if self.frame_job:
            self.after_cancel(self.frame_job)
            self.frame_job = None

        if self.cap:
            self.cap.release()
            self.cap = None
        self.preview_label.config(image="")
        self.preview_label.imgtk = None
        self.preview_label.pack_forget()
        self.open_camera_btn.config(
            text=" Open Camera",
            bg=COLORS["accent"]
        )
        self.captured_count = 0
        self.frame_count = 0
        self.capture_progress_frame.pack_forget()

    def update_frame(self):
        self.frame_job = None

        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()

        if not ok:
            self.schedule_frame_update()
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if self.capturing and self.captured_count < self.max_images:
                self.frame_count += 1

                if self.frame_count % 3 == 0:
                    crop = frame[y:y+h, x:x+w]
                    path = os.path.join(
                        self.dataset_dir,
                        f"{self.captured_count}_{self.current_capturing_user}.jpg"
                    )
                    cv2.imwrite(path, crop)
                    self.captured_count += 1
                    self.update_capture_progress()

                    if self.captured_count >= self.max_images:
                        self.capturing = False
                        self.running = False
                        self.open_camera_btn.config(
                            text=" ✅ Capture Complete",
                            bg=COLORS["success"]
                        )
                        self.status_label.config(text="✅ Capture finished!")
                        self.after(0, self._on_capture_complete)
                        break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)
        self.schedule_frame_update()

    def _on_capture_complete(self):
        self.stop_camera()
        messagebox.showinfo(
            "Capture Complete",
            f"Successfully captured {self.max_images} images for {self.current_capturing_user}"
        )
        self.show_refresh()
        self.show_treeview()
        self.refresh_users()

    def back_to_treeview(self):
        self.stop_camera()
        self.show_treeview()
        self.refresh_users()

    def back_to_start(self):
        self.stop_camera()
        self.show_refresh()
        self.controller.show_frame("StartPage")

    def cancel_capture(self):
        self.stop_camera()
        self.show_refresh()
        self.show_treeview()
        self.refresh_users()
        self.status_label.config(text="Ready")

    def on_tree_select(self, event):
        selected = self.tree.selection()

        if selected:
            item = self.tree.item(selected[0])
            user_data = item['values']
            username = user_data[0]
            status = user_data[4]
            self.recapture_btn.config(state=tk.NORMAL)

            if status == "✅ Trained":
                self.train_btn.config(state=tk.DISABLED)
            else:
                self.train_btn.config(state=tk.NORMAL)
            self.delete_btn.config(state=tk.NORMAL)
        else:
            self.recapture_btn.config(state=tk.DISABLED)
            self.train_btn.config(state=tk.DISABLED)
            self.delete_btn.config(state=tk.DISABLED)

    def refresh_users(self):

        if hasattr(self, 'tree'):
            for item in self.tree.get_children():
                self.tree.delete(item)
        users = get_user_info()

        if not users:

            if hasattr(self, 'tree'):
                self.tree.insert("", "end", values=("No users found", "", "", "", ""))
            self.stats_label.config(text="No registered users found")
            return
        total_users = len(users)
        trained_users = sum(1 for user in users if user['model_exists'])
        self.stats_label.config(text=f"Total Users: {total_users} | Trained: {trained_users} | Untrained: {total_users - trained_users}")

        if hasattr(self, 'tree'):
            for user in users:
                created_date = user['created'].strftime("%Y-%m-%d")
                created_time = user['created'].strftime("%H:%M:%S")
                image_text = f"{user['image_count']} images"

                if user['model_exists']:
                    status_text = "✅ Trained"
                else:
                    status_text = "⚠️ Untrained"
                row_tag = "evenrow" if len(self.tree.get_children()) % 2 == 0 else "oddrow"
                self.tree.insert("", "end",
                                values=(user['name'], created_date, created_time, image_text, status_text),
                                tags=(row_tag,))

    def get_selected_user(self):

        if hasattr(self, 'tree'):
            selected = self.tree.selection()

            if selected:
                item = self.tree.item(selected[0])
                return item['values'][0]
        return None

    def recapture_selected(self):
        username = self.get_selected_user()

        if not username:
            return

        if messagebox.askyesno("Confirm Recapture",
                             f"Recapture images for user '{username}'?\n\n"
                             f"⚠️  WARNING: This will delete:\n"
                             f"1. All existing captured images\n"
                             f"2. Existing trained model\n\n"
                             f"Start fresh with new capture session?"):

            classifier_path = os.path.join(get_classifiers_dir(), f"{username}_classifier.xml")
            dataset_path = get_user_dataset_dir(username)

            try:

                if os.path.exists(classifier_path):
                    os.remove(classifier_path)

                if os.path.exists(dataset_path):
                    import shutil
                    shutil.rmtree(dataset_path)
                    os.makedirs(dataset_path, exist_ok=True)

            except Exception as e:
                print(f"Error during cleanup: {e}")
                messagebox.showwarning("Cleanup Warning",
                                    f"Some cleanup failed: {e}\nProceeding anyway...")
            self.show_camera_mode(username)

    def train_selected(self):
        username = self.get_selected_user()

        if not username:
            return
        dataset_path = get_user_dataset_dir(username)

        if not os.path.exists(dataset_path):
            messagebox.showerror("Error", "No dataset found. Capture images first.")
            return
        cnt = len([f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg','.png','.jpeg'))])

        if cnt < 50:

            if not messagebox.askyesno("Few Images",
                                      f"Only {cnt} images found (recommended: 100+).\nContinue training anyway?"):
                return
        self.recapture_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.delete_btn.config(state=tk.DISABLED)
        self.show_progress()
        self.update_progress(0, "Starting training...")

        def training_thread():

            try:

                def update_progress_callback(progress, message):
                    self.after(0, lambda: self.update_progress(progress, message))
                success, message = train_classifier_with_progress(username, update_progress_callback)
                self.after(0, lambda: self.on_training_complete(success, message, username))

            except Exception as e:
                self.after(0, lambda: self.on_training_complete(False, f"Training error: {str(e)}", username))
        threading.Thread(target=training_thread, daemon=True).start()

    def on_training_complete(self, success, message, username):
        self.hide_progress()
        self.recapture_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
        self.delete_btn.config(state=tk.NORMAL)

        if success:
            messagebox.showinfo("Success", f"✅ Model trained successfully for {username}!")
        else:
            messagebox.showerror("Error", f"Training failed: {message}")
        self.refresh_users()

        if hasattr(self, 'tree'):
            for item in self.tree.get_children():

                if self.tree.item(item)['values'][0] == username:
                    self.tree.selection_set(item)
                    break

    def delete_selected(self):
        username = self.get_selected_user()

        if not username or username == "No users found":
            return

        if not messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete user '{username}'?\n\n"
            "This will permanently delete:\n"
            "  • All captured images in {dataset_path}\n"
            "  • Trained classifier file {classifier_path} (if present)\n\n"
            "This action cannot be undone."
        ):
            return

        try:
            import shutil
            dataset_path = get_user_dataset_dir(username)

            classifier_path = os.path.join(get_classifiers_dir(), f"{username}_classifier.xml")

            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)

            if os.path.exists(classifier_path):

                try:
                    os.remove(classifier_path)

                except Exception as e:
                    messagebox.showwarning("Warning", f"Removed dataset but failed to remove classifier: {e}")
            self.refresh_users()

            try:
                for btn in (self.recapture_btn, self.train_btn, self.delete_btn):
                    btn.config(state=tk.DISABLED)

            except Exception:
                pass

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete user '{username}': {e}")
            print("delete_selected error:", e)

    def hide_refresh(self):
        self.refresh_label.pack_forget()

    def show_refresh(self):

        if not self.refresh_label.winfo_ismapped():
            self.refresh_label.pack(side=tk.RIGHT)

if __name__ == "__main__":
    create_data_directories()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(2)

    except:
        pass
    root = tk.Tk()
    root.withdraw()

    def start_main_app():

        if sys.platform == "win32":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "com.sourav.FaceRecognitionPro.ModelDownloader"
            )
        root.destroy()
        app = MainUI()
        app.mainloop()

    if all_models_exist():
        start_main_app()
    else:

        if sys.platform == "win32":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "com.sourav.FaceRecognitionPro.ModelDownloader"
            )
        ModelDownloader(root, on_complete=start_main_app)
        root.mainloop()
