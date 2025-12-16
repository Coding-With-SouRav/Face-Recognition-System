# Demo Images

<img width="1217" height="860" alt="Screenshot 2025-12-16 205212" src="https://github.com/user-attachments/assets/181f5775-6233-4dd1-afd2-4a36a79196a8" />
<img width="1222" height="864" alt="Screenshot 2025-12-16 205228" src="https://github.com/user-attachments/assets/75017fe9-b655-4aa5-b317-d655492ad54e" />
<img width="1215" height="851" alt="Screenshot 2025-12-16 205248" src="https://github.com/user-attachments/assets/5213c734-1ec9-4a48-beb9-63008d5750fb" />
<img width="1217" height="854" alt="Screenshot 2025-12-16 205305" src="https://github.com/user-attachments/assets/3b369831-c178-49df-b5c1-77424ceb08cd" />
<img width="1214" height="854" alt="Screenshot 2025-12-16 205338" src="https://github.com/user-attachments/assets/bbc667ad-dc5b-4fec-b75d-3ed4b6492484" />



## 🎯 **Features Overview**

### **Core Face Recognition Features**
- **Multi-User Face Registration**: Capture and train models for multiple users
- **Real-Time Face Detection**: Live camera feed with face detection and recognition
- **High-Precision Recognition**: Uses LBPH (Local Binary Patterns Histograms) algorithm
- **Multi-User Recognition**: Recognize all trained users simultaneously
- **Face Dataset Management**: Capture 100+ images per user for training

### **Advanced Recognition Features**
- **Age Detection**: Categorizes age into 8 groups (0-2 to 60-100)
- **Gender Detection**: Classifies as Male or Female
- **Emotion Recognition**: Detects 7 emotions (Angry, Disgust, Scared, Happy, Sad, Surprised, Normal)
- **Combined Recognition**: Simultaneous face recognition + age/gender/emotion detection

### **User Management**
- **User Registration**: Create new user profiles with unique names
- **Dataset Management**: View, recapture, and delete user datasets
- **Training Status**: Track which users have trained models
- **Bulk Operations**: Train, recapture, or delete multiple users

### **Technical Features**
- **Auto Model Download**: Automatically downloads required AI models on first run
- **Progress Tracking**: Real-time progress bars for training and capture
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Cross-Platform Support**: Windows, macOS, and Linux compatible
- **Local Storage**: All data stored locally for privacy

## 🖥️ **UI Elements & Interface**

### **Main Interface Components**

1. **Start Page**
   - Application title and logo
   - Navigation buttons for main features
   - System status indicators

2. **User Registration Page**
   - Username input field
   - Validation for existing users
   - Next step navigation

3. **Capture Page**
   - Live camera preview (640x480)
   - Capture progress bar (0-100 images)
   - Start/Pause/Resume controls
   - Training initiation button

4. **Advanced Recognition Page**
   - Feature selection checkboxes (Gender/Age, Emotion, Face Recognition)
   - Model loading controls
   - Live recognition display with overlay information
   - Real-time bounding boxes and labels

5. **User Management Page**
   - Interactive table with user information:
     - Username
     - Creation date/time
     - Number of captured images
     - Training status (✅ Trained / ⚠️ Untrained)
   - Action buttons: Recapture, Train, Delete
   - Refresh functionality

6. **Camera Frame Components**
   - Multi-mode camera interface (Capture/Recognition)
   - Face detection bounding boxes
   - Real-time confidence scores
   - Status indicators and controls

### **Visual Design**
- **Color Scheme**: Professional blue/teal theme with accent colors
- **Fonts**: Segoe UI for clean, modern typography
- **Icons**: Custom icons for navigation and features
- **Progress Indicators**: Animated loading spinners and progress bars
- **Responsive Layout**: Fixed 1200x800 window with centered content

## 🚀 **How to Run the Application**

### **Prerequisites**

1. **Python 3.7+** installed on your system
2. **Required Python Packages**:
   ```bash
   pip install opencv-python
   pip install pillow
   pip install numpy
   pip install tensorflow
   pip install keras
   pip install tkinter  # Usually comes with Python
   ```

### **Installation Steps**

1. **Save the code** to a file named `app.py`

2. **Create necessary directories**:
   - Create an `Images` folder in the same directory as `app.py`
   - Add the following image files (or update paths in code):
     - `icon.png` - Application icon
     - `back.png` - Back button icon
     - `recognition.png` - Recognition feature icon
     - `gender.png` - Gender detection icon
     - `emotion.png` - Emotion detection icon
     - `recog_face.png` - Face recognition icon
     - `writing.png` - Registration icon
     - `camera.png` - Camera icon
     - `label_camera.png` - Camera label icon
     - `registered.png` - Registered users icon
     - `refresh.png` - Refresh icon

3. **Run the application**:
   ```bash
   python app.py
   ```

### **First Run Setup**

1. **On first launch**, the application will:
   - Create necessary data directories in your system's AppData/Library folder
   - Download required AI models (approximately 50-100MB)
   - Show a download progress window

2. **Model Download Locations**:
   - Windows: `C:\Users\{username}\AppData\Roaming\FaceRecognitionPro\models\`
   - macOS: `~/Library/Application Support/FaceRecognitionPro/models/`
   - Linux: `~/.local/share/FaceRecognitionPro/models/`

### **Using the Application**

#### **Step 1: Create a New User**
1. Click "➕ New User"
2. Enter a unique username
3. Click "Next"
4. Follow on-screen instructions to capture 100+ face images
5. Click "Train Model" after capture completes

#### **Step 2: Train the Model**
1. After capturing images, the system automatically trains the model
2. Training progress is shown with a progress bar
3. You'll receive a success message when complete

#### **Step 3: Test Recognition**
1. Go to "🔍 Recognize" from the main menu
2. The system will use your webcam for real-time recognition
3. Recognized faces will be highlighted with the user's name

#### **Step 4: Advanced Features**
1. From main menu, go to "🔍 Recognize"
2. Check desired features (Gender/Age, Emotion, Face Recognition)
3. Click "Load Models" (first time only)
4. Click "Start Recognition"

### **Troubleshooting**

#### **Common Issues & Solutions**

1. **Camera Not Opening**:
   - Ensure webcam is connected and not used by other applications
   - Grant camera permissions if prompted

2. **Model Download Fails**:
   - Check internet connection
   - Manual download option: Download models from GitHub URLs in code

3. **Import Errors**:
   - Ensure all Python packages are installed correctly
   - Try: `pip install --upgrade pip`

4. **Memory Issues**:
   - Close other applications
   - The application requires ~500MB RAM when all models loaded

5. **Performance Issues**:
   - Reduce camera resolution in code if needed
   - Disable some features for better performance

### **System Requirements**

- **Minimum**:
  - 4GB RAM
  - 500MB free disk space
  - Webcam
  - Python 3.7+

- **Recommended**:
  - 8GB RAM
  - 1GB free disk space
  - Dedicated GPU (for faster model inference)
  - Good lighting conditions for face capture

### **Directory Structure After Installation**
```
AppData/
└── FaceRecognitionPro/
    ├── user_datasets/
    │   ├── User1/
    │   │   ├── 0_User1.jpg
    │   │   └── ... (100+ images)
    │   └── User2/
    ├── classifiers/
    │   ├── User1_classifier.xml
    │   └── User2_classifier.xml
    └── models/
        ├── emotion/
        ├── age/
        ├── gender/
        └── haarcascade/
```

### **Important Notes**
1. **First run may take several minutes** due to model downloads
2. **Capture at least 100 images** per user for best recognition accuracy
3. **Good lighting is crucial** for face detection and recognition
4. **Models are stored locally** - no data is sent to external servers
5. **Regular training updates** improve recognition accuracy over time

### **Development Features**
- Modular design with separate pages for each functionality
- Threading for background operations (training, camera)
- Comprehensive error handling and user feedback
- Progress tracking for all long-running operations
- Clean separation of UI and logic

The application is ready for both personal use and as a foundation for more advanced face recognition projects!
