# ✨ AirWriter – Write in the Air, Get Instant Answers! ✨

**AirWriter** (aka **AirTouch**) is a Python-based application that lets you **draw letters or words in the air** with your hand. It detects your movements in real-time using **Mediapipe & OpenCV**, then leverages **Google Generative AI** to provide instant responses.

Think of it as a **magic pen… without the pen!** 🪄

<br>

## 🌟 Features

* **Air Writing Detection** – Track your hand movements and recognize drawn letters in real-time.
* **AI-powered Answers** – Convert your air-written input into meaningful responses using Google Generative AI.
* **Cross-platform & Lightweight** – Works on any system with Python 3.10+ with minimal dependencies.
* **Simple & Fun** – Interact with the app intuitively—your hand is your controller!

<br>

## 🚀 Quick Start

Follow these steps to get AirWriter up and running:

### 1️⃣ Setup Project Directory

```bash
mkdir AirWriterProject
cd AirWriterProject
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe google-generativeai python-dotenv Pillow
```

### 4️⃣ Create Required Files

```bash
type nul > air_solver.py  # Main script
type nul > .env           # Environment variables
```

### 5️⃣ Configure API Key

Add your Google API Key in `.env`:

```env
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

### 6️⃣ Paste Your Python Code

Add your logic in `air_solver.py`.

### 7️⃣ Run the App

```bash
python air_solver.py
```

> Press `Q` in the OpenCV window or `Ctrl+C` in the terminal to quit.

### 8️⃣ Deactivate Virtual Environment

```bash
deactivate
```

<br>

## ✍️ How to Use

1. Raise your hand in front of the camera.
2. Draw letters or words **slowly and clearly** in the air.
3. Watch as the app **interprets your air-writing** and provides **real-time answers**.

💡 **Pro Tip:** Clear, deliberate movements improve recognition accuracy.

<br>

## 📦 Dependencies

* [**OpenCV**](https://pypi.org/project/opencv-python/) – Hand tracking & drawing detection
* [**Mediapipe**](https://pypi.org/project/mediapipe/) – Efficient hand landmark detection
* [**Google Generative AI**](https://pypi.org/project/google-generativeai/) – Smart response generation
* [**Python-dotenv**](https://pypi.org/project/python-dotenv/) – Manage API keys securely
* [**Pillow**](https://pypi.org/project/Pillow/) – Image processing

<br>

## 🎨 Fun Tips

* Slow and deliberate movements = more accurate recognition.
* Experiment with **different lighting conditions** for best results.
* Treat the air as your **digital canvas**! ✨
