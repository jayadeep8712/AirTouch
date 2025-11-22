# AirWriter

**Write in the air and get the answer!**  

AirTouch is a Python-based application that lets you **draw letters or words in the air** with your hand, detects them in real-time using **Mediapipe & OpenCV**, and processes them with **Google Generative AI** to provide answers instantly.  



## Features 🌟
- **Air Writing Detection** – Track your hand movements and detect drawn letters in real-time.  
- **AI-powered Answers** – Converts your air-written input into meaningful responses using Google Generative AI.  
- **Simple Setup** – Lightweight Python app with minimal dependencies.  
- **Cross-platform Ready** – Works on any system with Python 3.10+.



## Setup 🚀

1. **Clone or navigate** to your desired directory:

```bash
mkdir AirWriterProject
cd AirWriterProject
````

2. **Create and activate virtual environment:**

```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install opencv-python mediapipe google-generativeai python-dotenv Pillow
```

4. **Create required files:**

```bash
type nul > air_solver.py
type nul > .env
```

5. **Edit `.env`** and add your Google API Key:

```env
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

6. **Paste your Python code** in `air_solver.py`.

7. **Run the application:**

```bash
python air_solver.py
```

> Press `Q` in the OpenCV window or `Ctrl+C` to stop the app.

8. **Deactivate virtual environment when done:**

```bash
deactivate
```



## How to Use ✍️

1. Raise your hand in front of the camera.
2. Draw letters or words in the air slowly and clearly.
3. Watch as the app interprets your air-writing and provides answers in real-time.



## Dependencies 📦

* [OpenCV](https://pypi.org/project/opencv-python/) – Hand tracking & drawing detection
* [Mediapipe](https://pypi.org/project/mediapipe/) – Efficient hand landmarks
* [Google Generative AI](https://pypi.org/project/google-generativeai/) – Smart response generation
* [Python-dotenv](https://pypi.org/project/python-dotenv/) – Manage API keys
* [Pillow](https://pypi.org/project/Pillow/) – Image processing



💡 **Fun Tip:** Practice drawing slowly and clearly for better recognition. The air is your canvas! ✨
