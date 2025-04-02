# AirTouch

:: 1. Navigate to your desired parent directory (e.g., Desktop)
cd %USERPROFILE%\Desktop

:: 2. Create and enter the project directory
mkdir AirWriterProject
cd AirWriterProject

:: 3. Create virtual environment
python -m venv venv

:: 4. Activate virtual environment (IMPORTANT!)
venv\Scripts\activate

:: 5. Install dependencies
pip install opencv-python mediapipe google-generativeai python-dotenv Pillow

:: 6. Create empty files
type nul > air_solver.py
type nul > .env

:: 7. MANUALLY EDIT .env file to add: GOOGLE_API_KEY=YOUR_API_KEY_HERE

:: 8. MANUALLY EDIT air_solver.py file to paste the Python code

:: 9. Run the application (make sure venv is active)
python air_solver.py

:: 10. To stop the script, press 'Q' in the OpenCV window or Ctrl+C in CMD

:: 11. To deactivate the virtual environment when done:
:: deactivate