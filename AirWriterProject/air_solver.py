import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from PIL import Image
import io
import datetime # For unique filenames

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set GOOGLE_API_KEY in your .env file.")

genai.configure(api_key=API_KEY)

# --- Face Detection Setup ---
face_cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path):
    print(f"Error: Haar Cascade file not found at {face_cascade_path}")
    print("Download it from OpenCV's GitHub repository.")
    # Optionally exit, or continue without face detection
    face_cascade = None
else:
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

# --- MediaPipe Hand Tracking Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# --- Drawing, Erasing, and Color Setup ---
drawing_canvas = None
eraser_color = (0, 0, 0) # Black

colors = { # BGR format
    'white': (255, 255, 255), 'red': (0, 0, 255), 'green': (0, 255, 0),
    'blue':  (255, 0, 0), 'yellow':(0, 255, 255),
}
color_keys = list(colors.keys())
current_color_index = 0
current_drawing_color = colors[color_keys[current_color_index]]

# Pen Thickness
current_pen_thickness = 5
min_pen_thickness = 1
max_pen_thickness = 50
pen_step = 1

# Eraser settings
current_eraser_thickness = 40
min_eraser_thickness = 10
max_eraser_thickness = 100
eraser_step = 5

# --- Smoothing Setup ---
smoothing_factor = 0.6
smoothed_point = None

# --- Mode Management & Hysteresis ---
MODE_PAUSE = 0
MODE_WRITE = 1
MODE_ERASE = 2
current_mode = MODE_PAUSE
detected_mode_this_frame = MODE_PAUSE
mode_switch_candidate = -1
mode_stable_count = 0
MODE_SWITCH_THRESHOLD = 3

last_point_for_line = None

# --- Feature Flags ---
live_blur_enabled = False
show_overlay_enabled = True

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
drawing_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# --- Gemini Setup ---
model = genai.GenerativeModel('gemini-1.5-flash')

# --- State Variables ---
last_gemini_response = ""
status_text = "Initializing..."
last_key_press_time = 0
key_debounce_time = 0.2 # Debounce for keyboard controls

# --- Finger Landmark IDs ---
tip_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
pip_ids = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]

# --- Helper Functions ---
def count_extended_fingers(hand_landmarks):
    VERTICAL_THRESHOLD = -0.04 # Adjust if needed
    if not hand_landmarks: return 0
    landmarks = hand_landmarks.landmark
    count = 0
    for tip_id, pip_id in zip(tip_ids, pip_ids):
        if landmarks[tip_id].y < landmarks[pip_id].y + VERTICAL_THRESHOLD:
            count += 1
    return count

def blur_faces(image, face_cascade_detector):
    """Detects faces and applies Gaussian blur."""
    if face_cascade_detector is None:
        return image # Return original if detector not loaded

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        # Apply Gaussian blur - kernel size depends on face size, make it odd
        kernel_width = (w // 7) | 1 # Make kernel width odd and proportional
        kernel_height = (h // 7) | 1
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_width, kernel_height), 30)
        # Put blurred face back into the image
        image[y:y+h, x:x+w] = blurred_face
    return image

# --- Main Loop ---
while True:
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)

    # --- Optional Live Blurring ---
    if live_blur_enabled and face_cascade:
        frame = blur_faces(frame, face_cascade)

    # --- Hand Processing ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True
    # Keep frame in BGR for OpenCV drawing
    # frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Not needed if we start with BGR

    current_time = time.time()
    raw_index_tip_pos = None
    num_fingers = 0
    detected_mode_this_frame = MODE_PAUSE

    # --- Hand Detection & Raw Data ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        num_fingers = count_extended_fingers(hand_landmarks)
        tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        cx, cy = int(tip.x * frame_width), int(tip.y * frame_height)
        raw_index_tip_pos = (cx, cy)

        # Determine Mode Detected THIS FRAME
        if num_fingers == 1: detected_mode_this_frame = MODE_WRITE
        elif num_fingers == 3: detected_mode_this_frame = MODE_ERASE
        else: detected_mode_this_frame = MODE_PAUSE

        # Apply EMA Smoothing
        if raw_index_tip_pos:
            if smoothed_point is None: smoothed_point = raw_index_tip_pos
            else:
                sx = int(smoothed_point[0] * (1 - smoothing_factor) + raw_index_tip_pos[0] * smoothing_factor)
                sy = int(smoothed_point[1] * (1 - smoothing_factor) + raw_index_tip_pos[1] * smoothing_factor)
                smoothed_point = (sx, sy)
        else: smoothed_point = None
    else:
        detected_mode_this_frame = MODE_PAUSE
        smoothed_point = None

    # --- Mode Switching Logic with Hysteresis ---
    if detected_mode_this_frame != current_mode:
        if detected_mode_this_frame == mode_switch_candidate:
            mode_stable_count += 1
        else:
            mode_switch_candidate = detected_mode_this_frame
            mode_stable_count = 1
    else: # Mode is stable or same as current
        mode_stable_count = 0
        mode_switch_candidate = -1

    if mode_switch_candidate != -1 and mode_stable_count >= MODE_SWITCH_THRESHOLD:
        print(f"Switching Mode: {current_mode} -> {mode_switch_candidate}")
        current_mode = mode_switch_candidate
        mode_stable_count = 0
        mode_switch_candidate = -1
        last_point_for_line = None # Reset line on any mode switch

    if not results.multi_hand_landmarks and current_mode != MODE_PAUSE:
        # print("Hand lost, forcing PAUSE") # Can be noisy, disable if needed
        current_mode = MODE_PAUSE
        mode_stable_count = 0
        mode_switch_candidate = -1
        last_point_for_line = None

    # --- Perform Action Based on STABLE Current Mode ---
    indicator_pos = smoothed_point
    action_performed_this_frame = False

    if indicator_pos:
        if current_mode == MODE_WRITE:
            status_text = f"MODE: Writing ({color_keys[current_color_index].upper()})"
            if last_point_for_line is not None:
                 cv2.line(drawing_canvas, last_point_for_line, indicator_pos, current_drawing_color, current_pen_thickness)
                 action_performed_this_frame = True
            last_point_for_line = indicator_pos
            cv2.circle(frame, indicator_pos, current_pen_thickness + 3, current_drawing_color, 2)

        elif current_mode == MODE_ERASE:
            status_text = f"MODE: Erasing (Size: {current_eraser_thickness})"
            cv2.circle(drawing_canvas, indicator_pos, current_eraser_thickness // 2, eraser_color, -1)
            last_point_for_line = None
            action_performed_this_frame = True
            cv2.circle(frame, indicator_pos, current_eraser_thickness // 2, (0, 0, 255), 2)

        else: # MODE_PAUSE
             status_text = "MODE: Paused"
             last_point_for_line = None
             cv2.circle(frame, indicator_pos, 10, (255, 150, 0), 2)
    else:
        status_text = "No hand detected"
        if current_mode != MODE_PAUSE: last_point_for_line = None

    # --- Combine frame and canvas (conditional overlay) ---
    if show_overlay_enabled:
        gray_canvas = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
        _, drawing_mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(drawing_mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        drawing_fg = cv2.bitwise_and(drawing_canvas, drawing_canvas, mask=drawing_mask)
        display_frame = cv2.add(frame_bg, drawing_fg)
    else:
        display_frame = frame # Show only the camera feed

    # --- Display Status and Info ---
    top_margin = 10
    line_height = 20
    font_scale_s = 0.5
    font_scale_m = 0.6

    # Line 1: Status
    cv2.putText(display_frame, status_text, (top_margin, top_margin + line_height*1), cv2.FONT_HERSHEY_SIMPLEX, font_scale_m, (50, 200, 255), 2)
    # Line 2: Color/Pen/Eraser
    tool_info = f"Clr:{color_keys[current_color_index][0].upper()} Pen:[{current_pen_thickness}] Ers:[{current_eraser_thickness}]"
    cv2.putText(display_frame, tool_info, (top_margin, top_margin + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, font_scale_s, (255, 255, 255), 1)
    # Line 3: Toggles
    toggle_info = f"Blur:[{'ON' if live_blur_enabled else 'OFF'}] Overlay:[{'ON' if show_overlay_enabled else 'OFF'}]"
    cv2.putText(display_frame, toggle_info, (top_margin, top_margin + line_height*3), cv2.FONT_HERSHEY_SIMPLEX, font_scale_s, (255, 255, 255), 1)

    # Bottom Controls Hint
    controls_text = "'S':Solve 'C':Clear 'W':Save 'B':Blur 'T':Overlay '[ ]':Pen '+-':Erase 1-5:Color 'Q':Quit"
    text_size, _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(display_frame, controls_text, (frame_width - text_size[0] - 10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Display Gemini Response (drawing upwards from bottom)
    y0, dy = frame_height - 40, 18
    resp_lines = last_gemini_response.split('\n')
    for i, line in enumerate(reversed(resp_lines)):
        y = y0 - i * dy
        if y > top_margin + line_height*4: # Prevent overlap with top status
             cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('Air Writing Solver', display_frame)

    # --- Keyboard Input Handling ---
    key = cv2.waitKey(1) & 0xFF
    allow_key = (current_time - last_key_press_time) > key_debounce_time

    if key != 255 and allow_key:
        last_key_press_time = current_time

        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'): # Solve with Gemini
            if np.sum(drawing_canvas) < 500:
                 print("Canvas seems empty. Draw something first!")
                 last_gemini_response = "Draw something first!"
            else:
                print("Preparing image for Gemini (with face blurring)...")
                status_text = "Sending to Gemini..."
                last_point_for_line = None # Ensure pen is lifted

                # Create the image to send (Combine current frame + canvas)
                # Make sure it reflects the *drawing*, overlay optional for context
                gray_canvas = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
                _, drawing_mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
                inv_mask = cv2.bitwise_not(drawing_mask)
                frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask) # Use latest frame
                drawing_fg = cv2.bitwise_and(drawing_canvas, drawing_canvas, mask=drawing_mask)
                send_frame = cv2.add(frame_bg, drawing_fg)

                # --- Apply Face Blurring before sending ---
                send_frame_blurred = blur_faces(send_frame.copy(), face_cascade)
                print("Face blurring applied for sending.")

                # Prepare image bytes
                img_rgb = cv2.cvtColor(send_frame_blurred, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                prompt_text = f"Analyze the drawing (dominant color: {color_keys[current_color_index]}) on this image. Treat it as a math equation or a question and provide the answer or solution. Ignore the hand and any blurred areas (faces)."

                try:
                    print("Sending to Gemini API...")
                    response = model.generate_content([prompt_text, {'mime_type': 'image/jpeg', 'data': img_bytes}])
                    if response.parts:
                        last_gemini_response = response.text.strip()
                        print("Gemini Response Received.")
                    else:
                        candidate = response.candidates[0] if response.candidates else None
                        finish_reason = candidate.finish_reason if candidate else "UNKNOWN"
                        last_gemini_response = f"Gemini: No text (Reason: {finish_reason})."
                        print(last_gemini_response, "Safety:", candidate.safety_ratings if candidate else [])
                except Exception as e:
                    print(f"Error calling Gemini API: {e}")
                    last_gemini_response = f"Error: {e}"

        # --- Feature Keys ---
        elif key == ord('c'): # Clear Canvas
            drawing_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            last_point_for_line = None
            last_gemini_response = "" # Clear old response
            print("Canvas Cleared.")
        elif key == ord('w'): # Write/Save Drawing
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"air_drawing_{timestamp}.png"
            cv2.imwrite(filename, drawing_canvas)
            print(f"Drawing saved as {filename}")
            last_gemini_response = f"Saved as {filename}" # Show feedback
        elif key == ord('b'): # Toggle Live Blur
            live_blur_enabled = not live_blur_enabled
            print(f"Live Face Blurring: {'ON' if live_blur_enabled else 'OFF'}")
        elif key == ord('t'): # Toggle Overlay
            show_overlay_enabled = not show_overlay_enabled
            print(f"Drawing Overlay: {'ON' if show_overlay_enabled else 'OFF'}")
        elif key == ord('['): # Decrease Pen Thickness
            current_pen_thickness = max(min_pen_thickness, current_pen_thickness - pen_step)
            print(f"Pen thickness decreased to: {current_pen_thickness}")
        elif key == ord(']'): # Increase Pen Thickness
            current_pen_thickness = min(max_pen_thickness, current_pen_thickness + pen_step)
            print(f"Pen thickness increased to: {current_pen_thickness}")
        elif key == ord('+') or key == ord('='): # Increase Eraser
            current_eraser_thickness = min(max_eraser_thickness, current_eraser_thickness + eraser_step)
            print(f"Eraser size increased to: {current_eraser_thickness}")
        elif key == ord('-'): # Decrease Eraser
            current_eraser_thickness = max(min_eraser_thickness, current_eraser_thickness - eraser_step)
            print(f"Eraser size decreased to: {current_eraser_thickness}")
        elif ord('1') <= key <= ord(str(len(color_keys))): # Change Color
             new_color_index = key - ord('1')
             if new_color_index < len(color_keys):
                 current_color_index = new_color_index
                 current_drawing_color = colors[color_keys[current_color_index]]
                 print(f"Color changed to: {color_keys[current_color_index]}")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()