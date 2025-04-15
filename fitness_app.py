#old reference code
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import mediapipe as mp
import time
import sys

def load_model_safely(model_path):
    try:
        # First attempt - standard loading
        print(f"Attempting to load model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully with standard method")
        return model
    except Exception as e:
        print(f"Standard loading failed: {e}")
        try:
            # Second attempt - with custom object scope
            with tf.keras.utils.custom_object_scope({}):
                model = tf.keras.models.load_model(model_path, compile=False)
                print("Model loaded successfully with custom_object_scope")
                return model
        except Exception as e:
            print(f"Second attempt failed: {e}")
            # If we can't load the model, build a new one
            print("Building new model instead of loading")
            return None

#added for rep counter
class ExerciseTracker:
    def __init__(self):
        self.rep_counter = 12  # Default reps
    
    def reset_rep_counter(self):
        """Reset rep counter to 12."""
        self.rep_counter = 12
        print("Rep counter reset to 12")

# Create a global instance of ExerciseTracker to use in app.py
exercise_instance = ExerciseTracker()


class FitnessPostureCorrection:
    def __init__(self, model_path=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize exercise types first - moved up in initialization order
        self.exercise_types = ['benchpress', 'deadlift', 'gym-exercises', 
                              'jalon', 'shoulder-press', 'squat']
        
        # Initialize counters
        self.rep_counter = 20
        self.good_form_streak = 0
        self.last_form_feedback = ""
        self.model_loaded = False  # Add a flag to track if model was loaded
        
        # State tracking for rep counting
        self.in_good_form = False
        self.consecutive_good_frames = 0
        self.consecutive_bad_frames = 0
        self.rep_started = False
        
        # Debug info
        self.debug_info = {}

        # Try to load the model first
        self.model = None
        if model_path and os.path.exists(model_path):
            loaded_model = load_model_safely(model_path)
            if loaded_model is not None:
                self.model = loaded_model
                self.model_loaded = True
                print("Model loaded successfully and ready for predictions")
            else:
                print("Building new model")
                self.model = self.build_model()
        else:
            print("Model path not provided or not found. Using default untrained model.")
            self.model = self.build_model()

        # Correction feedback templates
        self.correction_templates = {
            'benchpress': {
                'good': "Good bench press form!",
                'errors': [
                    "Keep your back flat against the bench",
                    "Lower the bar to mid-chest level",
                    "Keep your feet planted firmly on the ground",
                    "Elbows should be at approximately 45° to your body"
                ]
            },
            'deadlift': {
                'good': "Good deadlift form!",
                'errors': [
                    "Keep your back straight, not rounded",
                    "Bar should remain close to your body",
                    "Start with hips lower, shoulders back",
                    "Look forward, not up or down"
                ]
            },
            'shoulder-press': {
                'good': "Good shoulder press form!",
                'errors': [
                    "Keep your core engaged and avoid arching your back",
                    "Elbows should be at 90° at the bottom position",
                    "Push the weights straight up, not forward",
                    "Keep your wrists straight, not bent"
                ]
            },
            'squat': {
                'good': "Good squat form!",
                'errors': [
                    "Keep your knees aligned with your toes",
                    "Maintain a straight back throughout the movement",
                    "Go deeper - aim for thighs parallel to the ground",
                    "Keep your weight on your heels"
                ]
            },
            'jalon': {
                'good': "Good lat pulldown form!",
                'errors': [
                    "Keep your back straight, not leaning too far back",
                    "Pull the bar down to your upper chest",
                    "Control the movement on the way up",
                    "Keep your elbows pointing down, not out"
                ]
            },
            'gym-exercises': {
                'good': "Good exercise form!",
                'errors': [
                    "Maintain proper posture",
                    "Control the movement, avoid using momentum",
                    "Focus on the muscle you're targeting",
                    "Breathe consistently through the exercise"
                ]
            }
        }

    def build_model(self):
        """Build a model based on MobileNetV2 for exercise classification and form detection"""
        # Use Input layer with input_shape instead of batch_shape
        inputs = Input(shape=(224, 224, 3))
        
        # Base model with pre-trained weights
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )

        # Freeze the base model layers
        base_model.trainable = False

        # Create the model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(len(self.exercise_types), activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def analyze_pose(self, frame, exercise_type=None):
        """Analyze pose and provide corrections based on exercise type"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)

        # If pose landmarks were detected
        if results.pose_landmarks:
            # Draw pose landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # If exercise type not provided, predict it
            if exercise_type is None and self.model is not None and self.model_loaded:
                try:
                    # Preprocess frame for the classification model
                    resized_frame = cv2.resize(rgb_frame, (224, 224))
                    preprocessed = preprocess_input(np.expand_dims(resized_frame, axis=0))

                    # Predict exercise type
                    prediction = self.model.predict(preprocessed)
                    exercise_idx = np.argmax(prediction[0])
                    exercise_type = self.exercise_types[exercise_idx]
                    confidence = prediction[0][exercise_idx]

                    # Display predicted exercise
                    cv2.putText(frame, f"Exercise: {exercise_type} ({confidence:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error during exercise prediction: {e}")
                    # If there's an error, default to a generic exercise type
                    if exercise_type is None:
                        exercise_type = 'gym-exercises'
                        cv2.putText(frame, "Exercise detection unavailable",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif exercise_type is None:
                # If model is None, default to a generic exercise type
                exercise_type = 'gym-exercises'
                cv2.putText(frame, "Exercise detection unavailable",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Display manual selected exercise
                cv2.putText(frame, f"Exercise: {exercise_type}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Analyze form based on exercise type
            feedback = self.analyze_form(results.pose_landmarks, exercise_type)

            # Display feedback
            for i, line in enumerate(feedback.split('\n')):
                cv2.putText(frame, line, (10, 60 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Store the form status
            self.last_form_feedback = feedback
            
            return frame, exercise_type
        else:
            # No pose detected
            cv2.putText(frame, "No pose detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, exercise_type

    def analyze_form(self, landmarks, exercise_type):
        """Analyze exercise form and return corrections with more detail"""
        if exercise_type not in self.correction_templates:
            return "Unknown exercise type"

        landmarks = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]

        errors = []
        detailed_feedback = ""

        if exercise_type == 'benchpress':
            # Check if back is flat (simplified)
            shoulder_hip_alignment = self.check_shoulder_hip_alignment(landmarks)
            if not shoulder_hip_alignment:
                errors.append(self.correction_templates[exercise_type]['errors'][0])
                detailed_feedback += " Your back is arching - press your lower back into the bench. "

            # Check elbow angle
            elbow_angle = self.get_angle(landmarks,
                                        self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                        self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                        self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
            if elbow_angle < 80:
                errors.append("Elbows too close to body")
                detailed_feedback += f" Elbow angle is {int(elbow_angle)}°, too narrow. Widen your grip slightly. "
            elif elbow_angle > 100:
                errors.append("Elbows flaring out too much")
                detailed_feedback += f" Elbow angle is {int(elbow_angle)}°, too wide. Tuck your elbows more. "

        elif exercise_type == 'deadlift':
            # Check if back is straight
            back_straightness = self.check_back_straightness(landmarks)
            if not back_straightness:
                errors.append(self.correction_templates[exercise_type]['errors'][0])
                detailed_feedback += " Your back is rounding - engage your core and maintain a neutral spine. "

            # Check hip position
            hip_position = self.check_hip_position(landmarks)
            if not hip_position:
                errors.append(self.correction_templates[exercise_type]['errors'][2])
                detailed_feedback += " Hip position is incorrect - start with hips lower, shoulders back. "

        elif exercise_type == 'squat':
            # Check knee alignment
            knee_alignment = self.check_knee_alignment(landmarks)
            if not knee_alignment:
                errors.append(self.correction_templates[exercise_type]['errors'][0])
                detailed_feedback += " Your knees are caving inward - push them out in line with your toes. "

            # Check back straightness
            back_straightness = self.check_back_straightness(landmarks)
            if not back_straightness:
                errors.append(self.correction_templates[exercise_type]['errors'][1])
                detailed_feedback += " Keep your chest up and back straight throughout the movement. "

            # Check squat depth
            squat_depth = self.check_squat_depth(landmarks)
            if not squat_depth:
                errors.append(self.correction_templates[exercise_type]['errors'][2])
                detailed_feedback += " You're not squatting deep enough - aim for thighs parallel to ground. "

        # If no specific errors detected or detailed feedback provided
        if not errors and not detailed_feedback:
            self.in_good_form = True
            return self.correction_templates[exercise_type]['good']
        elif detailed_feedback:
            self.in_good_form = False
            return detailed_feedback.strip()
        else:
            self.in_good_form = False
            return '\n'.join(errors[:2])  # Limit to 2 corrections at a time

    def get_angle(self, landmarks, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array([landmarks[p1][0], landmarks[p1][1]])
        b = np.array([landmarks[p2][0], landmarks[p2][1]])
        c = np.array([landmarks[p3][0], landmarks[p3][1]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Clip the cosine value to the valid range [-1, 1] to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def check_shoulder_hip_alignment(self, landmarks):
        """Check if shoulders and hips are aligned (for bench press)"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2

        # Check if shoulders and hips are approximately at the same height
        return abs(shoulder_y - hip_y) < 0.05

    def check_back_straightness(self, landmarks):
        """Check if back is straight"""
        shoulders = [(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] +
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] +
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2]

        hips = [(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][0] +
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][0]) / 2,
               (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][1] +
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][1]) / 2]

        # Check alignment between shoulders and hips
        return abs(shoulders[0] - hips[0]) < 0.1

    def check_hip_position(self, landmarks):
        """Check if hips are positioned correctly for deadlift"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2

        # Hips should be lower than normal standing position but not as low as squat
        return hip_y > knee_y and hip_y < knee_y + 0.2

    def check_knee_alignment(self, landmarks):
        """Check if knees are aligned with toes"""
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        left_alignment = abs(left_knee[0] - left_ankle[0]) < 0.05
        right_alignment = abs(right_knee[0] - right_ankle[0]) < 0.05

        return left_alignment and right_alignment

    def check_squat_depth(self, landmarks):
        """Check if squat is deep enough"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2

        # Check if hips are below or at knee level
        return hip_y >= knee_y

    def check_arm_vertical_alignment(self, landmarks):
        """Check if arms are vertically aligned during shoulder press"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        left_vertical = abs(left_shoulder[0] - left_wrist[0]) < 0.05
        right_vertical = abs(right_shoulder[0] - right_wrist[0]) < 0.05

        return left_vertical and right_vertical

    def select_exercise(self):
        """Allow user to select an exercise before starting the webcam"""
        print("\nALIGN")
        print("---------------------")
        print("\nPlease select an exercise type before starting:")
        
        for i, ex_type in enumerate(self.exercise_types):
            print(f"{i+1}. {ex_type}")
        print("0. Auto-detect exercise")
        
        while True:
            selected = input("\nEnter the number of the exercise type (0 for auto-detection): ")
            if selected.isdigit():
                selected = int(selected)
                if 0 <= selected <= len(self.exercise_types):
                    if selected == 0:
                        print("Auto-detecting exercise")
                        return None
                    else:
                        selected_exercise = self.exercise_types[selected-1]
                        print(f"Selected exercise: {selected_exercise}")
                        return selected_exercise
            print("Invalid selection. Please try again.")

    def track_repetition(self, is_good_form):
        """
        Improved repetition tracking using state machine logic.
        Only counts a rep when user maintains good form for multiple consecutive frames.
        """
        required_good_frames = 1  # Need 10 consecutive frames of good form to count a rep
        required_bad_frames = 1   # Need 10 consecutive frames of bad form to reset
        
        # Update debug info
        self.debug_info = {
            "is_good_form": is_good_form,
            "consecutive_good_frames": self.consecutive_good_frames,
            "consecutive_bad_frames": self.consecutive_bad_frames,
            "rep_started": self.rep_started
        }
        
        if is_good_form:
            self.consecutive_good_frames += 1
            self.consecutive_bad_frames = 0
            
            # If we have enough consecutive good frames and we've entered the rep
            if self.consecutive_good_frames >= required_good_frames and self.rep_started:
                # Count the rep as completed
                if self.rep_counter > 0:
                    self.rep_counter -= 1
                    print(f"Rep completed! {self.rep_counter} remaining.")
                
                # Reset for next rep
                self.rep_started = False
                self.consecutive_good_frames = 0
        else:
            # If form is bad
            self.consecutive_bad_frames += 1
            self.consecutive_good_frames = 0
            
            # If we have enough consecutive bad frames, reset the rep state
            if self.consecutive_bad_frames >= required_bad_frames:
                self.rep_started = True
                
        return self.rep_counter

    def run_webcam(self, initial_exercise=None):
        """Run real-time pose correction on webcam feed in local environment"""
        # Create window with controls
        cv2.namedWindow('Fitness Form Analyzer', cv2.WINDOW_NORMAL)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)  # Use default camera
        
        # Set webcam resolution to a lower value for wider view
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Get webcam properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        exercise_type = initial_exercise
        
        # Reset counters
        self.rep_counter = 20
        self.good_form_streak = 0
        self.consecutive_good_frames = 0
        self.consecutive_bad_frames = 0
        self.rep_started = True  # Start in a state ready to count reps
        
        print("\nALIGN Running")
        print("----------------------------")
        print("Press 'q' to quit")
        print("Press 'r' to reset rep counter")
        print("Press 's' to switch exercise (will prompt in console)")
        print("Press 'd' to toggle debug info")
        
        show_debug = False
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame. Camera might be disconnected.")
                break
            
            try:
                # Analyze the frame
                processed_frame, detected_exercise = self.analyze_pose(frame, exercise_type)
                
                # Get feedback
                landmarks = None
                feedback = "No pose detected"
                
                if detected_exercise:
                    landmarks = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
                    if landmarks:
                        feedback = self.analyze_form(landmarks, detected_exercise)
                        
                        # Update rep counter with improved tracking
                        is_good_form = feedback == self.correction_templates[detected_exercise]['good']
                        self.track_repetition(is_good_form)
                
                # Add rep counter to the frame
                cv2.rectangle(processed_frame, (frame_width-180, 10), (frame_width-10, 80), (0, 0, 0), -1)
                cv2.putText(processed_frame, f"Reps left: {self.rep_counter}", 
                            (frame_width-170, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                
                # Add feedback box at the bottom
                cv2.rectangle(processed_frame, (10, frame_height-80), (frame_width-10, frame_height-10), (0, 0, 0), -1)
                
                # Determine feedback color
                feedback_color = (255, 255, 255)  # Default white
                if "Good" in feedback:
                    feedback_color = (0, 255, 0)  # Green for good form
                elif any(error in feedback for error in ['incorrect', 'too', 'not']):
                    feedback_color = (0, 0, 255)  # Red for corrections
                
                # Display feedback text
                cv2.putText(processed_frame, feedback, 
                            (20, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
                
                # Show debug info if enabled
                if show_debug:
                    debug_y = 120
                    cv2.putText(processed_frame, f"Good form: {self.in_good_form}", 
                                (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, f"Consecutive good frames: {self.consecutive_good_frames}", 
                                (10, debug_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, f"Consecutive bad frames: {self.consecutive_bad_frames}", 
                                (10, debug_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, f"Rep started: {self.rep_started}", 
                                (10, debug_y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, f"Model loaded: {self.model_loaded}", 
                                (10, debug_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Fitness Form Analyzer', processed_frame)
            
            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.putText(frame, "Error processing frame", 
                        (20, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Fitness Form Analyzer', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' to quit
            if key == ord('q') or cv2.getWindowProperty("Fitness Form Analyzer", cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # 'r' to reset rep counter
            elif key == ord('r'):
                self.rep_counter = 20
                self.rep_started = True  # Ensure it starts tracking again
                print("Rep counter reset to 20")

                # Force immediate UI update
                cv2.rectangle(processed_frame, (frame_width-180, 10), (frame_width-10, 80), (0, 0, 0), -1)
                cv2.putText(processed_frame, f"Reps left: {self.rep_counter}", 
                            (frame_width-170, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            
            # 'd' to toggle debug info
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug info: {'ON' if show_debug else 'OFF'}")
            
            # 's' to switch exercise
            elif key == ord('s'):
                print("\nAvailable exercise types:")
                for i, ex_type in enumerate(self.exercise_types):
                    print(f"{i+1}. {ex_type}")
                print("0. Auto-detect exercise")
                
                selected = input("Enter the number of the exercise type (0 for auto-detection): ")
                if selected.isdigit():
                    selected = int(selected)
                    if 0 <= selected <= len(self.exercise_types):
                        if selected == 0:
                            exercise_type = None
                            print("Auto-detecting exercise")
                        else:
                            exercise_type = self.exercise_types[selected-1]
                            print(f"Selected exercise: {exercise_type}")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Path to the model downloaded from Colab
    model_path = "model/fitness_posture_model.keras"

    # Create and initialize the fitness posture correction system
    fitness_app = FitnessPostureCorrection(model_path)

    # Check for command-line arguments
    if len(sys.argv) > 1:
        exercise_type = sys.argv[1]
        # Validate the exercise type
        if exercise_type in fitness_app.exercise_types or exercise_type == 'auto':
            if exercise_type == 'auto':
                initial_exercise = None
            else:
                initial_exercise = exercise_type
            print(f"Starting with exercise: {exercise_type}")
        else:
            print(f"Invalid exercise type: {exercise_type}")
            initial_exercise = fitness_app.select_exercise()
    else:
        # If no arguments, allow user to select exercise
        initial_exercise = fitness_app.select_exercise()

    # Start the webcam analysis with the selected exercise
    fitness_app.run_webcam(initial_exercise)


if __name__ == "__main__":
    main()