import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import mediapipe as mp
import time

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
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Exercise categories 
        self.exercise_types = ['benchpress', 'deadlift', 'squat']
        self.rep_counter = 20
        self.good_form_streak = 0
        self.last_form_feedback = ""

        # Display settings
        self.display_width = 1280
        self.display_height = 720

        # Load model
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("Model path not found. Make sure you've downloaded the model from Colab.")
            self.model = self.build_model()
            print("Using default untrained model - results will not be accurate without training.")

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
        # Base model with pre-trained weights
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Freeze the base model layers
        base_model.trainable = False

        # Create the model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(len(self.exercise_types), activation='softmax')  # Exercise classification
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def analyze_pose(self, frame, exercise_type=None):
        """Analyze pose and provide corrections based on exercise type"""
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)

        # If pose landmarks were detected
        if results.pose_landmarks:
            # Draw pose landmarks on frame with custom styling
            self.mp_drawing.draw_landmarks(
                display_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
            )

            # If exercise type not provided, predict it
            if exercise_type is None:
                # Preprocess frame for the classification model
                resized_frame = cv2.resize(rgb_frame, (224, 224))
                preprocessed = preprocess_input(np.expand_dims(resized_frame, axis=0))

                # Predict exercise type
                prediction = self.model.predict(preprocessed)
                exercise_idx = np.argmax(prediction[0])
                exercise_type = self.exercise_types[exercise_idx]
                confidence = prediction[0][exercise_idx]

                # Display predicted exercise with better styling
                cv2.rectangle(display_frame, (10, 10), (300, 40), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Exercise: {exercise_type} ({confidence:.2f})",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Analyze form based on exercise type
            feedback = self.analyze_form(results.pose_landmarks, exercise_type)
            self.last_form_feedback = feedback

        return display_frame, exercise_type

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
            return self.correction_templates[exercise_type]['good']
        elif detailed_feedback:
            return detailed_feedback.strip()
        else:
            return '\n'.join(errors[:2])  # Limit to 2 corrections at a time

    def get_angle(self, landmarks, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array([landmarks[p1][0], landmarks[p1][1]])
        b = np.array([landmarks[p2][0], landmarks[p2][1]])
        c = np.array([landmarks[p3][0], landmarks[p3][1]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
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
    
    def draw_text_with_background(self, img, text, position, font_scale=0.7, color=(255, 255, 255), 
                                 thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 0, 0), 
                                 padding=10, alpha=0.7):
        """Draw text with semi-transparent background for better readability"""
        # Get text size
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        
        # Calculate background rectangle dimensions
        rect_x, rect_y = position[0] - padding, position[1] - text_h - padding
        rect_w, rect_h = text_w + 2 * padding, text_h + 2 * padding
        
        # Create overlay for semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), bg_color, -1)
        
        # Apply the overlay with transparency
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw text
        cv2.putText(img, text, (position[0], position[1] - padding), font, font_scale, color, thickness)
        
        return img

    def run_webcam(self):
        """Run real-time pose correction on webcam feed with improved visualization"""
        # Create window with controls
        cv2.namedWindow('Fitness Form Analyzer', cv2.WINDOW_NORMAL)
        
        # Create exercise type dropdown
        exercise_type = None
        print("Available exercise types:")
        for i, ex_type in enumerate(self.exercise_types):
            print(f"{i+1}. {ex_type}")
        
        print("0. Auto-detect exercise")
        
        selected = input("Enter the number of the exercise type (0 for auto-detection): ")
        if selected.isdigit():
            selected = int(selected)
            if 1 <= selected <= len(self.exercise_types):
                exercise_type = self.exercise_types[selected-1]
                print(f"Selected exercise: {exercise_type}")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)  # Use default camera
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam. Make sure your camera is connected and not being used by another application.")
            return
        
        # Set webcam properties for better resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual webcam properties (may differ from requested)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set window size to match camera resolution
        cv2.resizeWindow('Fitness Form Analyzer', actual_width, actual_height)
        
        # Reset counters
        self.rep_counter = 20
        self.good_form_streak = 0
        
        print("\nFitness Form Analyzer")
        print("---------------------")
        print("Press 'q' to quit")
        print("Press 'r' to reset rep counter")
        print("Press 's' to switch exercise (will prompt in console)")
        print("Press 'f' to toggle fullscreen")
        
        # Track if we're in fullscreen mode
        fullscreen_mode = False
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame. Camera might be disconnected.")
                break
            
            # Resize frame if needed to fit display
            if frame.shape[1] != self.display_width or frame.shape[0] != self.display_height:
                frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            # Analyze the frame
            processed_frame, detected_exercise = self.analyze_pose(frame, exercise_type)
            
            # Get feedback
            landmarks = None
            feedback = "No pose detected"
            
            if detected_exercise:
                landmarks = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
                if landmarks:
                    feedback = self.analyze_form(landmarks, detected_exercise)
                    
                    # Update rep counter if form is good
                    if feedback == self.correction_templates[detected_exercise]['good']:
                        self.good_form_streak += 1
                        if self.good_form_streak >= 3:  # Need 10 frames of good form to count as a rep
                            if self.rep_counter > 0:
                                self.rep_counter -= 1
                            self.good_form_streak = 0
                    else:
                        self.good_form_streak = 0
            
            # Create a stylish UI overlay
            # 1. Add semi-transparent top bar
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.display_width, 80), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
            
            # 2. Add exercise type and rep counter
            exercise_text = f"Exercise: {detected_exercise if detected_exercise else 'Auto-detect'}"
            cv2.putText(processed_frame, exercise_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Add rep counter with colored background
            rep_text = f"Reps left: {self.rep_counter}"
            rep_color = (50, 205, 50) if self.rep_counter > 5 else (0, 0, 255)  # Green if > 5, else red
            rep_bg = (30, 60, 30) if self.rep_counter > 5 else (60, 30, 30)
            
            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            rep_x = self.display_width - text_width - 30
            
            # Draw rep counter with background
            cv2.rectangle(processed_frame, (rep_x - 10, 10), (rep_x + text_width + 10, 60), rep_bg, -1)
            cv2.putText(processed_frame, rep_text, (rep_x, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, rep_color, 2)
            
            # 3. Add feedback panel at bottom
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (0, self.display_height - 100), 
                         (self.display_width, self.display_height), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
            
            # Determine feedback color
            feedback_color = (255, 255, 255)  # Default white
            if "Good" in feedback:
                feedback_color = (0, 255, 0)  # Green for good form
            elif any(error in feedback for error in ['incorrect', 'too', 'not']):
                feedback_color = (0, 100, 255)  # Orange for corrections
            
            # Display feedback text with nice formatting
            cv2.putText(processed_frame, "FORM FEEDBACK:", 
                        (20, self.display_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(processed_frame, feedback, 
                        (20, self.display_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
            
            # Add controls guide at bottom right
            controls_text = "Q: Quit | R: Reset | S: Switch | F: Fullscreen"
            cv2.putText(processed_frame, controls_text, 
                        (self.display_width - 400, self.display_height - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Display the frame
            cv2.imshow('Fitness Form Analyzer', processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' to quit
            if key == ord('q'):
                break
            
            # 'r' to reset rep counter
            elif key == ord('r'):
                self.rep_counter = 20
                print("Rep counter reset to 20")
            
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
            
            # 'f' to toggle fullscreen
            elif key == ord('f'):
                fullscreen_mode = not fullscreen_mode
                if fullscreen_mode:
                    cv2.setWindowProperty('Fitness Form Analyzer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('Fitness Form Analyzer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Path to the model downloaded from Colab
    model_path = "fitness_posture_model.keras"

    # Create and initialize the fitness posture correction system
    fitness_app = FitnessPostureCorrection(model_path)

    # Start the webcam analysis
    fitness_app.run_webcam()


if __name__ == "__main__":
    main()
