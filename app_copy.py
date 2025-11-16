from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, url_for, session, flash
from flask_mysqldb import MySQL
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from fitness_app import exercise_instance  # Import the global instance
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from sqlalchemy import create_engine
#added for workout progress
from datetime import date
#added import session to store user data
from flask import session
import pandas as pd
import stripe
import logging
import subprocess
import sys
import MySQLdb
import sqlite3
import datetime
from pynput import mouse
import random
app = Flask(__name__)

# ✅ MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'  # Use 'localhost' if 127.0.0.1 fails
app.config['MYSQL_USER'] = ''
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'fitness'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # Enables dictionary output for queries
app.config['SECRET_KEY'] = os.urandom(24)

#added upload folder for pfp
UPLOAD_FOLDER = 'static/profile_pics'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


from checkmysql import get_mysql_connection
conn = get_mysql_connection()

mysql=MySQL(app)  # Initialize MySQL correctly #remove if any error occurs

#  Ensure MySQL Connection Works
try:
    cur = conn.cursor(dictionary=True)  # Use dictionary format for queries
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()
    print(f"✅ MySQL Connected Successfully! Version: {version}")
    cur.close()
except Exception as e:
    print(f"❌ Failed to connect to MySQL.\n⚠️ Error: {e}")

#  Check MySQL Connection Before Each Request
@app.before_request
def check_mysql_connection():
    try:
        conn.ping(reconnect=True)
        if not conn:
            print("❌ MySQL connection lost, attempting reconnect...")
            mysql.connection.ping(reconnect=True)
    except Exception as e:
        print(f"⚠️ MySQL connection error: {e}")

# Ensure Table Exists
with app.app_context():
    try:
        cur = conn.cursor(dictionary=True)  # Use dictionary format for queries
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
         

        print("✅ Users table checked/created successfully.")
    except Exception as e:
        print(f"⚠️ Error creating users table: {e}")

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/validate_login', methods=['POST'])
def validate_login():
    data = request.get_json()
    
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'})

    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cur.close()


@app.route('/main_menu')
def main_menu():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    global userid
    userid=[user_id]
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT profile_picture FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()

    return render_template('main_menu.html', user=user)

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        username = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Please fill in all fields!', 'error')
            print("❌ Form fields are empty!")
            return redirect(url_for('create_account'))

        # ✅ Ensure MySQL Connection is Active
        if conn is None:
            print("❌ Database connection is None! Check checkmysql.py")
            flash('Database connection error!', 'error')
            return redirect(url_for('create_account'))

        try:
            cur = conn.cursor(dictionary=True)
            print(f"🔍 Checking if username '{username}' already exists...")

            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            existing_user = cur.fetchone()
            print(f"🔍 Existing User Query Result: {existing_user}")

            if existing_user:
                flash('Username already exists!', 'error')
                print(f"⚠️ Username '{username}' already exists.")
                return redirect(url_for('create_account'))

            # ✅ Correct Password Hashing (IMPORTANT)
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            print(f"🔍 Inserting User: {username}, Password Hash: {hashed_password}")

            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                        (username, hashed_password))
            
            conn.commit()
            print("✅ User inserted successfully!")

            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"❌ Database error: {str(e)}")
            conn.rollback()
            flash('An error occurred while creating your account.', 'error')
            return redirect(url_for('create_account'))
        finally:
            cur.close()

    return render_template('create_account.html')

#added settings route
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))
    
    user_id = session['user_id']  # Get logged-in user ID

    # Fetch user details from DB
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT full_name, username, phone, gender, country_code, profile_picture FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        phone = request.form.get('phone')
        country_code = request.form.get('country_code')
        gender = request.form.get('gender')
        new_username = request.form.get('username')
        new_password = request.form.get('password')

        # Handle Profile Picture Upload
        profile_picture = request.files.get('profile_picture')
        profile_picture_path = user["profile_picture"]  # Default to existing picture
        #newline
        if not user['profile_picture']:  # If no profile picture exists
            user['profile_picture'] = 'profile_pics/default.png'  # Set default picture path
        #newline over
        if profile_picture and profile_picture.filename:
            filename = secure_filename(profile_picture.filename)
            profile_picture_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            profile_picture.save(profile_picture_path)
            profile_picture_path = f"profile_pics/{filename}"  # Relative path for DB

        try:
            cur = conn.cursor()
            
            # Base update query
            update_query = """
                UPDATE users 
                SET full_name = %s, phone = %s, country_code = %s, gender = %s, username = %s
            """
            
            params = [full_name, phone, country_code, gender, new_username]

            # Update password only if provided
            if new_password:
                hashed_password = generate_password_hash(new_password)
                update_query += ", password = %s"
                params.append(hashed_password)

            # Update profile picture only if a new one is uploaded
            if profile_picture_path:
                update_query += ", profile_picture = %s"
                params.append(profile_picture_path)

            # Add WHERE condition
            update_query += " WHERE id = %s"
            params.append(user_id)

            # Execute update query
            cur.execute(update_query, tuple(params))
            conn.commit()

            print("Database update successful!")  # Debugging
            flash("Profile updated successfully!", "success")

        except Exception as e:
            conn.rollback()
            flash(f"Error updating profile: {str(e)}", "error")

        finally:
            cur.close()

        return redirect(url_for('main_menu'))

    return render_template('settings.html', user=user)
#added for view profile
@app.route('/profile')
def view_profile():
    # Ensure the user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated

    user_id = session['user_id']
    print(f"User ID from session: {user_id}")  # Debugging
    
    # Fetch user details from the database
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, full_name, username, phone, gender, country_code, profile_picture, password FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()

    if not user:
        return "User not found!", 404  # Handle case where user is not found

    return render_template("profile.html", user=user)


@app.route('/select_exercise')
def select_exercise():
    return render_template('exercise_select.html')


#@app.route('/start_exercise/<exercise_type>')
#@app.route('/start_exercise/<exercise_type>')
@app.route('/start_exercise/<exercise_type>')
def start_exercise(exercise_type):
    user_id=session['user_id']
    try:
        # Get the absolute path to fitness_app.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'fitness_app.py')

        # Get the Python executable path
        python_exe = sys.executable

        # Start the fitness app in the same process without opening a new console window
        process = subprocess.Popen([python_exe, script_path, exercise_type], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start_time = datetime.datetime.now()
        print(f"Exercise {exercise_type} started at {start_time}")

        global repcount, accuracy, end_time, duration

        # Function to log end time on mouse click
        def on_click(x, y, button, pressed):
            if pressed:
                global end_time, duration, repcount, accuracy
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds() / 60  # Convert to minutes
                print(f"Exercise {exercise_type} ended at {end_time}")
                print(f"Total duration: {duration:.2f} minutes")
                listener.stop()
                print("Listener stopped.")
                # Simulate rep count and accuracy
                repcount = random.randint(5, 12)
                accuracy = (repcount / 20) * 100
                [print(f"Rep count: {repcount}, Accuracy: {accuracy:.2f}%")]
                print(userid)
                # Get user ID (assuming it's stored in session)
                # Insert data into MySQL database
                try:

                    cursor = conn.cursor(dictionary=True)
                    query = """INSERT INTO workout (user_id, start_time, end_time, duration, repcount, accuracy) 
                               VALUES (%s, %s, %s, %s, %s, %s)"""
                    values = (user_id, start_time, end_time, duration, repcount, accuracy)
                    print("hello")
                    cursor.execute(query, values)
                    conn.commit()
                    cursor.close()
                    conn.close()
                    print("Workout data saved to database.")

                except Exception as db_error:
                    print(f"Database error: {db_error}")

        # Start listening for mouse clicks
        listener = mouse.Listener(on_click=on_click)
        listener.start()

        # Check if process started successfully
        if process.poll() is None:
            flash(f'Exercise program started successfully for {exercise_type}', 'success')
            return render_template('exercise.html')
        else:
            flash('Failed to start exercise program', 'error')
            return redirect(url_for('main_menu'))

    except Exception as e:
        print(f"Error starting exercise: {e}")
        flash('Error starting exercise program', 'error')
        return redirect(url_for('main_menu'))
#added reset rep counter
@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    try:
        if exercise_instance:
            exercise_instance.reset_rep_counter()
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'No active exercise session'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

#added for updating progress daily


@app.route("/view_progress")
def view_progress():
    user_id = session['user_id']
    print(f"User ID from session: {user_id}")
    cur = conn.cursor(dictionary=True)
    
    # Fetch total workouts, time spent, improvement percentage
    cur.execute("SELECT id,user_id,start_time,end_time,duration,repcount,accuracy FROM workout WHERE user_id = %s", (user_id,))
    progress_summary = cur.fetchall()
    cur.close()
    print(progress_summary)
    return render_template('view_progress.html', progress_data=progress_summary)



if __name__ == '__main__':

    app.run(debug=True)
