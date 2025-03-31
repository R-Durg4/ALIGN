from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, url_for, session, flash
from flask_mysqldb import MySQL
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from fitness_app import exercise_instance  # Import the global instance
from werkzeug.security import generate_password_hash
from sqlalchemy import create_engine
import pandas as pd
import stripe
import logging
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

# ✅ MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'  # Use 'localhost' if 127.0.0.1 fails
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Durga@0360'
app.config['MYSQL_DB'] = 'fitness'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # Enables dictionary output for queries

mysql = MySQL(app)

# ✅ Ensure MySQL Connection Works
try:
    conn = mysql.connection
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()
    print(f"✅ MySQL Connected Successfully! Version: {version}")
    cur.close()
except Exception as e:
    print(f"❌ Failed to connect to MySQL.\n⚠️ Error: {e}")

# ✅ Check MySQL Connection Before Each Request
@app.before_request
def check_mysql_connection():
    try:
        conn = mysql.connection
        if not conn:
            print("❌ MySQL connection lost, attempting reconnect...")
            mysql.connection.ping(reconnect=True)
    except Exception as e:
        print(f"⚠️ MySQL connection error: {e}")

# ✅ Ensure Table Exists
with app.app_context():
    try:
        cur = mysql.connection.cursor()  # FIXED: Changed from `app.mysql_conn` to `mysql.connection`
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        mysql.connection.commit()
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
    username = data.get('username')
    password = data.get('password')
    
    cur = mysql.connection.cursor()
    try:
        # Check if credentials exist in database
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        
        if user:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cur.close()

@app.route('/main_menu')
def main_menu():
    return render_template('main_menu.html')

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()  # Default to empty string
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Please fill in all fields!', 'error')
            return redirect(url_for('create_account'))

        cur = mysql.connection.cursor()
        try:
            # Check if username already exists
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                flash('Username already exists!', 'error')
                return redirect(url_for('create_account'))

            # Hash password before storing
            hashed_password = generate_password_hash(password)

            # Insert user into database
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                        (username, hashed_password))
            mysql.connection.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Database error: {str(e)}")
            mysql.connection.rollback()
            flash('An error occurred while creating your account.', 'error')
            return redirect(url_for('create_account'))
        finally:
            cur.close()

    return render_template('create_account.html')

@app.route('/select_exercise')
def select_exercise():
    return render_template('exercise_select.html')


#@app.route('/start_exercise/<exercise_type>')
#@app.route('/start_exercise/<exercise_type>')
@app.route('/start_exercise/<exercise_type>')
def start_exercise(exercise_type):
    try:
        # Get the absolute path to fitness_app.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'fitness_app.py')
        
        # Get the Python executable path
        python_exe = sys.executable
        
        # Start the fitness app in the same process without opening a new console window
        process = subprocess.Popen([python_exe, script_path, exercise_type], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
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


if __name__ == '__main__':
    app.run(debug=True)