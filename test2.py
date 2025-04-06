from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, url_for ,session, flash,request
from flask_mysqldb import MySQL
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import os
from datetime import date, datetime
from sqlalchemy import create_engine
import pandas as pd
import stripe
import logging
import subprocess
import sys

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'rdurg'
app.config['MYSQL_PASSWORD'] = 'Durga@0360'
app.config['MYSQL_DB'] = 'fitness'
app.config['SECRET_KEY'] = os.urandom(24)

mysql = MySQL(app)
app.mysql = mysql  # new line: expose mysql to other modules

with app.app_context():
    cur = conn.cursor(dictionary=True)
    # Drop table if exists
    #cur.execute("DROP TABLE IF EXISTS users")
    mysql.connection.commit()
    
    # Create new table
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

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/validate_login', methods=['POST'])
def validate_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    cur = conn.cursor(dictionary=True)
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
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields!', 'error')
            return render_template('create_account.html')
        
        cur = conn.cursor(dictionary=True)
        try:
            # Check if username already exists
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                flash('Username already exists!', 'error')
                return render_template('create_account.html')
                
            # Insert user credentials into database
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                       (username, password))
            mysql.connection.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Database error: {str(e)}")
            mysql.connection.rollback()
            flash(f'Error creating account: {str(e)}', 'error')
            return render_template('create_account.html')
        finally:
            cur.close()
    
    return render_template('create_account.html')


@app.route('/select_exercise')
def select_exercise():
    return render_template('exercise_select.html')


@app.route('/start_exercise/<exercise_type>')
def start_exercise(exercise_type):
    try:
        # Get the absolute path to fitness_app.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'fitness_app.py')
        
        # Get the Python executable path
        python_exe = sys.executable
        
        # Start the fitness app in a new process with the exercise type as argument
        process = subprocess.Popen([python_exe, script_path, exercise_type], 
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
        
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

if __name__ == '__main__':
    app.run(debug=True)