from flask import Flask
from flask_mysqldb import MySQL

app = Flask(__name__)

# ✅ Correct MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'  # Use IP instead of 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Durga@0360'
app.config['MYSQL_DB'] = 'fitness'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)  # ✅ Initialize MySQL correctly

# ✅ Test MySQL Connection in Flask
try:
    conn = mysql.connection
    if conn:
        print("✅ Flask-MySQLdb Connected Successfully!")
    else:
        print("❌ Flask-MySQLdb Connection Failed.")
except Exception as e:
    print(f"⚠️ Error connecting to MySQL in Flask: {e}")

@app.route("/")
def home():
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT 1")  # Test query
        cur.close()
        return "✅ MySQL is working in Flask!"
    except Exception as e:
        return f"❌ MySQL Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
