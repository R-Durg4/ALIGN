import mysql.connector

try:
    connection = mysql.connector.connect(
        host="127.0.0.1",  # Try 'localhost' or '127.0.0.1'
        user="rdurg",
        password="Durga@0360",
        database="fitness"
    )
    
    if connection.is_connected():
        print("✅ Successfully connected to MySQL!")
    else:
        print("❌ Connection failed.")
except Exception as e:
    print(f"❌ Error: {e}")
