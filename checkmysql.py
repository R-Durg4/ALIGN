import mysql.connector

def get_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",  
            user="rdurg",
            password="Durga@0360",
            database="fitness"
        )
        if connection.is_connected():
            print("✅ Successfully connected to MySQL!")
        return connection
    except Exception as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None
