from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

@app.route("/")
def index():
    # connect to the database
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()
    # execute a SELECT statement to retrieve data from the database
    cursor.execute("SELECT * FROM mytable")
    rows = cursor.fetchall() # fetch all the rows returned by the SELECT statement
    # close the connection to the database
    conn.close()
    # render the template and pass the data to it
    return render_template("index.html", rows=rows)

if __name__ == "__main__":
    app.run()
