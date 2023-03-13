import sqlite3
import os
import time
class duombaze:
    def __init__(self):
        # Connect to SQLite3 database (creates a new one if it doesn't exist)
        self.conn = sqlite3.connect('example.db')
        self.veido_path = os.path.join('application_data', 'input_image', 'input_image.jpg')

    def _create_table(self):
        # Create a table
        cursor = self.conn.cursor()
        #BLOB skirtas vaizdui, date skirtas aptikimo datai
        cursor.execute('''CREATE TABLE veidai
                (rezultatas text, aptikimo_laikas date, veidas blob)''')

    def close_table(self):
        # Commit changes and close connection
        self.conn.commit()
        self.conn.close()

    def update_table(self):
        self.veido_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
        with open(self.veido_path, 'rb') as f:
            veido_duomenys = f.read()
        s1 = os.parth.getmtime(self.veido_path)
        s2 = time.ctime(s1)
        s3 = time.strptime(s2)
        veido_data = time.strftime("%Y-%m-%d %H:%M:%S", s3)
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO veidas (name, image) VALUES (?, ?)', (self.veido_path, veido_duomenys))
        cursor.execute('INSERT INTO aptikimo_laikas (name, date) VALUES (?, ?)', (self.veido_path, veido_data))