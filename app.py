from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
from datetime import datetime
import json

app = Flask(__name__, static_folder='static')

attendance = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_name', methods=['GET', 'POST'])
def add_name():
    if request.method == 'POST':
        name = request.form['name']
        # Run headshots_picam.py with the provided name
        subprocess.run(['python', 'headshots_picam.py', name])
        return redirect(url_for('index'))
    return render_template('add_name.html')

@app.route('/mark_attendance')
def mark_attendance():
    # Run facial_rec.py to mark attendance
    subprocess.run(['python', 'facial_rec.py'])
    return redirect(url_for('index'))

@app.route('/see_attendance')
def see_attendance():
    return render_template('see_attendance.html', attendance=attendance)

if __name__ == '__main__':
    app.run(debug=True)
