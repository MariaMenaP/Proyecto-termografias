import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
import numpy as np
from preprocesamiento import preprocesamiento
from areas import info_areas
from musculatura import musculatura
from skimage import io



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)

	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		global filename
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route("/zona_a_tratar", methods=["POST"])
def zona_a_tratar():
	image = io.imread(r"/Users/mariamena/Desktop/Adamo/static/uploads/{}".format(filename))
	#image = cv2.imread(r"/Users/mariamena/Desktop/Adamo/static/uploads/{}.png".format(filename))
	preprocesamiento(image)
	#io.imsave(r"/Users/mariamena/Desktop/Adamo/static/zona_a_tratar/{}".format(filename),x)
	return redirect(url_for('static', filename='zona_a_tratar/1.png'))

@app.route("/colores", methods=["POST"])
def colores():
	image = io.imread(r"/Users/mariamena/Desktop/Adamo/static/uploads/{}".format(filename))
	info_areas(image)
	return redirect(url_for('static', filename='zona_a_tratar/2.png'), code=301)

@app.route("/musc", methods=["POST"])
def musc():
	image = io.imread(r"/Users/mariamena/Desktop/Adamo/static/uploads/{}".format(filename))
	musculatura(image)
	return redirect(url_for('static', filename='zona_a_tratar/3.png'), code=301)

if __name__ == "__main__":
    app.run(port = 3000, debug =True)

