from flask import Flask, render_template, request, render_template, send_from_directory, redirect, send_file, session, \
	jsonify, make_response, url_for,send_file
from flask_session import Session
import neuralStyleProcess
import test
import os
from stitch import multiStitching,loadImages,opencvStitching
import cv2
import numpy as np
import glob
import timeit
from flask_dropzone import Dropzone
import imutils
import pytesseract
# pip install gTTS
from gtts import gTTS
from forms import QRCodeData
import utils
from utils import base64_to_pil
from utils import np_to_base64
from PIL import Image
import secrets
import base64
import route
import utilss.Filters as filter
import utilss.Operations as op
import math
from video_utils import *
from config import *
app = Flask(__name__)
app.config['SECRET_KEY'] = "JLK24JO3I@!!$#Yoiouoln!#@oo=5y9y9youjuy952ou9859u923kjfhiy23ho"
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)
app.config['INITIAL_FILE_UPLOADS'] = 'static/uploads'
UPLOAD_FOLDER = 'static/image/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.abspath(os.path.dirname(__file__))
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=10,
    DROPZONE_MAX_FILES=100,

	AUDIO_FILE_UPLOAD = os.path.join(basedir, 'static/audio_files/')
)
app.config['DROPZONE_REDIRECT_VIEW'] = 'decoded'
dropzone = Dropzone(app)


def model_predict(img1, img2):
	# .......open cv here
	# img1 = cv2.imread("images/s1.png")
	# img2 = cv2.imread("images/s2.png")

	img1 = (np.array(img1))
	img2 = (np.array(img2))

	if (len(img1.shape) == 2):
		img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

	if (len(img2.shape) == 2):
		img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

	images = [img1, img2]
	stitcher = cv2.Stitcher_create()
	(status, stitched) = stitcher.stitch(images)

	stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
								  cv2.BORDER_CONSTANT, (0, 0, 0))
	gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	mask = np.zeros(thresh.shape, dtype="uint8")
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

	minRect = mask.copy()
	sub = mask.copy()
	while cv2.countNonZero(sub) > 0:
		minRect = cv2.erode(minRect, None)
		sub = cv2.subtract(minRect, thresh)

	cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	stitched = stitched[y:y + h, x:x + w]
	# cv2.imwrite("output.png", stitched)
	return stitched

# @app.route("/")
# def wel():
# 	return render_template("welcome.html")
@app.route("/", methods = ['GET', 'POST'])
def index():
	if (request.method == "POST"):
		if not request.files:
			return make_response({"status": "error"}), 400

		imageBuffer = request.files["image"].read()
		data = np.frombuffer(imageBuffer, np.uint8)
		image = cv2.imdecode(data, 1)

		filter = request.form.get("filter")
		if (filter == "1"):
			result = route.grayscale(image)
		elif (filter == "2"):
			result = route.setBrighness(image, 60)
		elif (filter == "3"):
			result = route.setBrighness(image, -60)
		elif (filter == "4"):
			result = route.laplasianFilter(image)
		elif (filter == "5"):
			result = route.sobel(image, 1, 0)
		elif (filter == "6"):
			result = route.sobel(image, 0, 1)
		elif (filter == "7"):
			result = route.sobel(image, 1, 1)
		elif (filter == "8"):
			result = route.canny(image)
		elif (filter == "9"):
			result = route.averagingBlur(image, (9, 9))
		elif (filter == "10"):
			result = route.gaussianBlur(image, (9, 9), 0)
		elif (filter == "11"):
			result = route.medianBlur(image, 9)
		elif (filter == "12"):
			result = route.bilateralBlur(image, 25, 75, 75)
		elif (filter == "13"):
			result = route.daoanh(image)
		image = cv2.imencode('.png', result)[1]

		return make_response(base64.b64encode(image.tobytes()))
	else:

		return render_template("index.html")
@app.route("/transfer")
def transfer():
	return render_template("style_transfer.html")


@app.route("/uploadtransfer", methods=['POST'])
def upload():
	target = os.path.join(APP_ROOT, 'images/')
	print("TARGET", target)

	if not os.path.isdir(target):
		os.mkdir(target)
	else:
		print("Couldn't create upload directory: {}".format(target))

	data = request.form.get("style")
	print(data)

	myFiles = []

	for file in request.files.getlist("file"):
		print("file", file)
		filename = file.filename
		print("filename", filename)
		destination = "".join([target, filename])
		print("destination", destination)
		file.save(destination)
		myFiles.append(filename)
	print(myFiles)

	return render_template("completetransfer.html", image_names=myFiles, selected_style=data)


# in this function send_image will HAVE to take in the parameter name <filename>
@app.route('/uploadtransfer/<filename>')
def send_original_image(filename):
	return send_from_directory("images", filename)


# this app route cant be the same as above
@app.route('/completeuploadtransfer/<filename>/<selected_style>')
def send_processed_image(filename, selected_style):
	directoryName = os.path.join(APP_ROOT, 'images/')

	newImg = neuralStyleProcess.neuralStyleTransfer(directoryName, filename, selected_style)

	return send_from_directory("images", newImg)
###
# @app.route('/uploadpanaroma', methods=['POST'])
# def handle_upload():
# 	for key, f in request.files.items():
# 		if key.startswith('file'):
# 			f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
# 	return '', 204
#
#
# @app.route('/formpanaroma', methods=['POST'])
# def handle_form():
# 	start = timeit.default_timer()
# 	opt = request.form.get('opt')
# 	resize = request.form.get('resize')
# 	print(resize)
# 	if resize == '1':
# 		list_images = loadImages(os.path.join(app.config['UPLOADED_PATH']), resize=1)
# 	else:
# 		list_images = loadImages(os.path.join(app.config['UPLOADED_PATH']), resize=0)
#
# 	for k in glob.glob(os.path.join(app.config['UPLOADED_PATH'] + '/*.*')):
# 		os.remove(k)
# 	crop = True
# 	if opt == '0':
# 		panorama = multiStitching(list_images, option='SURF', ratio=0.75)
# 	if opt == '1':
# 		panorama = multiStitching(list_images, option='ORB', ratio=0.75)
# 	if opt == '2':
# 		# panorama=multiStitching(list_images,option='SIFT',ratio=0.75)
# 		panorama = opencvStitching(list_images)
#
# 	cv2.imwrite('static/panorama.jpg', panorama)
# 	stop = timeit.default_timer()
# 	return render_template('resultparanoma.html', timee=stop - start)
#
#
# @app.route('/resultparanoma')
# def viewpanaroma():
# 	return render_template('resultparanoma.html')
@app.route("/panaroma")
def panaroma():
	return render_template("panaroma.html")
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img1,img2=base64_to_pil(request.json)
        img3=model_predict(img1,img2)
        img3=cv2.resize(img3, (400,300),interpolation=cv2.INTER_AREA)
        img3=np_to_base64(img3)
        #return jsonify({'image_url': '/output.png'})
        return jsonify(img3)
    return None
@app.route("/imgtext")
def imgtext():
	return render_template("imagetotext.html")
###
@app.route("/uploadimgtext", methods=["GET", "POST"])
def uploadimgtext():
	if request.method == 'POST':

		# set a session value
		sentence = ""

		f = request.files.get('file')
		filename, extension = f.filename.split(".")
		generated_filename = secrets.token_hex(10) + f".{extension}"

		file_location = os.path.join(app.config['UPLOADED_PATH'], generated_filename)

		f.save(file_location)

		# print(file_location)

		# OCR here
		pytesseract.pytesseract.tesseract_cmd = 'D:\\APPS\\Terra\\tesseract'

		img = cv2.imread(file_location)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		boxes = pytesseract.image_to_data(img)
		# print(boxes)

		for i, box in enumerate(boxes.splitlines()):
			if i == 0:
				continue

			box = box.split()
			# print(box)

			# only deal with boxes with word in it.
			if len(box) == 12:
				sentence += box[11] + " "

		# print(sentence)
		session["sentence"] = sentence

		# delete file after you are done working with it
		os.remove(file_location)

		return redirect("/decoded/")

	else:
		return render_template("uploadimgtext.html", title="Home")


@app.route("/decoded", methods=["GET", "POST"])
def decoded():
	sentence = session.get("sentence")
	# print(sentence)

	# print(lang)
	lang, _ = utils.detect_language(sentence)
	# print(lang, conf)

	form = QRCodeData()

	if request.method == "POST":
		generated_audio_filename = secrets.token_hex(10) + ".mp4"
		text_data = form.data_field.data
		translate_to = form.language.data
		# print("Data here", translate_to)

		translated_text = utils.translate_text(text_data, translate_to)
		print(translated_text)
		tts = gTTS(translated_text, lang=translate_to)

		file_location = os.path.join(
			app.config['AUDIO_FILE_UPLOAD'],
			generated_audio_filename
		)

		# save file as audio
		tts.save(file_location)

		# return redirect("/audio_download/" + generated_audio_filename)

		form.data_field.data = translated_text

		return render_template("decoded.html",
							   title="Decoded",
							   form=form,
							   lang=utils.languages.get(lang),
							   audio=True,
							   file=generated_audio_filename
							   )

	# form.data_field.data = sentence
	form.data_field.data = sentence

	# set the sentence back to defautl blank
	# sentence = ""
	session["sentence"] = ""

	return render_template("decoded.html",
						   title="Decoded",
						   form=form,
						   lang=utils.languages.get(lang),
						   audio=False
						   )
@app.route("/crops")
def crops():
		return render_template('crop.html')
@app.route("/resize", methods=["GET", "POST"])
def resize():

	# Execute if request is get
	if request.method == "GET":
		full_filename =  'static/uploads/photo.png'
		return render_template("resize.html", full_filename = full_filename)

	# Execute if reuqest is post
	if request.method == "POST":
		#option = request.form['options']
		width = int(request.form['width'])
		height = int(request.form['height'])
		image_upload = request.files['image_upload']
		imagename = image_upload.filename
		image = Image.open(image_upload)
		image = image.resize((width,height))
		img=image
		img.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'resize_image.png'))
		full_filename =  'static/uploads/resize_image.png'
		return render_template('resize.html', full_filename = full_filename)
###
@app.route("/setImageFil", methods=['POST', 'GET'])
def setImageFil():
    if request.method == "GET":
        return "<h1> it is a post method</h1>"
    image_mat = request.form["image"]
    session["image"] = image_mat
    if "output" in session:
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("filterim"))
@app.route("/filterim" , methods = ['GET', 'POST'])
def filterim():
	if "image" in session:
		show_image = "data:image/png;base64," + session["image"]
		if "output" in session:
			output = session["output"]
			op_name = session["op_name"]
			return render_template("filterim.html", orginal=show_image, output=output, op_name=op_name)
		return render_template("filterim.html", orginal=show_image)
	return render_template("filterim.html", orginal="no Image")
@app.route("/lines", methods=['GET'])
def lines():
    session["output"] = "data:image/png;base64," + \
        filter.lines(session["image"])
    session["op_name"] = "LINES"
    return redirect(url_for("filterim"))
@app.route("/pixel", methods=['GET'])
def pixelArt():
    session["output"] = "data:image/png;base64," + \
        filter.pixelArt(session["image"])
    session["op_name"] = "PIXXEL"
    return redirect(url_for("filterim"))
@app.route("/grayish", methods=['GET'])
def grayish():
    session["output"] = "data:image/png;base64," + \
        filter.grayish(session["image"])
    session["op_name"] = "GRAYISH"
    return redirect(url_for("filterim"))
@app.route("/emboss", methods=['GET'])
def emboss():
    session["output"] = "data:image/png;base64," + \
        filter.emboss(session["image"])
    session["op_name"] = "EMBOSS"
    return redirect(url_for("filterim"))
@app.route("/pop", methods=['GET'])
def popArt():
    session["output"] = "data:image/png;base64," + \
        filter.popArt(session["image"])
    session["op_name"] = "BONCUK"
    return redirect(url_for("filterim"))
@app.route("/moon", methods=['GET'])
def moon():
    session["output"] = "data:image/png;base64," + \
        filter.moon(session["image"])
    session["op_name"] = "MOON"
    return redirect(url_for("filterim"))
@app.route("/oldtv", methods=['GET'])
def oldtv():
    session["output"] = "data:image/png;base64," + \
        filter.oldtv(session["image"])
    session["op_name"] = "90's TV"
    return redirect(url_for("filterim"))
@app.route("/sketch", methods=['GET'])
def sketch():
    session["output"] = "data:image/png;base64," + \
        filter.sketch(session["image"])
    session["op_name"] = "SKETCH"
    return redirect(url_for("filterim"))
@app.route("/splash", methods=['GET'])
def splash():
    session["output"] = "data:image/png;base64," + \
        filter.splash(session["image"])
    session["op_name"] = "SPLASH"
    return redirect(url_for("filterim"))
@app.route("/sepya", methods=['GET'])
def sepya():
    session["output"] = "data:image/png;base64," + \
        filter.sepya(session["image"])
    session["op_name"] = "SEPIA"
    return redirect(url_for("filterim"))
@app.route("/cartoon", methods=['GET'])
def cartoon():
    session["output"] = "data:image/png;base64," + \
        filter.cartoon(session["image"])
    session["op_name"] = "CARTOON"
    return redirect(url_for("filterim"))
@app.route("/oily", methods=['GET'])
def oily():
    session["output"] = "data:image/png;base64," + \
        filter.oily(session["image"])
    session["op_name"] = "OILY"
    return redirect(url_for("filterim"))
@app.route("/abstractify", methods=['GET'])
def abstractify():
    session["output"] = "data:image/png;base64," + \
        filter.abstractify(session["image"])
    session["op_name"] = "NOTIONAL"
    return redirect(url_for("filterim"))
@app.route("/balmy", methods=['GET'])
def balmy():
    session["output"] = "data:image/png;base64," + \
        filter.warm(session["image"])
    session["op_name"] = "BALMY"
    return redirect(url_for("filterim"))
@app.route("/cold", methods=['GET'])
def cold():
    session["output"] = "data:image/png;base64," + \
        filter.cold(session["image"])
    session["op_name"] = "FROSTBITE"
    return redirect(url_for("filterim"))

@app.route("/blush", methods=['GET'])
def blush():
    session["output"] = "data:image/png;base64," + \
        filter.blush(session["image"])
    session["op_name"] = "BLUSH"
    return redirect(url_for("filterim"))
@app.route("/glass", methods=['GET'])
def glass():
    session["output"] = "data:image/png;base64," + \
        filter.glass(session["image"])
    session["op_name"] = "GLASS"
    return redirect(url_for("filterim"))
@app.route("/xpro", methods=['GET'])
def xpro():
    session["output"] = "data:image/png;base64," + \
        filter.xpro(session["image"])
    session["op_name"] = "XPRO"
    return redirect(url_for("filterim"))
@app.route("/daylight", methods=['GET'])
def daylight():
    session["output"] = "data:image/png;base64," + \
        filter.daylight(session["image"])
    session["op_name"] = "DAYLIGHT"
    return redirect(url_for("filterim"))
@app.route("/blueish", methods=['GET'])
def blueish():
    session["output"] = "data:image/png;base64," + \
        filter.blueish(session["image"])
    session["op_name"] = "BLUEISH"
    return redirect(url_for("filterim"))
###
@app.route("/rotates")
def rotates():
    if "image" in session:
        show_image = "data:image/png;base64,"+session["image"]
        if "output" in session:
            output = session["output"]
            op_name = session["op_name"]
            return render_template("rotate.html", orginal=show_image, output=output, op_name=op_name)
        return render_template("rotate.html", orginal=show_image)
    return render_template("rotate.html", orginal="no Image")
@app.route("/setImage", methods=['POST', 'GET'])
def setImage():
    if request.method == "GET":
        return "<h1> it is a post method</h1>"
    image_mat = request.form["image"]
    session["image"] = image_mat
    if "output" in session:
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("rotates"))


# @app.route("/save", methods=['GET'])
# def saveChanges():
#     if "output" in session:
#         session["image"] = session["output"].replace(
#             "data:image/png;base64,", "")
#         session.pop("output")
#     if "op_name" in session:
#         session.pop("op_name")

    # return redirect(url_for("rotates"))

@app.route("/rotate", methods=['GET'])
def rotate():
    angle = int(request.headers["angle"])
    session["output"] = "data:image/png;base64," + \
        op.rotate(session["image"], angle)

    session["op_name"] = "ROTATE"
    return redirect(url_for("rotates"))
#####
@app.route("/editimgadvan")
def editimgadvan():
    if "image" in session:
        show_image = "data:image/png;base64,"+session["image"]
        if "output" in session:
            output = session["output"]
            op_name = session["op_name"]
            return render_template("editimgadvan.html", orginal=show_image, output=output, op_name=op_name)
        return render_template("editimgadvan.html", orginal=show_image)
    return render_template("editimgadvan.html", orginal="no Image")
@app.route("/setImageedita", methods=['POST', 'GET'])
def setImageedita():
    if request.method == "GET":
        return "<h1> it is a post method</h1>"
    image_mat = request.form["image"]
    session["image"] = image_mat
    if "output" in session:
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("editimgadvan"))
@app.route("/cropselect", methods=['GET'])
def crop_select():

    session["op_name"] = "CROP_SELECT"
    session["output"] = "select for crop"

    return redirect(url_for("editimgadvan"))

@app.route("/crop", methods=['GET'])
def crop():

    p = request.headers["points"]
    w = int(request.headers["width"])
    p = p.split(',')
    points = []
    i = 0

    while i <= len(p)-2:
        points.append([math.floor(float(p[i])), math.floor(float(p[i+1]))])
        i += 2

    session["output"] = "data:image/png;base64," + \
        op.crop(session["image"], points, w)
    session["op_name"] = "CROP"

    return redirect(url_for("editimgadvan"))


@app.route("/flip", methods=['GET'])
def flip():
    try:
        _hor = request.headers["hor"]
        if(_hor == "false"):
            hor = False
        else:
            hor = True
        _ver = request.headers["ver"]

        if(_ver == "false"):
            ver = False
        else:
            ver = True

    except:
        hor = False
        ver = False

    session["output"] = "data:image/png;base64," + \
        op.flip(session["image"], hor, ver)

    session["op_name"] = "FLIPPER"
    return redirect(url_for("editimgadvan"))
@app.route("/contrast", methods=['GET'])
def contrast():
    contrast = int(request.headers["contrast"])
    session["output"] = "data:image/png;base64," + \
        op.contrast(session["image"], contrast)

    session["op_name"] = "CONTRAST"
    return redirect(url_for("editimgadvan"))
@app.route("/lumos", methods=['GET'])
def lumos():
    lumen = int(request.headers["lumen"])
    session["output"] = "data:image/png;base64," + \
        op.lumos(session["image"], lumen)

    session["op_name"] = "BRIGHT"
    return redirect(url_for("editimgadvan"))
@app.route("/save", methods=['GET'])
def saveChanges():
    if "output" in session:
        session["image"] = session["output"].replace(
            "data:image/png;base64,", "")
        session.pop("output")
    if "op_name" in session:
        session.pop("op_name")

    return redirect(url_for("editimgadvan"))

########
@app.route('/editvideo')
def editvideo():
    return render_template("editvideo.html")


@app.route('/clips/<filename>')
def renderClip(filename):
    return send_file(video_savepath + filename)


# Uploads a video file to server and returns filename
@app.route('/upload_video', methods=['POST'])
def uploadVideo():
    # check if video savepath exists
    if not os.path.isdir("./clips"):
        os.mkdir("./clips")
    try:
        videofile = request.files['videofile']
        filepath = video_savepath + videofile.filename
        videofile.save(filepath)
    except:
        return "ERROR"

    return str(filepath)


# Main video editing pipeline
@app.route('/edit_video/<actiontype>', methods=['POST'])
def editVideo(actiontype):
    if actiontype == "trim":
        try:
            edited_videopath = trimVideo(request.form['videofile'], int(request.form['trim_start']),
                                         int(request.form['trim_end']))
            return {
                "status": "success",
                "message": "video edit success",
                "edited_videopath": edited_videopath
            }
        except Exception as e:
            return {
                "status": "error",
                "message": "video edit failure: " + str(e),
            }


@app.route('/merged_render', methods=['POST'])
def mergedRender():
    try:
        videoscount = int(request.form['videoscount'])
        if videoscount > 0:
            videoclip_filenames = []
            for i in range(videoscount):
                videoclip_filenames.append(request.form['video' + str(i)])

            finalrender_videopath = mergeVideos(videoclip_filenames)
            return {
                "status": "success",
                "message": "merged render success",
                "finalrender_videopath": finalrender_videopath
            }
        else:
            return {
                "status": "error",
                "message": "merged render error. Invalid videos count"
            }

    except Exception as e:
        return {
            "status": "error",
            "message": "video merge failure: " + str(e),
        }
if __name__ == '__main__':
    app.run(debug=True)


