from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

#Initializes the app
app = Flask(__name__)

#Configures upload settings for images
photos = UploadSet('photos',IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/results")
def results():

    result_img = None

    if request.args:
        result_img = request.args["result_img"]

    #Default html page render
    if not result_img:
        return render_template("results.html",result_text = "Empty Input",result_img = "static/img/fruit-basket.jpg")

    #Performs ML processing if there is a result image
    else:
        fruit_text = None #Put ML function call here
        fruit_img = None #Initial fruit image default

        if fruit_text == "Apple":
            fruit_img = "apple.jpg"

        elif fruit_text == "Orange":
            fruit_img = "orange.jpg"

        elif fruit_text == "Banana":
            fruit_img = "banana.jpg"

        return render_template("results.html",result_text = fruit_text, result_img = "static/img/{}".format(fruit_img))

@app.route("/handle-image",methods=["GET","POST"])
def handle_image():

    #Checks for a photo upload
    if request.method == "POST" and "photo" in request.files:
        filename = photos.save(request.files["photo"])
        return redirect(url_for("results",result_img = filename))

    return "Please check that you have uploaded a file"

if __name__ =="__main__":
    app.run(debug = True)


