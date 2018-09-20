from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
from SparkRottenFreshApple import SRFA
import findspark

#Initializes the app and dependency classes
app = Flask(__name__)

#Configures upload settings for images
photos = UploadSet('photos',IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

#Init Spark node + ml model
srfa = SRFA()

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

        #Performs processing of the image
        '''
        print("static/img/{}".format(result_img))
        abo_util = ABO("static/img/{}".format(result_img))
        fruit_text = abo_util.getFruitList()[0]
        '''
        fruit_state = srfa.getFruitState("static/img/{}".format(result_img))[0]

        if(fruit_state == "Rotten"):
            fruit_img = "rottenApple.jpg"
        elif(fruit_state == "Fresh"):
            fruit_img = "freshApple.jpg"

        return render_template("results.html",result_text = fruit_text, result_img = "static/img/{}".format(fruit_img))

@app.route("/handle-image",methods=["GET","POST"])
def handle_image():

    #Checks for a photo upload
    if request.method == "POST" and "photo" in request.files:
        request.files["photo"].filename = str(hash(request.files["photo"].filename))+".jpg"
        filename = photos.save(request.files["photo"])
        return redirect(url_for("results",result_img = filename))

    return "Please check that you have uploaded a file"

if __name__ =="__main__":
    app.run(debug = True)


