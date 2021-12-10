from flask import Flask, render_template, url_for, request
from hwgan import *
import matplotlib.pyplot as plt
import time

app = Flask(__name__)


cgens, cdims = get_generators()
ldict = get_emnist_byclass_dict()
pdict = get_punc_dict('!,.?')
cdicts = (ldict, pdict)
# print(cgens, cdims, cdicts)


def savefig(text, filename, cdims, cdicts, cgens):
    img = get_text_lines(text, 40, 28, 28, cdims, cdicts, cgens)
    fig = plt.figure(figsize=(30, 20))
    plt.imshow(img, cmap = 'gray_r')
    plt.axis('off')
    plt.savefig(filename, dpi=300)

@app.route('/', methods=["POST", "GET"])
def index():
    if request.method == "POST":
        text = request.form['textinp']
        image = 'hwimage' + str(time.time()) + '.png'
        savefig(text, 'static/' + image, cdims, cdicts, cgens)
        return render_template("gan_display.html", image=image)
    else:
        return render_template("gan_display.html")

if __name__ == "__main__":
    app.run(debug=True)


