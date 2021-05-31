from flask import Flask,render_template,request,jsonify
import numpy as np 
import pickle as p 
import pandas as pd 
import json 
from sklearn import svm

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/algolist")
def show_list():
    return render_template("algolist.html")

@app.route("/showform/<number>")
def show_form(number):
    return render_template("form.html",id=number)

@app.route("/predict/<id>",methods=["POST"])
def predict(id):
    d=request.form.to_dict()
    for i in d:
        print(i)
        d[i]=float(d[i])

    if(id=="10"):
        remove=["litr","sext","scp","popd","polda","poldp"]
        for k in remove:
            d.pop(k,None)
        d["rpop"]=(d["rpop"]-43.18)/(85.39-43.18)
        d["upop"]=(d["upop"]-14.61)/(56.82-14.61)
        d["stp"]=(d["stp"]-1.24)/(19.03-1.24)
        d["marg"]=(d["marg"]-7.24)/(22.4-7.24)
        d["main"]=(d["main"]-77.6)/(92.76-77.6)
        print(d)
        modelfile = 'models/prediction.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        prediction=float(prediction)*30+2
        return jsonify(prediction)
    
    if(id=="11"):
        modelfile = 'models/svmprediction.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T)[0]
        return jsonify(prediction)
    if(id=="12"):
        remove=["sext","polda","poldp","scp","popd"]
        for k in remove:
            d.pop(k,None)
        modelfile = 'models/GLMprediction.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        return jsonify(prediction)
        

    #print(d)
    
if __name__ == '__main__': 
    app.run(debug=True) 