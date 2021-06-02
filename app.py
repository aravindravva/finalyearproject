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

@app.route("/visualize")
def show():
    return render_template("visualizations.html")


@app.route("/crimeprediction")
def show_options():
    return render_template("algolistwithcrime.html")

@app.route("/foliumcontrol",methods=["POST"])
def show_maps():
    crime=request.form.get("crime")
    year=request.form.get("year")
    file_name="folium/"+crime+"/"+crime+"_"+year+".html"
    return render_template(file_name)
    

@app.route("/showform",methods=["POST"])
def show_form():
    crime=request.form.get("crime")
    algorithm=request.form.get("algo")
    #algorithm== mlr,forest,glm
    #crime== rape murder robbery
    if(crime=="rape" and algorithm=="mlr"):
        return render_template("form.html",id="11",algo="Predicting Rape - MLR")
    elif(crime=="rape" and algorithm=="forest"):
        return render_template("form.html",id="12",algo="Predicting Rape - RFR")
    elif(crime=="rape" and algorithm=="glm"):
        return render_template("form.html",id="13",algo="Predicting Rape - GLM")
    elif(crime=="murder" and algorithm=="mlr"):
        return render_template("form.html",id="21",algo="Predicting Murder - MLR")
    elif(crime=="murder" and algorithm=="forest"):
        return render_template("form.html",id="22",algo="Predicting Murder - RFR")
    elif(crime=="murder" and algorithm=="glm"):
        return render_template("form.html",id="23",algo="Predicting Murder - GLM")
    elif(crime=="robbery" and algorithm=="mlr"):
        return render_template("form.html",id="31",algo="Predicting Robbery - MLR")
    elif(crime=="robbery" and algorithm=="forest"):
        return render_template("form.html",id="32",algo="Predicting Robbery - RFR")
    elif(crime=="robbery" and algorithm=="glm"):
        return render_template("form.html",id="33",algo="Predicting Rape - GLM")
    


@app.route("/predict/<id>",methods=["POST"])
def predict(id):
    d=request.form.to_dict()
    for i in d:
        d[i]=float(d[i])

    if(id=="11"):
        remove=["litr","sext","scp","popd","polda","poldp"]
        for k in remove:
            d.pop(k,None)
        d["rpop"]=(d["rpop"]-43.18)/(85.39-43.18)
        d["upop"]=(d["upop"]-14.61)/(56.82-14.61)
        d["stp"]=(d["stp"]-1.24)/(19.03-1.24)
        d["marg"]=(d["marg"]-7.24)/(22.4-7.24)
        d["main"]=(d["main"]-77.6)/(92.76-77.6)
        print(d)
        modelfile = 'models/rape/MLRpredictionrape.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        prediction=float(prediction)*30+2 #scaling 
        prediction=round(prediction,2)
        return render_template("result.html",result=prediction,crime="rape",algo="Predicting Rape - MLR")
    
    if(id=="12"):
        modelfile = 'models/rape/forestpredictionrape.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T)[0]
        prediction=round(float(prediction),2)
        return render_template("result.html",result=prediction,crime="rape",algo="Predicting Rape - RFR")
    
    if(id=="13"):
        remove=["sext","polda","poldp"]
        for k in remove:
            d.pop(k,None)
        modelfile = 'models/rape/GLMpredictionrape.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        prediction=round(float(prediction),2)
        return render_template("result.html",result=prediction,crime="rape",algo="Predicting Rape - GLM")
    
    if(id=="21"):
        remove=["marg","main","polda","poldp","scp","litr"]
        for k in remove:
            d.pop(k,None)
        modelfile = 'models/murder/MLRpredictionmurder.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        prediction=round(float(prediction),2)
        return render_template("result.html",result=prediction,crime="murder",algo="Predicting Murder - MLR")

    if(id=="22"):
        modelfile = 'models/murder/forestpredictionmurder.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T)[0]
        prediction=prediction/1.05
        prediction=round(float(prediction),2)
        return render_template("result.html",result=prediction,crime="murder",algo="Predicting Murder - RFR")
    
    if(id=="23"):
        remove=["poldp","marg","main","litr","polda"]
        for k in remove:
            d.pop(k,None)
        modelfile = 'models/murder/GLMmurderprediction.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        prediction=round(float(prediction),2)
        return render_template("result.html",result=prediction,crime="murder",algo="Predicting Murder - GLM")
    
    if(id=="31"):
        remove=["marg","sext","poldp","litr","popd"]
        for k in remove:
            d.pop(k,None)
        modelfile = 'models/robbery/MLRpredictionmurder.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        prediction=round(float(prediction),2)
        return render_template("result.html",crime="robbery",result=prediction,algo="Predicting Robbery - MLR")
    
    if(id=="32"):
        modelfile = 'models/robbery/forestpredictionrobbery.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T)[0]
        prediction=round(float(prediction),2)
        return render_template("result.html",crime="robbery",result=prediction,algo="Predicting Robbery - RFR")

    
    if(id=="33"):
        remove=["sext","marg","polda"]
        for k in remove:
            d.pop(k,None)
        modelfile = 'models/robbery/GLMpredictionrobbery.pickle' 
        model = p.load(open(modelfile, 'rb')) 
        prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
        prediction=round(float(prediction),2)
        return render_template("result.html",crime="robbery",result=prediction,algo="Predicting Robbery - GLM")


    	
        

    
    
if __name__ == '__main__': 
    app.run(debug=True) 
