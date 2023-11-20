#!/usr/bin/env python
# coding: utf-8

# In[3]:


import io
import os
import numpy as np
import json, random
from flask import request
from flask import Response
from markupsafe import escape
from flask import Flask, render_template
import numpy as np
import pickle

os.chdir(r"C:\Users\Renu\Downloads\P302\deployment")
app = Flask(__name__)

#Define the route to be home. 

@app.route('/')
def home():
    return render_template('classification.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /result page with the output
@app.route('/result',methods=['POST'])
def result():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = np.array(int_features).reshape(1,-1)  #Convert to the form [[a, b]] for input to the model
    model = pickle.load(open('models\model.pkl','rb'))
    prediction = model.predict(features)[0]

    return render_template("classification.html", prediction=prediction)


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run(debug = True)


# In[4]:


get_ipython().run_line_magic('tb', '')


# In[ ]:




