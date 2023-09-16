#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import os
import imageio
import base64
import numpy as np
import json, random
from flask import request
from flask import Response
from markupsafe import escape
from flask import Flask, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from final_model_forecast import final_model

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def index():
    return render_template("forecasting.html")

@app.route('/forecast',methods=['Post'])
def forecast():

    forecasting_steps = [x for x in request.form.values()] 
    for n in forecasting_steps:
        i = int(n)
        df = final_model(i)['my_df']
        df.columns = ['forecasted visitors']
        fig = px.line(df, x = df.index ,y = 'forecasted visitors')
        fig.update_layout(xaxis_title=None)
        fig.update_layout(autosize=False,
                          width=500,height=200,
                          margin=dict(l=20, r=20, t=20, b=20),
                          paper_bgcolor="LightSteelBlue",)
        dat = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('forecasting.html', data=dat)

if __name__ == '__main__':
    app.run(debug = True)


# In[ ]:




