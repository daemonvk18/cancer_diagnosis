import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go



def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl','rb'))
    scaler = pickle.load(open('model/scaler.pkl','rb'))
    values = np.array(list(input_data.values())).reshape(1,-1)
    scaled_values = scaler.transform(values)
    prediction  = model.predict(scaled_values)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")

    if prediction[0]==0:
        st.write("<span class='diagnosis benign'>BENIGN</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>MALIGNANT</span>",unsafe_allow_html=True)   

    st.write("Pobability of being benign:",model.predict_proba(scaled_values)[0][0])
    st.write("Pobability of being malignant:",model.predict_proba(scaled_values)[0][1])

    st.write("This app can assist the medical professionals in making the diagnosis,but cannot be replaced with the professional diagnosis")     



def get_scaled_values(input_dict):
    # scaler = pickle.load(open('model/scaler.pkl','rb'))
    # #converting the dictionary values to numpy array
    # values = np.array(list(input_dict.values())).reshape(-1,1)
    # scaled_values = scaler.fit_transform(values)
    # scaled_data = {k:scaled_values[i][0] for i ,(k,v) in enumerate(input_dict.items())}
    # return scaled_data

    data = pickle.load(open('model/data.pkl','rb'))
    X = data.drop(['diagnosis'], axis=1)
  
    scaled_dict = {}
  
    for key, value in input_dict.items():
       max_val = X[key].max()
       min_val = X[key].min()
       scaled_value = (value - min_val) / (max_val - min_val)
       scaled_dict[key] = scaled_value
  
    return scaled_dict



def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius','Texture','Perimeter',
              'Area', 'Smoothness','Compactness','Concavity','Concave points','Symmetry','Fractal dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']],
      theta=categories,
      fill='toself',
      name='Mean value'
))
    fig.add_trace(go.Scatterpolar(
      r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']],
      theta=categories,
      fill='toself',
      name='Standard error value'
))
    
    fig.add_trace(go.Scatterpolar(
      r=[ input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']],
      theta=categories,
      fill='toself',
      name='Worst value'
))

    fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True
)

    return fig



def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")

    data = pickle.load(open('model/data.pkl','rb'))


    slider_labels = [("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),]
    
    #dictionary for storing the values of the slider
    slider_values = {}
    
    #loop through the slider_labels
    for label,key in slider_labels:
        slider_values[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return slider_values    


def main():
    st.set_page_config(
         page_title='Breast Cancer Diagnosis',
         page_icon="",
         layout='wide',
         initial_sidebar_state='expanded'
    )

    with open('assets/styles.css') as file:
        st.markdown("<style>{}</style>".format(file.read()),unsafe_allow_html=True)

    input_data = add_sidebar()


    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("Our app provides accurate diagnostic feedback and offers educational resources on breast health.This app predicts using a machine learning model whether a breast mass is benign or malignant.Connect the app to a cytology lab for comprehensive cancer prediction and analysis. Users can update measurements easily using sliders in the sidebar.")

    col1,col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)    


    


if __name__ == "__main__":
    main()