import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
from PIL import Image
import bz2file as bz2


aaa = []
        
change = {
    'SpiceJet':[0,0,0,0,1,0],
    'AirAsia':[1,0,0,0,0,0],
    'Vistara':[0,0,0,0,0,1],
    'GO_FIRST':[0,0,1,0,0,0],
    'Indigo':[0,0,0,1,0,0],
    'Air_India':[0,1,0,0,0,0],
    'Mumbai':[0,0,0,0,0,1],
    'Bangalore':[1,0,0,0,0,0],
    'Kolkata':[0,0,0,0,1,0],
    'Hyderabad':[0,0,0,1,0,0],
    'Chennai':[0,1,0,0,0,0],
    'Delhi':[0,0,1,0,0,0],
    'Evening':[0,0,1,0,0,0],
    'Early_Morning':[0,1,0,0,0,0],
    'Morning':[0,0,0,0,1,0],
    'Afternoon':[1,0,0,0,0,0],
    'Night':[0,0,0,0,0,1],
    'Late_Night':[0,0,0,1,0,0],
    "Economy":0, 
    "Business":1,
}



def load_model(text):
    data = bz2.BZ2File(text, 'rb')
    data = pickle.load(data)
    return data

@st.cache
def load_df(path):
    df = pd.read_csv(path)
    return df

df = load_df('Clean_Dataset.csv')

def main_page():
    st.markdown("# EDA 🎈")
    code = '''def hello():
    print("Hello, Streamlit!")'''
    st.code(code, language='python')

    
    # fig = px.bar(df, y='price', x='airline', text_auto='.2s',
    #         title="Default: various text sizes, positions and angles")

    # st.write(fig)

def page2():
    st.markdown("# MODEL ❄️")
    st.success(
        '''
        This is a success message!
        saldjfas;ldfjas;d 
        
        ''', icon="✅")
    image = Image.open('flights.jpeg')
    st.image(image, caption='Sunrise by the mountains')
    # st.write(dataset_test.head(1))
    
    with st.sidebar:
        option = st.selectbox(
        'Selected model',
        ('LinearRegression','XGBRegressor','DecisionsTree','KNeighborsRegressor'))
        if option == 'DecisionsTree':
            model = load_model('DecisionsTree.pbz2')
        # elif option == 'RandomForestRegressor':
        #     model = load_model('RandomForestRegressor.pbz2')
        elif option == 'LinearRegression':
            model = load_model('LinearRegression.pbz2')
        elif option == 'KNeighborsRegressor':
            model = load_model('KNeighborsRegressor.pbz2')
        elif option == 'XGBRegressor':
            model = load_model('XGBRegressor.pbz2')
        else:
            st.warning('no model load')


        alass = st.selectbox(
            "CLASS",
            ("Economy", "Business")
        )
        

        airline = st.selectbox(
            "AIRLINE",
            ('SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo',
       'Air_India')
        )


        destination_city = st.selectbox(
            "Destination city",
            ('Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi')

        )
        source_city = st.selectbox(
            "Source city",
            ('Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi')
        )

        Departure_time = st.selectbox(
            "Departure time ",
            ('Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night',
       'Late_Night')
        )

        Arrival_time = st.selectbox(
            "Arrival time",
            ('Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening',
       'Late_Night')       
        )

        time  = st.number_input('TIME') ## durations

        durations = st.slider('days_left', 0, 100, 10) #day
 
        aaa.append(change[alass])
        aaa.append(time)
        aaa.append(durations)
        aaa.extend(change[airline])
        aaa.extend(change[source_city])
        aaa.extend(change[destination_city])
        aaa.extend(change[Departure_time])
        aaa.extend(change[Arrival_time])



    st.title('Predcited with dataset test')
    
    
    
    answer2 = np.array(aaa)
    answer2 = answer2.reshape(1,-1)
    st.info(f'Predict v2 : {model.predict(answer2)[0]:.2f}')
    

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "EDA": main_page,
    "MODEL AND TEST DATA": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

