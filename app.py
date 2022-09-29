import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
from PIL import Image
import bz2file as bz2
from plotly.subplots import make_subplots
import plotly.graph_objs as go



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

st.set_page_config(
    page_title="EDA AND PREDICT",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def load_model(text):
    data = bz2.BZ2File(text, 'rb')
    data = pickle.load(data)
    return data

@st.cache
def load_df(path):
    df = pd.read_csv(path,index_col=0)
    return df

df = load_df('Clean_Dataset.csv')
df2 = load_df('ml.csv')
df3 = load_df('sample.csv')

def main_page():
    st.markdown("# EDA 🎈")
    st.write('Featurn enginering')
    code = '''def load_df(path):
    df = pd.read_csv(path,index_col=0)
    return df'''
    st.code(code, language='python')
    st.write(df.head(5))
    st.write(df.shape)
    st.title('Histogram')
    fig1 = px.histogram(df, x="airline", y="price",
             color='class', barmode='group',
             height=500)
    st.write(fig1)

    df_temp = df.groupby(['days_left'])['price'].mean().reset_index()
    fig2 = px.scatter(df_temp, x="days_left", y="price", 
                 trendline="ols", trendline_options=dict(log_x=True),
                 title="Average prizes depending on the days left",
                 trendline_color_override = 'red')
    st.title('experiment data with ols regressions')
    st.write(fig2)
    st.write('''
    
    prices rise slowly and then drastically start rising 20 days
    before the flight, but fall just one day before the flight up to three times cheaper.
    This can be explain by the fact the companies want to fill their empty seats 
    and thus lower the prices of the tickets to ensure the planes remains full.
        
    ''')


    fig5 = px.box(df, y="price", x="departure_time")
    trace0 = go.Box(
        y=df['price'],
        x=df['departure_time'],
        name='Airline prices based on the departure time'    
    )
    trace1 = go.Box(
    y=df['price'],
    x=df['arrival_time'],
    name='Airline prices based on the arrival time'    
    )
    fig = make_subplots(rows=1, cols=2)
    fig.append_trace(trace0, row = 1, col = 1)
    fig.append_trace(trace1, row = 1, col = 2)
    fig.update_layout(
    title='Arrival time and departure time',
    autosize=False,
    width=1000,
    height=500,)
    st.title('Find different between arrival time and  departure time')
    st.write(fig)
    st.write('**Price arrival time early_morning too cheap opposite departure time**')
    # fig = px.bar(df, y='price', x='airline', text_auto='.2s',
    #         title="Default: various text sizes, positions and angles")

    # st.write(fig)

    st.title('Medthod selected feature')
    st.markdown('**Mutual information**')
    with st.expander("mutual information meaning"):
            st.write("""
                The Mutual Information between two random variables measures non-linear relations between them. Besides, it indicates how much information can be obtained from a random variable by observing another random variable.
            """)

    code2 = '''
    #mutual information
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    
    '''

    tab1, tab2 = st.tabs(["Code", "Score ml"])

    with tab1:
        st.code(code2, language='python')

    with tab2:
        st.table(df2)

    # with tab3:
    #     st.header("An owl")
    #     st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


    # col1, col2= st.columns(2)

    # with col1:

    #     st.write(df2)

    # with col2:
    #     st.code(code2, language='python')

    st.title('Preprocess data')
    st.markdown("""
            Encode variables and dummies variables
    """)
    code3 = """
    Encode variables
    df["class"] = df["class"].replace({'Economy':0,'Business':1}).astype(int)
    dummies_variables = ["airline","source_city","destination_city","departure_time","arrival_time"]
    dummies = pd.get_dummies(df[dummies_variables])
    df = pd.concat([df,dummies],axis=1)
    """

    tab3, tab4 = st.tabs(["Code", "After Encode"])

    with tab3:
        st.code(code3, language='python')

    with tab4:
        st.table(df3)

    st.title('List Model ')
    with st.expander("Selected model"):
        st.write("""
        🔸 KNeighborsRegressor\n
        🔸 LinearRegression\n
        🔸 XGBRegressor\n
        🔸 Decisions Tree\n
        """)
    st.write('**😄 You can try the model on another page.😄**')
def page2():

    st.markdown("# MODEL ❄️")
    with st.expander("Infomatios Feature meaning"):
        st.write("""
        🔸 Airline: The name of the airline company is stored in the airline column. It is a categorical feature having 6 different airlines.\n
        🔸 Flight: Flight stores information regarding the plane's flight code. It is a categorical feature.\n
        🔸 Source City: City from which the flight takes off. It is a categorical feature having 6 unique cities.\n
        🔸 Departure Time: This is a derived categorical feature obtained created by grouping time periods into bins. It stores information about the departure time and have 6 unique time labels.\n
        🔸 Arrival Time: This is a derived categorical feature created by grouping time intervals into bins. It has six distinct time labels and keeps information about the arrival time.\n
        🔸 Destination City: City where the flight will land. It is a categorical feature having 6 unique cities.\n
        🔸 Class: A categorical feature that contains information on seat class; it has two distinct values: Business and Economy.\n
        🔸 Duration: A continuous feature that displays the overall amount of time it takes to travel between cities in hours.\n
        🔸 Days Left: This is a derived characteristic that is calculated by subtracting the trip date by the booking date.\n
        🔴 Price: Target variable stores information of the ticket price.
        """)
    image = Image.open('flights.jpeg')
    st.image(image, caption='Price dependant on feature')
    
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

        durations = st.slider('Days_Left', 0, 50, 10) #day
 
        aaa.append(change[alass])
        aaa.append(time)
        aaa.append(durations)
        aaa.extend(change[airline])
        aaa.extend(change[source_city])
        aaa.extend(change[destination_city])
        aaa.extend(change[Departure_time])
        aaa.extend(change[Arrival_time])

    st.title('R squared score')
    with st.expander("R squared maening"):
        st.write("""
            R-squared (R2) is a statistical measure that represents the proportion of the 
            variance for a dependent variable that's explained by an independent variable or variables in a regression model.
        """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DecisionsTree",'97%')
    col2.metric("LinearRegression","90%")
    col3.metric("KNeighborsRegressor","67%")
    col4.metric("XGBRegressor","98%")
    
    answer2 = np.array(aaa)
    answer2 = answer2.reshape(1,-1)
    st.success(f'Ticket price  : {model.predict(answer2)[0]:.2f}')
    

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "EDA": main_page,
    "MODEL AND PREDICT": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

