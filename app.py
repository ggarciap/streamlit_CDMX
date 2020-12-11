from streamlit.hashing import _CodeHasher
import pandas as pd
import streamlit as st
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import folium 
import altair as alt
import pydeck as pdk
import streamlit as st
from streamlit_folium import folium_static
from folium import plugins
from shapely.geometry import Point, Polygon
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import geopandas as gpd
import pandas as pd
import geopy
import cv2
import numpy as np


st.set_page_config(
    page_title="GGP | Classification",
    initial_sidebar_state='expanded',
    page_icon=":flag-mx:"
)

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

lat_cdmx = 19.432608
lon_cdmx = -99.133209

def haversine_distance(geo_one, geo_two):
    g1 = geo_one.split(',')
    g2 = geo_two.split(',')
    lat1 = float(g1[0])
    lat2 = float(g2[0])
    lon1 = float(g1[1])
    lon2 = float(g2[1])
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)


def quick_dataframe(df, max_rows=1000, **st_dataframe_kwargs):
    """Display a subset  by taking a boostrap sample of the original
       DataFrame to speed up app renders.
    
    Parameters
    ----------
    df : DataFrame | ndarray
        The DataFrame or NumpyArray to render.
    max_rows : int
        The number of rows to display.
    st_dataframe_kwargs : Dict[Any, Any]
        Keyword arguments to the st.dataframe method.
    """
    n_rows = len(df)
    if n_rows <= max_rows:
        # As a special case, display small dataframe directly.
        # st.write(df)
        percentage  = max_rows/df.shape[0]
    else:
        # Take bootstrap sample of original dataframe
        percentage  = max_rows/df.shape[0]
        print(percentage)
        df = df.sample(frac=percentage, replace=True, random_state=42)
        


def main():
    state = _get_state()
    pages = {
        "About": page_about,
        "EDA": page_eda,
        "FE+Model": page_fe_pred,
        "Predictor": page_model_pred,
        "Resources" : page_resources,
    }

    st.sidebar.title(":book: Content")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def page_about(state):
    st.subheader('About this project')



    img = cv2.imread("./images/Mexico-City-2014-City-Clock-7.jpg",0)
    st.image(img, width =700)

    st.markdown(
        '''
        Mexico city is considered to be in the top 10 largest cities in the word. As pointed out by statista, "the metrobus system of the Mexican capital transported nearly 404 million passengers in the first quarter of 2019, up from 398.5 million passengers in the same period of the previous year." The large number of affluence in all the public transport services creates an opportunity for the rate of crimes to grow, and at the same time it makes it difficult for authorities to narrow down which types of crimes are more likely to happen. 

        The purpose of this project is to explore the Mexico city database wich contains numerous datasets of the different insitutions of the local govermennt. In this case I decided to focus on the the folders of investigation of crimes at street level of the Attorney General's Office of Mexico City from January 2016 to June 2019 [(source)](https://datos.cdmx.gob.mx/explore/dataset/carpetas-de-investigacion-pgj-cdmx/information/). In addition, the model presented takes into consideration the dataset of daily influx by metrobus.

        This projects presents a model that is inteded to perform better than a baseline model createdy with sklearn `dummyClassifer`. An `XGBoost` multiclass classifier will predict the categorization of a new possible robbery based on the affluence, distance from the metrobus stations and other geographical attributes of the crime. In addition, this model seeks to perform well on the weighted average. 
        '''
    )

@st.cache
def load_model_df():
    df = pd.read_csv('./data/master_modeling_years-2.csv')
    df['fecha_hechos'] = pd.to_datetime(df['fecha_hechos'])
    quick_dataframe(df)
    return df
@st.cache
def load_mb_df():
    df = pd.read_csv('./data/estaciones-metrobus0.csv')
    df['lat'] = df['Geo Point'].apply(lambda x: float(x.split(',')[0]))
    df['long'] = df['Geo Point'].apply(lambda x: float(x.split(',')[1]))
    quick_dataframe(df)
    return df
@st.cache
def load_preds_df():
    df = pd.read_csv('./data/master_modeling_years-predictions.csv')
    return df

def page_eda(state):
    st.subheader('Project EDA')
    stations_mb = load_mb_df()
    master_df = load_model_df()

    st.markdown(
            '''
            The metorbus stations are located throughout the metropolitan area of Mexico City. This public transpotation offers 7 lines covering around 127km and allows users to reach all areas of the city. In the following map the 283 stations are prsented. 
            '''
    ) 

    with st.spinner(f"Loading MB Visualization ..."):


        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=19.432608,
                longitude=-99.133209,
                zoom=11,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=stations_mb, 
                    get_position='[long, lat]',
                    get_color='[180, 0, 200, 140]', 
                    get_radius= 100,
                ),
            ]
        ))

    st.markdown(
        '''
        For this particular project all the crimes that occurred in a distance (km) range of [0, .3] form the nearest metrobus station were considered to be significant for modeling porpuses. The following plot presents all the  crimes considered to be close enough to a metrobus station be included in futher stages of the project. 
        '''
    )

    code_dist = '''
    master_df['dist_km'] = master_df['dist_km'].apply(lambda x: x if x <= .3 else None)
    master_df = master_df.dropna()
    '''

    st.code(code_dist, language='python')

    with st.spinner(f"Loading MB Visualization ..."):
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=19.432608,
                longitude=-99.133209,
                zoom=11,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=stations_mb, 
                    get_position='[long, lat]',
                    get_color='[180, 0, 200, 140]', 
                    get_radius= 150,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=master_df, 
                    get_position='[longitud, latitud]',
                    get_color='[8, 104, 172]', 
                    get_radius= 40,
                ),
                
            ],
        ))


    st.markdown(
        '''
        One important aspect of this project was to determine which crimes had a bigger impact on the population. I decided to keep the crimes which represented more that 10% the totality of all observation of crimes committed from 2016 to 2019  and create a multiclassification from those categories of crime. 
        '''
    )
    code_presentation =  '''
    def significance_filter(zip_element):
        significant = []
        not_significant = []
        for i in zip_element:
            if i[1] >=.1:
                significant.append(i[0])
            else:
                not_significant.append(i[0])
        return significant, not_significant
    '''
    st.code(code_presentation, language='python')
    st.markdown(
        '''
        The crimes categories that were considered in the model can be seen in orange in the barchart. These categories represent almost 70% of all the observations in the dataset, also they tend to be the crimes that daily users of the metrobus tend to be worried about during their commutes.   
        '''
    )
    
    s = (pd.value_counts(master_df['delito']) / len(master_df['delito']))
    data = list(zip(s,s.index))[:20]
    tmp = pd.DataFrame(data, columns =['category_proportion', 'Crime']) 
    tmp['cat'] = tmp['category_proportion'].apply(lambda x:'sig' if x > .1 else 'n_sig')
    prop_chart = alt.Chart(tmp).mark_bar().encode(
        x='category_proportion',
        y=alt.Y('Crime', sort='-x'),
        color=alt.Color('cat', legend=None),
        tooltip=['Crime','category_proportion']
    ).properties(
        height=500,
        width=500).interactive()
    st.altair_chart(prop_chart)

    st.subheader('Project Challenges and Limitaitons')

    st.markdown(
        '''
        One of the main challanges for this project was to explore several ways to try to create new features that could help in classifitying observations that tend to be geographically close from each other. The following map presents the clusters of the subset of crimes that were considered to be significant. Doing further analysis on any of the highly dense clusters it is clear that by just using coordinates without any feature engieenering it can make it hard for a model to correclty classify the crimes of new test set.  
        '''
    )





    with st.spinner(f"Loading Cluster ..."):
        cdmx_map=folium.Map(
            location=[lat_cdmx, lon_cdmx],
            zoom_start=12
        )

        # instantiate a mark cluster object for the incidents in the dataframe
        incidents = plugins.MarkerCluster().add_to(cdmx_map)

        # loop through the dataframe and add each data point to the mark cluster
        for lat, lon, label in zip(master_df['latitud'], master_df['longitud'], master_df['delito']):
            folium.Marker(
                location=[lat, lon],
                popup=label,
                icon=None
            ).add_to(incidents)

        folium_static(cdmx_map)
    
    st.markdown(
        '''
        As seen in the jointplot there are high density areas where most of the economic and social activity happens in Mexico City. People usually commute to the high density areas from others parts of the city. This also creates another layer of difficulty due to the fact that by nature most of the observations are going to be concentrated in certain points. 
        '''
    )


    fig, ax = plt.subplots()
    ax = sns.jointplot(data=master_df, x='longitud', y='latitud', kind='hex', height=20 )
    st.pyplot(ax)
    
def page_fe_pred(state):
    st.subheader("Feature Engineering: `nearest_mb`")
    st.markdown(
        '''
        As mentioned in the exploratory data analysis of the project, due to the fact that the a large proportion of the crimes are close geographically it makes it harder to find potential ways to differentiate the observations. A simple K-NN model was used for creating the first engineered feature for determining the closest MB station depending the location of the crime. This feature is present in the `master_df.csv` feature as `nearest_mb`.

        '''
    )

    st.subheader("Feature Engineering: `dist_km` and `manhattan_dist` ")
    st.markdown(
        '''
        The master dataset that was created by combining different datasets from the Mexico City's open databases allows us to have the geolocations of the crime and the nearest metrobus. These features are used for creating engineering features that represent the distance between both points. 

        The following code block presents the functions used for creating these new features. The following medium post was used as reference for creating the functions methods [(source)](https://maney020.medium.com/feature-engineering-all-i-learned-about-geo-spatial-features-649871d16796).  
        '''
    )

    code_dist = '''
        def haversine_dist(lat1,lng1,lat2,lng2):
            lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
            radius = 6371  # Earth's radius taken from google
            lat = lat2 - lat1
            lng = lng2 - lng1
            d = np.sin(lat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng/2) ** 2
            h = 2 * radius * np.arcsin(np.sqrt(d))
            return h


        def manhattan_dist(geo_one, geo_two):
            g1 = geo_one.split(',')
            g2 = geo_two.split(',')
            lat1 = float(g1[0])
            lat2 = float(g2[0])
            lng1 = float(g1[1])
            lng2 = float(g2[1])
            a = haversine_dist(lat1, lng1, lat1, lng2)
            b = haversine_dist(lat1, lng1, lat2, lng1)
            return a + b
    
    '''

    st.code(code_dist, language='python')

    st.subheader("Feature Engineering: `bearing`")

    st.markdown(
        '''
        Still using the geolocation of both the crime and the corresponding closest metrobus station the bearing can be calculated as presented in the following code snippet. In this way we can have a metric that describes direction.   
        '''
    )

    code_bearing = '''
        def bearing_array(geo_one, geo_two):
            g1 = geo_one.split(',')
            g2 = geo_two.split(',')
            lat1 = float(g1[0])
            lat2 = float(g2[0])
            lng1 = float(g1[1])
            lng2 = float(g2[1])
            AVG_EARTH_RADIUS = 6371  # in km
            lng_delta_rad = np.radians(lng2 - lng1)
            lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
            y = np.sin(lng_delta_rad) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
            return np.degrees(np.arctan2(y, x))
    '''
    st.code(code_bearing, language='python')

    st.subheader("Feature Engineering: `geo_hash_crime` and `geo_hash_mb`")

    st.markdown(
        '''
        Trying to explore different ways to enclose several observations without adding uncessary complexity to the feature engineeirng process the geocdoe system known as Geohash was used to created two new engineered features. As explained by [wikipedia](https://en.wikipedia.org/wiki/Geohash) Geohash "is a hierarchical spatial data structure which subdivides space into buckets of grid shape". In this way new categorical features can be used for further examining close observations and improving the model performance. 
        '''
    )

    code_geo_h = '''
    def geohash(geo_loc, precision=8):
        g1 = geo_loc.split(',')
        lat1 = float(g1[0])
        lng1 = float(g1[1])
        enc_coords = gh.encode(lat1,lng1, precision)
        return enc_coords
    '''
    st.code(code_geo_h, language='python')

    st.subheader('Model implementation and performance')

    st.markdown(
        '''
        As discussed previoulsy, due to some of the challenges that the dataset presents in terms of having a lot of their observations happening close from each other the performance of the model wasn't overwhelmenly good, but it presents a base of a possible route to imporove the classification of these crimes. 
        '''
    )

    img = cv2.imread("./images/confusion_matrix.png")
    st.image(img, width =300, channels='BGR')

    st.markdown(
        '''
       The model is capable of reaching on the train set mean accuracy of .81, and in the test set a weighted f1-score of .62 and mean accuracy of .61 wich is better than the baseline Classifier (using a most frequent strategy) which has a mean accuracy of .41. Some of the aspects worth highlighting from the XGBoost classification model, is that it is capable of classifiying with type 3 crimes with are theft of objects with recall score of .71 on the test set.
        '''
    )


    img2 = cv2.imread("./images/log_logss.png")
    st.image(img2, width =400, channels='BGR')

    img3 = cv2.imread("./images/class_error.png")
    st.image(img3, width =400, channels='BGR')



def page_model_pred(state):
    state.lat=0
    state.lon=0
    st.subheader('Make your own prediction:')
    state.hour = st.sidebar.number_input("Aprox. Hour of event (24-Hr)", value=0)
    state.street = st.sidebar.text_input("Street", "75 Bay Street")
    state.city = st.sidebar.text_input("City", "Toronto")
    state.province = st.sidebar.text_input("Province", "Ontario")
    state.country = st.sidebar.text_input("Country", "Canada")

    

    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(state.street+", "+state.city+", "+state.province+", "+state.country)



    state.checkbox_coor = st.sidebar.checkbox("Use coordinates", state.checkbox_coor)

    if state.checkbox_coor:
        v1 = 19.435484269438177 
        v2 =-99.13663181202588
        tem_str = str(v1)+','+str(v2)

        #geo_lat = st.sidebar.text_input("Geolocation", tem_str)
        state.geo_input = st.sidebar.text_input("Geolocation", state.geo_input or tem_str)
        lat_input = float(state.geo_input.split(',')[0])
        long_input  = float(state.geo_input.split(',')[1])
        

        location = geolocator.reverse(str(lat_input)+','+str(long_input))
        state.lat = lat_input
        state.lon = long_input
    else:
        state.lat = location.latitude
        state.lon = location.longitude

    map_data = pd.DataFrame({'lat': [state.lat], 'lon': [state.lon]})

    st.map(map_data,zoom=12)
    # if st.sidebar.button("Reset"):
    #     state.clear()
    
    with open('./models/xgboost_pipe.pkl', 'rb') as file:
        xgboost_pipe = pkl.load(file)

    cols_when_model_builds = xgboost_pipe.get_booster().feature_names
    geo_user_input = str(state.lat)+','+str(state.lon)

    state.pred_row = None
    state.keeper_dist = float("inf")
    with st.spinner(f"Classifying event  ..."):
        master_df = load_preds_df()
        
        for index, row in master_df.iterrows():
            geo_temp = str(row['latitud'])+","+str(row['longitud'])
            tempor = haversine_distance(geo_temp, geo_user_input)
            if tempor < state.keeper_dist:
                pred_row = master_df.iloc[index]
                state.keeper_dist = tempor
    if state.keeper_dist > .3:
        st.markdown(':negative_squared_cross_mark: Event not found to be less than or equal to .3 Km from any metrobus station. Try again ...')
    else:
        pred_row= pred_row[cols_when_model_builds]
        pred_row = np.array(pred_row).reshape((1,-1))

        # Need to convert back to pandas data frame becuase the model
        # was originally fit with a df instead of np.array 
        pred_row = pd.DataFrame(pred_row, columns =cols_when_model_builds )
        pred_row['longitud'] = state.lon
        pred_row['latitud'] = state.lat
        pred_row['hechos_hour'] = state.hour

        predicted_value = xgboost_pipe.predict(pred_row)[0]

        classes_target = {3 : 'THEFT OF OBJECTS', 
        2 : 'CELL PHONE THEFT WITHOUT VIOLENCE',
        1 : 'CELL PHONE THEFT WITH VIOLENCE'}

        st.info(f'Predicted crime:  {classes_target[predicted_value]}')

    if st.sidebar.button("Reset"):
        state.clear()


def page_resources(state):
    st.subheader('These are the list of the the resources used for this project: ')
    st.markdown('''
    Datasets:
    - [Portal de datos de la Ciudad de México](https://datos.cdmx.gob.mx/pages/home/)
    
    Medium Posts:
    - [Reverse Geocoding in Python](https://towardsdatascience.com/reverse-geocoding-in-python-a915acf29eb6)
    - [Introducing Streamlit Components](https://medium.com/streamlit/introducing-streamlit-components-d73f2092ae30)
    - [Managing Date, Datetime, and Timestamp in Python/Pandas] (https://deallen7.medium.com/managing-date-datetime-and-timestamp-in-python-pandas-cc9d285302ab)
    - [Boosting with AdaBoost and Gradient Boosting] (https://medium.com/diogo-menezes-borges/boosting-with-adaboost-and-gradient-boosting-9cbab2a1af81)

    Github:
    - [Awesome Streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit)

    Pydeck:
    - [ScreengridLayer — pydeck 0.5.0 documentation](https://deckgl.readthedocs.io/en/latest/gallery/screengrid_layer.html)
    - [ScatterplotLayer — pydeck 0.5.0 documentation](https://deckgl.readthedocs.io/en/latest/gallery/scatterplot_layer.html)

    Other:
    - [A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

    ''')



class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    st.title('Mexico City robberies multi-class classification')
    main()