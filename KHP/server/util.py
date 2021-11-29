import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_location_names():
    return __locations

def get_estimated_price(location, sqft, bedrooms, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bath
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    result = __model.predict([x])[0]
    result_1 = result + (0.35 * result)
    return round(result_1 / 100000, 2)

def load_saved_artifacts():
    print("Loading saved artifacts....start")
    global __data_columns
    global __locations
    global __model

    with open('./artifacts/columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open('./artifacts/karachi_home_prices_model.pickle','rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts....done")


if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Nazimabad', 1800, 4, 3))
    print(get_estimated_price('Scheme 33', 1080, 3, 2))
    print(get_estimated_price('DHA Defence', 9000, 7, 6))