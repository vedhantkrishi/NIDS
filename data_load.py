from Imports import *

def get_data(bin_data):

    # Assuming `bin_data` is already imported 
    X = bin_data.iloc[:, 0:94]
    Y = bin_data[['intrusion']]

    return X, Y