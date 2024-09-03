from Imports import *


# bin_data  = pd.read_csv('dataset/bin_data_test.csv')
# bin_data_test = pd.read_csv('dataset/bin_data_train.csv')


def feat_extract(data):
    data.drop(columns=bin_data.columns[0], axis=1,  inplace=True)

    #extracting one hot encodings for categorical data
    cat_col = ['protocol_type','service','flag']
    categorical = data[cat_col]
    categorical = pd.get_dummies(categorical,columns=cat_col)

    #feature analysis for numerical data
    data.drop(columns= data.columns[44:46], axis=1,  inplace=True)
    numeric_col = data.select_dtypes(include='number').columns

    # creating a dataframe with only numeric attributes of binary class dataset and encoded label attribute 
    numeric_bin_data = bin_data[numeric_col]
    numeric_bin_data['intrusion'] = bin_data['intrusion']

    # finding the attributes which have more than 0.4 correlation with encoded attack label attribute 
    corr= numeric_bin_data.corr(method='spearman')
    corr_y = abs(corr['intrusion'])
    highest_corr = corr_y[corr_y >0.4]
    highest_corr.sort_values(ascending=True)

    #after analysis, selectively picking features
    numeric_bin_data = data[['dst_host_count','count','srv_serror_rate','serror_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
                         'logged_in','dst_host_same_srv_rate','dst_host_srv_count','same_srv_rate']]
     

    # joining the selected attribute with the one-hot-encoded categorical dataframe
    numeric_bin_data = numeric_bin_data.join(categorical)
    # then joining encoded, one-hot-encoded, and original attack label attribute
    data = numeric_bin_data.join(data[['intrusion','abnormal','normal','label']])


    return data


