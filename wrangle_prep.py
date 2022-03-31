import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing

from env import user, password, host
import warnings
warnings.filterwarnings('ignore')


def wrangle_zillow():
    ''' 
    This function pulls data from the zillow database from SQL and cleans the data up by changing the column names and romoving rows with null values.  
    Also changes 'fips' and 'year_built' columns into object data types instead of floats, since they are more catergorical, after which the dataframe is saved to a .csv.
    If this has already been done, the function will just pull from the zillow.csv
    '''

    filename = 'zillow.csv'
    
    if os.path.exists(filename):
        print('Reading cleaned data from csv file...')
        return pd.read_csv(filename)

    url = f"mysql+pymysql://{user}:{password}@{host}/zillow"

    query = '''
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, fips
        FROM properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusedesc IN ('Single Family Residential', 'Inferred Single Family Residential')
        '''

    df = pd.read_sql(query, url)

    # Rename columns
    df = df.rename(columns = {
                          'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'sqr_feet',
                          'taxvaluedollarcnt':'tax_value', 
                          'yearbuilt':'year_built'})
    
    # Drop null values
    df = df.dropna()

    # Remove Outliers
    df = df[df.tax_value < 847733.0]
    df = df[df.sqr_feet < 50000.0]
    df= df[df.bedrooms > 0.0]
    df = df[df.year_built > 1950]
    
    # Change the dtype of 'year_built' and 'fips'
    # First as int to get rid of '.0'
    df.year_built = df.year_built.astype(int)
    df.fips = df.fips.astype(int)
    
    # Then as object for categorical sorting
    df.year_built = df.year_built.astype(object)
    df.fips = df.fips.astype(object)

    # Download cleaned data to a .csv
    df.to_csv(filename, index=False)

    return df


def split_data(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=1313)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=1313)

    # Take a look at your split datasets

    print(f'train <> {train.shape}')
    print(f'validate <> {validate.shape}')
    print(f'test <> {test.shape}')
    return train, validate, test



def scale_data(train, validate, test, scaler, return_scaler=False):
    '''
    This function takes in train, validate, and test dataframes and returns a scaled copy of each.
    If return_scaler=True, the scaler object will be returned as well
    '''
    
    num_columns = ['bedrooms', 'bathrooms', 'sqr_feet', 'tax_value', 'taxamount']
    
    train_scaled = train.copy()
    validated_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler.fit(train[num_columns])
    
    train_scaled[num_columns] = scaler.transform(train[num_columns])
    validate_scaled[num_columns] = scaler.transform(validate[num_columns])
    test_scaled[num_columns] = scaler.transform(test[num_columns])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

    
    
