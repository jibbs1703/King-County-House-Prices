import pandas as pd

def file_reader(path):
    '''This function locates a csv file, loads it and returns a dataframe. The function's
    input is either a direct link to the CSV file or a path directory to the CSV file on a local machine.

    Input : CSV filepath
    Output: DataFrame

    '''

    # Import Dataframe from Filepath
    data = pd.read_csv(path, na_values=["nan", "n.a", "not available", "?", "NaN"])

    # Return Imported Pandas DataFrame
    return data