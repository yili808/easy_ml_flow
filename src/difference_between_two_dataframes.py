"""
Identify difference between two dataframes
"""

import numpy as np
import pandas as pd

def diff_in_2dfs(df1, df2):
    """
    Identify difference between two dataframes.
    
    Reference: https://github.com/pandas-dev/pandas/blob/master/pandas/util/testing.py
               from pandas.util.testing import assert_frame_equal
    
    Step1: identify the difference in columns
    Step2: identify the difference the rows/indices
    Step3: identify the difference in values based on the common columns and common rows 
    
    Parameters
    ----------
    df1: pandas dataframe
        Input dataframe.
    
    df2: pandas dataframe
        Input dataframe.
    
    Returns
    -------
    common_df_diff : pandas dataframe
        The difference between the common part of df1 and df2.
    """
    # Step1: identify the difference in columns
    print("Identifying the difference in columns...")
    common_cols = set(df1) & set(df2)
    common_cols_score = 0.5*(len(common_cols)/df1.shape[1] + len(common_cols)/df2.shape[1])
    print("Num of common columns: {}, score: {}".format(len(common_cols), common_cols_score))
    if common_cols_score == 0:
        raise ValueError("Two dataframes' columns not match at all!")
    elif common_cols_score < 1:
        print("Columns in df1 but not in df2 (count: {}): {}".format(len(set(df1) - set(df2)), set(df1) - set(df2)))
        print("Columns in df2 but not in df1 (count: {}): {}".format(len(set(df2) - set(df1)), set(df2) - set(df1)))
        
    # Step2: identify the difference the rows/indices
    print("\nIdentifying the difference in rows/indices...")
    common_rows = set(df1.index) & set(df2.index)
    common_rows_score = 0.5*(len(common_rows)/df1.shape[0] + len(common_rows)/df2.shape[0])
    print("Num of common rows: {}, score: {}".format(len(common_rows), common_rows_score))
    if common_rows_score == 0:
        raise ValueError("Two dataframes' rows not match at all!")
    elif common_rows_score < 1:
        print("Rows in df1 but not in df2 (count: {}): {}".format(len(set(df1.index) - set(df2.index)), set(df1.index) - set(df2.index)))
        print("Rows in df2 but not in df1 (count: {}): {}".format(len(set(df2.index) - set(df1.index)), set(df2.index) - set(df1.index)))
    
    # Step3: identify the difference in values based on the common columns and common rows
    print("\nIdentifying the difference in values based on the common columns and common rows...")
    common_df1 = df1.loc[common_rows, common_cols]
    common_df2 = df2.loc[common_rows, common_cols]
    if common_df1.equals(common_df2):
        print("Two dataframes are 100% the same on common rows and columns.")
        return None
    equal_bool = np.isclose(common_df1, common_df2, equal_nan=True) # tolerate small difference
    similarity_score = 1.0 * equal_bool.sum() / (common_df1.shape[0] * common_df1.shape[1])
    print("Two dataframes are {:2f} % the same on common rows and columns.".format(100 * similarity_score))
    common_df_diff = pd.concat([common_df1.mask(equal_bool).stack(), common_df2.mask(equal_bool).stack()], axis=1)
    common_df_diff.columns=["df1", "df2"]
    return common_df_diff
