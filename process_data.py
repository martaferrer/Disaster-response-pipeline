'''
The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a
SQLite database in the specified database file path.

The first part of your data pipeline is the Extract, Transform, and Load process:
 - loads and merges the messages and categories datasets
 - cleans the dataset
 - splits the categories column into separate, clearly named columns, converts values to binary
 - drops duplicates
 - stores the output in a SQLite database
'''

# We expect you to do the data cleaning with pandas. To load the data into an SQLite database, you can use the pandas
# dataframe .to_sql() method, which you can use with an SQLAlchemy engine.
# Feel free to do some exploratory data analysis in order to figure out how you want to clean the data set.


