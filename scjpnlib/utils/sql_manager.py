import pandas as pd
import sqlite3 as sql3
from pandasql import sqldf

def pysqldf(query_string):
    return sqldf(query_string, globals())

class SQLiteDBManager:
    def __init__(self, db_fn):
        self.db_fn = db_fn
        self.conn = sql3.connect(db_fn)
        
    def get_db_fn(self):
        return self.db_fn

    def get_conn(self):
        return self.conn

    def new_cursor(self):
        return self.conn.cursor()

    def sql_query_to_df(self, query_string):
        c = self.new_cursor()
        c.execute(query_string)
        cols = [x[0] for x in c.description]
        result = c.fetchall()
        df = pd.DataFrame(result, columns=cols)
        return df

class PySQLDFManager:
    def __init__(self, globals):
        self.globals = globals

    def pysqldf(self, query_string):
        return sqldf(query_string, self.globals)