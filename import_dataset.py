from os.path import join
import pandas as pd

def prepare_data() -> tuple:
    
    df = pd.read_csv(join('data', 'TS.CF.N2.30yr.csv'))
    df_country = pd.read_csv(join('data', 'EMHIRESPV_TSh_CF_Country_19862015.csv'))

    # Create hourly DateTime column
    t = pd.date_range('1/1/1986', periods = len(df.index), freq = 'h')

    df['Hour'] = t
    df_country['Hour'] = t

    df = df.set_index('Hour')
    df_country = df_country.set_index('Hour')

    return df, df_country

if __name__ == '__main__':
    
    df, df_country = prepare_data()