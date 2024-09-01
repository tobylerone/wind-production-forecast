from os.path import join
import pandas as pd
import constants

def prepare_data() -> tuple:
    
    df = pd.read_csv(join('data', 'TS.CF.N2.30yr.csv'))
    df_country = pd.read_csv(join('data', 'EMHIRESPV_TSh_CF_Country_19862015.csv'))

    # Create hourly DateTime column
    t = pd.date_range('1/1/1986', periods = len(df.index), freq = 'h')

    df['Date'] = t
    df_country['Date'] = t

    df = df.set_index('Hour')
    df_country = df_country.set_index('Hour')

    return df, df_country

def prepare_data_gfm() -> pd.DataFrame:
    
    #df = pd.read_csv(join('data', 'TS.CF.N2.30yr.csv'))
    df = pd.read_csv(join('data', 'EMHIRESPV_TSh_CF_Country_19862015.csv'))

    # Create hourly DateTime column
    t = pd.date_range('1/1/1986', periods = len(df.index), freq = 'h')

    df['Date'] = t

    #df = df.set_index('Hour')

    # Transform country columns into a single row for global forecasting model
    df_melted = df.melt(
        id_vars='Date',
        var_name='Country_Code',
        value_name='Wind_Energy_Potential'
        )
    
    # Add numerically-encoded country codes
    df_melted['Country_Numerical_Code'] = df_melted['Country_Code'].map(constants.COUNTRY_NUMERICAL_CODES_MAP)

    # Drop the original string categorical
    df_melted = df_melted.drop('Country_Code', axis=1)
    
    return df_melted

if __name__ == '__main__':

    df = prepare_data_gfm()
    
    df.to_csv(join('data', 'all_countries_cleaned_30_years.csv'))