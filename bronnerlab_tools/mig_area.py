import glob
import os
import warnings
import numpy as np
import scipy.stats as st
import pandas as pd
import numba

def process_directory(directory_path):
    '''
    Process all files in data directory
    ---
    Arguments:
        directory_path: str
            Path to directory.
    Returns:
        DataFrame
            Pandas DataFrame containing metadata and migration area data from each file in directory.
    '''
    # Read in file names
    filenames = glob.glob(os.path.join(directory_path, '*.csv'))
    nfiles = len(filenames)
    df_list = []
    
    # Define metadata quantities
    metadata = ['date', 'treatment', 'concentration', 'fluorescent channels', 'embryo number', 'somite stage', 
                 'magnification']
    metadata_dict = {}
    
    # Set up empty list for each metadata quantity
    for quantity in metadata:
        metadata_dict[quantity] = []
    
    # For each file, get metadata from file name
    for i, file in enumerate(filenames):
        # Read in file as a dataframe and store in df_list
        df = pd.read_csv(file)
        labels = [l.split(':')[1] for l in df['Label'].values]
        df['Label'] = labels
        df_list.append(df)
        # Parse metadata from file name and add to metadata dictionary
        fname = file[(file.rfind('/')+1):file.rfind('.')]
        fields = fname.split('_')
        for j, quantity in enumerate(metadata):
            metadata_dict[quantity].append(fields[j])
    
    # Combine dataframes for all embryos
    df_combined = pd.concat(df_list)
    df_combined = df_combined.reset_index()
    
    # Create dictionary to store data
    data = ['CntlArea', 'ExptArea']
    data_dict = {}
    # Add data from each embryo to data_dict
    for value in data:
        data_dict[value] = df_combined.loc[df_combined['Label'] == value]['Area'].values
    
    # Create final dataframe combining metadata and data
    final_dict = {**metadata_dict, **data_dict}

    return pd.DataFrame.from_dict(final_dict)

def normalize_migration_areas(df):
    '''
    Compute normalized migration areas from raw values.
    ---
    Arguments:
        df: DataFrame
            DataFrame containing control and experimental migration areas.
    Returns:
        df: DataFrame
            Updated DataFrame containing normalized migration areas.
    '''
    # Compute mean control area
    cntl_mean = np.mean(df['CntlArea'].values)
    # Divide data by mean control area
    df['Norm Cntl Area'] = df['CntlArea'].values/cntl_mean
    df['Norm Expt Area'] = df['ExptArea'].values/cntl_mean
    df['Expt/Cntl'] = df['Norm Expt Area']/df['Norm Cntl Area']
    
    return df

def create_stats_df_single_treatment(df):
    '''
    Create DataFrame of statistics from migration area data.
    ---
    Arguments:
        df: DataFrame
            Normalized migration area DataFrame.
    Returns:
        DataFrame
            DataFrame containing summary statistics.
    '''
    n = len(df)
    t_stat, p_value = st.ttest_ind(df['Norm Cntl Area'].values, df['Norm Expt Area'].values)
    power = get_power(get_effsize(df['Norm Cntl Area'].values, df['Norm Expt Area'].values,
                                  paired=False, meantest='t-test', tails='2-tailed'), 4, 0.05)
    stats_dict = {'treatment': df['treatment'].values[:2], 
                  'N embryos': [n, n],
                  'area': ['Control', 'Experiment'], 
                  'Mean Norm Area':[np.mean(df['Norm Cntl Area'].values), np.mean(df['Norm Expt Area'].values)],
                 'Standard Dev': [np.std(df['Norm Cntl Area'].values), np.std(df['Norm Expt Area'].values)], 
                 't-statistic':[t_stat, t_stat], 
                 'p-value': [p_value, p_value], 
                 'power': [power, power]}
    
    return pd.DataFrame.from_dict(stats_dict)

def get_effsize(x1, x2, paired=False, meantest='t-test', tails='2-tailed'):
    '''
    Calculates effect size using Cohensd_s if unpaired sample or Cohensd_z if paired
    sample.
    ---
    Keywords:
        x1: 1D array.
            x1 is a 1D array containing your data.
        x2: 1D array.
            x2 is a 1D array containing your second data sample.
        paired: Boolean. Default=False.
            States whether data is paired. Determines which Cohensd effect statistic
            to calculate.
    Returns:
        effsize: float.
            Effect size.
    '''
    import numpy as np

    if paired == True:
        check_paired_samples(x1, x2)
        effsize = get_cohensd_z(x1, x2, paired=True)

    if paired == False:
        effsize = get_cohensd_s(x1, x2, paired=False)

    return effsize

def get_cohensd_z(x1, x2, paired=False):
    '''
    Calculates Cohensd_z effect size for paired sample.
    ---
    Keywords:
        x1: 1D array.
            x1 is a 1D array containing your data.
        x2: 1D array.
            x2 is a 1D array containing your second data sample. x1 and x2 are paired.
        paired: Boolean. Default=False.
            Indicates whether samples are paired. Samples must be paired to calculate
            Cohensd_z statistic.
    Returns:
        cohensd_z: float.
            Cohensd_z effect size estimate.
    '''
    import numpy as np

    # Check samples are paired. If samples are unpaired Cohensd_z cannot be used.
    assert paired==True, 'Samples must be paired to calculate Cohensd_z'

    # Calculate differences between paired samples.
    diffs = x2 - x1

    # Calculate Cohens d_z
    meandiff = np.mean(diffs)
    standard_dev = np.std(diffs, ddof=1)
    cohensd_z = meandiff/standard_dev

    return cohensd_z

def get_cohensd_s(x1, x2, paired=False):
    '''
    Calculates Cohensd_s effect size for 2 independent samples.
    ---
    Keywords:
        x1: 1D array.
            x1 is a 1D array containing your data.
        x2: 1D array.
            x2 is a 1D array containing your second data sample. x1 and x2 are independent.
        paired: Boolean. Default=False.
            Indicates whether samples are paired. Samples must be independent to calculate
            Cohensd_s statistic.
    Returns:
        cohensd_s: float.
            Cohensd_s effect size estimate.
    '''
    import numpy as np

    # Check samples are independent.
    assert paired==False, 'If samples are paired use Cohensd_z instead'

    # Calculate Cohensd_s. Degrees of freedom = N-1.
    n_obs1_adj = len(x1) - 1
    n_obs2_adj = len(x2) - 1

    denom = n_obs1_adj + n_obs2_adj

    std1 = np.std(x1)
    std2 = np.std(x2)

    meandiff = np.mean(x1) - np.mean(x2)

    cohensd_s = meandiff/np.sqrt((n_obs1_adj*std1**2 + n_obs2_adj*std2**2)/denom)

    return cohensd_s


def get_power(effsize, n_obs, alpha, sides='two-sided'):
    '''
    Calculates power statistic. Power indicates the probability that the
    null hypothesis was rejected correctly. Power = 1 - P(Type II error)
    ---
    Keywords:
        effsize: float.
            Effect size calculated from Cohensd statistic.
        n_obs: int.
            Number of observations of the sample.
        alpha: float.
            P-value.
        sides: str. Default='two-sided'.
            Determines 2-tailed or 1-tailed test. 2-tailed ('two-sided')
            is preferred. If 1-sided, pass in 'larger' or 'smaller'.
    Returns:
        power: float.
            Calculated power of statistical test.
    '''
    # Import statsmodels.stats.power package to use TTestPower function
    import statsmodels.stats.power as stats_power

    # Calculate power
    power = stats_power.TTestPower.power('self', effsize, n_obs, alpha, alternative=sides)
    return power
