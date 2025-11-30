import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import csv
import sklearn
from sklearn.metrics import f1_score


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Compute features
    df.loc[:,"pT"] = abs(1/df.loc[:,"q/pt"])

    phi_0 = df['Phi_0'].values
    phi_2 = df['Phi_2'].values
    phi_3 = df['Phi_3'].values
    phi_4 = df['Phi_4'].values

    theta_0 = df['Theta_0'].values
    theta_2 = df['Theta_2'].values
    theta_3 = df['Theta_3'].values
    theta_4 = df['Theta_4'].values

    # Compute sine and cosine using NumPy
    df['cos_Phi_0'] = np.cos(np.radians(phi_0))
    df['cos_Phi_2'] = np.cos(np.radians(phi_2))
    df['cos_Phi_3'] = np.cos(np.radians(phi_3))
    df['cos_Phi_4'] = np.cos(np.radians(phi_4))

    df['sin_Phi_0'] = np.sin(np.radians(phi_0))
    df['sin_Phi_2'] = np.sin(np.radians(phi_2))
    df['sin_Phi_3'] = np.sin(np.radians(phi_3))
    df['sin_Phi_4'] = np.sin(np.radians(phi_4))

    # Compute eta using NumPy
    df['Eta_0'] = -np.log(np.tan(np.radians(theta_0 / 2)))
    df['Eta_2'] = -np.log(np.tan(np.radians(theta_2 / 2)))
    df['Eta_3'] = -np.log(np.tan(np.radians(theta_3 / 2)))
    df['Eta_4'] = -np.log(np.tan(np.radians(theta_4 / 2)))
    
    # Handle momentum
    label =df.loc[:,"pT"]

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = label.quantile(0.25)
    Q3 = label.quantile(0.75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    # Define the acceptable range (1.5 times the IQR below Q1 and above Q3)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    filtered_df = df[(label >= lower_bound) & (label <= upper_bound)]
    filtered_df = filtered_df.reset_index(drop=True)
    # Extract the indices of the filtered dataset
    filtered_indices = filtered_df.index.to_numpy()

    # Split the filtered indices into training and test sets
    train_idx, test_idx = train_test_split(filtered_indices, test_size=0.2, random_state=1)
    
    # Scale features
    scaler = StandardScaler()
    selected_columns = [
        'sin_Phi_0', 'sin_Phi_2', 'sin_Phi_3', 'sin_Phi_4', 'cos_Phi_0', 'cos_Phi_2', 'cos_Phi_3', 'cos_Phi_4', 'Eta_0', 'Eta_2', 'Eta_3', 'Eta_4', 
        'BendingAngle_0', 'BendingAngle_2', 'BendingAngle_3', 'BendingAngle_4','pT'
    ]

    # Select the columns from the DataFrame
    filtered_df = filtered_df[selected_columns]
    filtered_df.loc[:,"sin_Phi_0":] = scaler.fit_transform(filtered_df.loc[:,"sin_Phi_0":])
    selected_features = ['sin_Phi_0', 'sin_Phi_2', 'sin_Phi_3', 'sin_Phi_4', 'cos_Phi_0',
       'cos_Phi_2', 'cos_Phi_3', 'cos_Phi_4', 'Eta_0', 'Eta_2', 'Eta_3',
       'Eta_4', 'BendingAngle_0', 'BendingAngle_2', 'BendingAngle_3',
       'BendingAngle_4']
    
    
    # Split data
    x_data = filtered_df[selected_features].to_numpy()
    label = filtered_df['pT'].to_numpy()
    
    return filtered_df, x_data, label, train_idx, test_idx, scaler

def f1_comp(y_true, y_pred):
    f1 = []
    for i in range(100): 
        grnd = y_true >= i
        pred = y_pred >= i
        f1.append(f1_score(grnd, pred))
    return f1

def acc_comp(y_true, y_pred):
    acc = []
    for i in range(100):
        grnd = y_true >= i
        pred = y_pred >= i
        cmp = np.sum(np.equal(grnd, np.squeeze(pred))) 
        acc.append(cmp / len(grnd) * 100)
    return acc

def cuts(datain, datacheck, minval, maxval):
    return datain[np.logical_and(datacheck > minval, datacheck < maxval)]

def plot_gaussian(true, pred, space, varname, lower, upper, bins):
    resmeans = []
    stdevs = []
    bincenters = []
    samples = []
    
    width = (upper - lower) / bins
    true = cuts(true, pred, lower, upper)
    pred = cuts(pred, pred, lower, upper)
    pred = cuts(pred, true, lower, upper)
    true = cuts(true, true, lower, upper)
    
    resids = true - pred
    
    for i in range(bins):
        lowertemp = lower + width * i
        uppertemp = lower + width * (i + 1)
        bincentertemp = (uppertemp + lowertemp) / 2
        
        residscut = cuts(resids, true, lowertemp, uppertemp)
        varnametemp = f'{varname} Residuals Distribution ({lowertemp:.2f} to {uppertemp:.2f})'
        residscut.sort()
        
        resmeantemp, resstdtemp = norm.fit(residscut)
        samplestemp = len(residscut)
        pdf = stats.norm.pdf(residscut, resmeantemp, resstdtemp)
        
        plt.hist(residscut, bins=bins, histtype='step', color='blue', density=1, label='Residuals')
        plt.plot(residscut, pdf, label='Normal Curve', color='black')
        plt.title(varnametemp)
        plt.axvline(resmeantemp, label=f'Mean: {resmeantemp:.2f}', color='red')
        plt.axvspan(resmeantemp - resstdtemp / 2, resmeantemp + resstdtemp / 2, facecolor='g', alpha=.3, label='Stdev')        
        plt.legend()
        plt.savefig(f"{space}_gaussian.png", bbox_inches='tight')
        plt.close()
        
        resmeans.append(resmeantemp)
        samples.append(samplestemp)
        stdevs.append(resstdtemp)
        bincenters.append(bincentertemp)
    
    resmean = np.mean(resids)
    stddev = np.std(resids)    
    
    return resmeans, stdevs, bincenters, samples, resmean, stddev

def heatmap(true, pred, space, varname, lower, upper, bins):
    heatmap, xedges, yedges = np.histogram2d(true, pred, bins=bins, range=[[lower, upper], [lower, upper]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.plot([lower, upper], [lower, upper], color='blue')
    fig = plt.gcf()
    plt.set_cmap('gist_heat_r')
    plt.xlabel(f'{varname} True')
    plt.ylabel(f'{varname} Pred')
    plt.title('Frequency Heatmap')
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.colorbar()
    fig.savefig(f"{space}/_heatmap.png")
    plt.close()


def save_results(test_loss, y_true, y_pred, save_dir, min_pT, max_pT):
    results = {
        'Test_loss': test_loss,
        'MAE': sklearn.metrics.mean_absolute_error(y_true, y_pred),
        'MSE': sklearn.metrics.mean_squared_error(y_true, y_pred)
    }

    with open(f'{save_dir}/evaluation_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for key, value in results.items():
            writer.writerow([key, value])
 
    plot_gaussian(y_true, y_pred, save_dir, 'pT', min_pT, max_pT, 100)
    heatmap(y_true, y_pred, save_dir, 'pT', min_pT, max_pT, 100)