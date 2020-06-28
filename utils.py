import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns

# Get features categorical and numerical
def get_columns(data, columns):
    cat_columns = []
    num_columns = []

    for col in data.columns.values:
        if col in columns:
            continue
        elif data[col].dtypes == 'int64':
            num_columns += [col]
        else:
            cat_columns += [col]
    return num_columns, cat_columns

def handle_missing_values(data, median_val):
    df = data.copy()
    for col in df:
        if col in median_val.index.values:
            df[col] = df[col].fillna(median_val[col])
        else:
            df[col] = df[col].fillna("Missing value")
    
    return df

def target_distribution(y_var, data):
    val = data[y_var]

    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 13})
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))

    cnt = val.value_counts().sort_values(ascending=True)
    labels = cnt.index.values

    sizes = cnt.values
    colors = sns.color_palette("Reds", len(labels))

    # histogram
    ax1.bar(cnt.index.values, cnt.values, color=colors)
    ax1.set_title('Count plot of '+y_var)

def plot_bar(data, col, Y_columns, max_cat=10):
    df = data.copy()
    
    fig, axs = plt.subplots(1,2,figsize=(25,6))
    cat_val = df[col].value_counts()[0:max_cat].index.values
    df = df[df[col].isin(cat_val)]

    for i in range(0,2):
        y_col = Y_columns[i]
        Y_values = df[y_col].dropna().drop_duplicates().values
        #print(Y_values)
        for val in Y_values:
            cnt = df[df[y_col] == val][col].value_counts().sort_index()
            axs[i].barh(cnt.index.values, cnt.values)
        axs[i].legend(Y_values, loc='center right')
        axs[i].set_title("Number of "+col+" by "+y_col)

    plt.show()