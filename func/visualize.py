import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pylab
from IPython.display import display
from statsmodels.graphics import tsaplots
import statsmodels.api as sm

def target_plot(df):
    fig = df['price'].hist(figsize=(5,5), bins=50, color = 'blue')
    return fig
def num_feature_plot(df):
    num_cols = list(df.select_dtypes(exclude=['object']).columns)
    fig = df[num_cols].drop(columns = ['price']).hist(figsize=(18,15), bins=50, color = 'blue')
    return fig

def cat_feature_plot(df):
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    plt.figure(figsize=(18, 35))
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(6, 2, i)
        sns.countplot(data=df, x=col)
        plt.ylabel('Count')
        if col in ['zipcode']:
            plt.xticks(rotation=90)
        else:
            plt.xticks()
    plt.show()

def corr_plot(df):
    corr_matrix = df.corr(numeric_only = False)
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis')
    plt.title('Correlation Matrix of Features')
    plt.show()

def corr_table(df):
    corr = df.corr(numeric_only = False)
    display(corr)

def acf_plot(x):
    fig = tsaplots.plot_acf(x.resid, lags=10)
    plt.ylim([0.00, 1.00])
    plt.title('Autocorrelation Function Plot of Model Residuals')
    plt.show();

def norm_plot(x):
    sm.qqplot(x.resid, line='s')
    pylab.show();

def feature_importance(model, features):
    explainer = shap.explainers.Linear(model, features)
    shap_values = explainer(features)
    return shap.plots.bar(shap_values, max_display=99, show=False)
