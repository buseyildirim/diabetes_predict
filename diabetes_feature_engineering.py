import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load(dataset_path):
    data = pd.read_csv(dataset_path)
    return data

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# Lets analyse data
df = load("datasets/diabetes.csv")
check_df(df)
# get categorik,numerik and cat but cardinal columns.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Outcome is target.
cat_summary(df, "Outcome", plot=True)

# Numeric column analysis

for col in num_cols:
    num_summary(df, col, plot=True)

# Lets Make target variable analysis.


# The only categorical variable here is the target itself. So it won't be meaningful.
for col in cat_cols:
    target_summary_with_cat(df,"Outcome",col)


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)
# Variables with the highest target explanatory are insulin and glucose

# Data Preperation

# Outlier Analysis

for col in num_cols:
    print(f"{col} : {check_outlier(df,col)}")
    grab_outliers(df, col)


# All variable have outlier :)

# Missing Value Analysis

missing_values_table(df,True)

# Missing value does not exist. :)

# Analysis of Correlation

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


# Missing Value Processing
df.describe().T
# Glucose, bloodpressure,skinthickness,insulin and bmi can not be 0 normally. These are missing value.

df.loc[df["Glucose"]==0,"Glucose"] = float('NaN')
df["Glucose"].isnull().sum()

df.loc[df["BloodPressure"]==0,"BloodPressure"] = float('NaN')
df["BloodPressure"].isnull().sum()

df.loc[df["SkinThickness"]==0,"SkinThickness"] = float('NaN')
df["SkinThickness"].isnull().sum()

df.loc[df["Insulin"]==0,"Insulin"] = float('NaN')
df["Insulin"].isnull().sum()

df.loc[df["BMI"]==0,"BMI"] = float('NaN')
df["BMI"].isnull().sum()

# Let's check the missing value again.
missing_column = missing_values_table(df,True)

for column in missing_column:
    df.loc[(df["Outcome"] == 0) & (df[column].isnull()), column] = df[df["Outcome"] == 0][column].median()
    df.loc[(df["Outcome"] == 1) & (df[column].isnull()), column] = df[df["Outcome"] == 1][column].median()

#Let's check the missing value again.
missing_column = missing_values_table(df,True)
# Outlier Processing

for col in num_cols:
    replace_with_thresholds(df, col)

#Let's check the outlier again.

for col in num_cols:
    print(check_outlier(df,col))

# Feature Engineering
df.head()
df["age_bmi_ratio"] = df["Age"]/df["BMI"]
df["pregnancies_age_ratio"] = df["Pregnancies"]/df["Age"]
df["glucose_age_ratio"] = df["Glucose"]/df["Age"]
df["skinthickness_age_ratio"] = df["SkinThickness"]/df["Age"]
df["insulin_age_ratio"] = df["Insulin"]/df["Age"]
df["glocose_square"] = df["Glucose"]**2
df["glocose_dot_insulin_sqaure"] = (df["Glucose"]*df["Insulin"])**2
df["pregnancies_age_ratio_flag"] = pd.qcut(df['pregnancies_age_ratio'], 3 , labels=["Low","Medium","High"])
df["age_flag"] = pd.qcut(df['Age'], 3, labels=["young","middle_aged","old"])


# Label Encoding


df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()
# Feature Scaling

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

#############################################
# 8. Model
#############################################
import lightgbm as lgbm
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

lgbm_model = lgbm.LGBMClassifier().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_pred, y_test)
plot_importance(lgbm_model,X_train,len(X))