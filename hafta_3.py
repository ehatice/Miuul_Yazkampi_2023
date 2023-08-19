# !pip install missingno yükledik

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%3f' % x )
pd.set_option('display.width', 500)


def load_application_train():
    data = pd.read_csv("dataset/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("dataset/titanic.csv")
    return data

df = load()
df.head()



###################################
#####  AYKIRI DEĞER YAKALAMA ######
###################################

# GRAFİK TEKNİĞİ İLE AYRIK DEĞERLER

sns.boxplot(x = df["Age"])
plt.show(block = True)


#aykırı değerler nasıl yakalanır

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1

up = q3 + 1.5 *iqr
low = q1 - 1.5 * iqr # eksi değer geldi yaş değişkeni - olamayacağına göre görmezden gelinecek


df[(df["Age"] < low ) | (df["Age"] > up)].index

# ayrıkı değer var mı
df[(df["Age"] < low ) | (df["Age"] > up)].any(axis = None)

df[(df["Age"] < low )].any(axis = None) # çünkü eksi değer yoktu demiştik

# eşik değerlere eriştik
# aykırılara eriştik
# hızlıca ayrıkırı değer var mı yok mu soduk ( bool şeklinde)

################################
### İŞLEMLERİ FONKSİYONLAŞTIRMA
################################

def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantil_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantil_range
    low_limit = quartile1 - 1.5 * interquantil_range
    return low_limit, up_limit


outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")
df[(df["Fare"] < low ) | (df["Fare"] > up)].head()


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return  True
    else:
        False


check_outlier(df, "Age")

################
# grab_col_names
################

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """
    veri setindeki kategorik, nümerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    not: kategorik değilkenlerin içerisine nümerik görünümlü kategorik deüişkenler de dahildir.

    :param
    dataframe: değişkenlerin isimlerini alınmak istenen dataframe
    cat_th: int,optinal
        nümerik fakat kategorik değişkenker sınıf eşik değeri
    car_th: int, optinal
        kategorik kafat kardinal değişkenler için eşik değeri

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes != "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat ]


    print(f"observations: {dataframe.shape[0]}")
    print(f"variable: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
# doğru çalışmıyor çözemedim gitti


num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))
    #aykırı değerler var mı diye kontrol ettik bool şeklinde


### AYKIRI DEĞERLERİN KENDİLERİNE ERİŞMEK

def grab_outliers(dataframe, col_name, index = False):
    low, up = outlier_thresholds(dataframe,col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age",True)

################################
# AYKIRI DEĞER PROBLEMLERİ ÇÖZME
################################

## SİLME

low, up = outlier_thresholds(df, "Fare")

df.shape

df[~((df["Fare"]< low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((df["Fare"]< low_limit) | (df["Fare"] > up_limit))]
    return df_without_outliers



cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape
for col in num_cols:
    new_df = remove_outlier(df,col)

df.shape[0] - new_df.shape[0]
# kaç tane silindiğinin çıktısını aldık

########## BASKILAMA YÖNTEMİ (RE-ASSİGNMENT WİTH THRESHOLDS)

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"]< low) | (df["Fare"] > up))]["Fare"]

df.loc[(df["Fare"]< low) | (df["Fare"] > up), "Fare"] # locla yapımı

df.loc[(df["Fare"] > up), "Fare"] = up # aykırı değerlerinin hepisine up değerini koymuş olduk yani baskıladık


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#############################################################
# ÇOK DEĞİŞKENLİ AYKIRI DEĞER ANALİZİ: LOCAL OUTLIER FACTOR
############################################################

# 17 yaşında 3 evlilik yapmış biri aykırı bir değer olur( tek tek ele alsak bir aykırılık söz kousu değildir)

# mesela 100 değişkenli bir yapıyı nasıl 2 boyuta indirgeyebiliriz (mülakat sorularundan)


# low ve up değerlerine 0.25 ve 0.75 almıştık fakat outlierlaea bakarak 5-95 yapabiliriz

###################################
# MISSING VALUES( EKSİK DEĞERLER)
###################################

# EKSİK DEĞERLERİN YAKALANMASI

df = load()
df.head()

#eksik gözlem var mı yok mu sorgusu
df.isnull().values.any() # bool cevap veriyor


df.isnull().sum()


# en az bir tane eksik değere sahip gözlem birimleri
df[df.isnull().any(axis = 1)]

# azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending = False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending = False) # yüzdelik olarak aldık


na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


# bir fonksiyonda yapyıklarımızı birleştirelim

def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending= False)
    ratio = (dataframe[na_columns].isnull().sum()/ dataframe.shape[0] * 100).sort_values(ascending= False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)],axis = 1, keys = ['n_miss', 'ratio'])
    print(missing_df, end = "\n")
    if na_name:
        return na_columns

missing_values_table(df)

###############################
#eksik değer problemini çözme
##############################

# ağaç yöntemi varsa gözardı edebiliriz.etkisi düşük yapacağımız çözümlee

#ÇÖZÜM 1: HIZLICA SİLMEK
df.dropna().shape # en az bir tane bile eksik veri varsa siliyor. eğer çok fazla verimiz varsa silebilriz

#ÇÖZÜM 2: BASİT ATAMA YÖNTEMLERİ İLE DOLDURMAK

df["Age"].fillna(df["Age"].mean())

df["Age"].fillna(df["Age"].mean()).isnull().sum()

df["Age"].fillna(0).isnull().sum()
# sabit bir değişken ile de doldurabiilriz


df.apply(lambda x: x.fillna(x.mean(), axis = 0))
# axis 0 aşağı doğrı satırları alacağız
# hata verdi çünkü sözel verileri de sayısal olarak doldurmaya çalıştk

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis = 0).head()


dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis = 0)

dff.isnull().sum().sort_values(ascending= False)
# sayısal değişkenlerin null kısımalrını doldurduk fakat sözel veriler kalsdı


df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing") # kendiöiz belirlediğğimiz bri şeyi koyduk


df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis =0).isnull().sum()

############################################
# KATEGORİK DEĞİŞKEN KIRILIMINA DEĞER ATAMA
############################################

df.groupby("Sex")["Age"].mean() # cinsiyete göre kırılım yapacağız

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]


#ÇÖZÜM 3: TAHMİNE DAYALI ATAMA İLE DOLDURMA

#makine öğrenmesi var


def grab_col_names(dataframe,cat_th= 10, car_th = 20):
    """
    veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir

    :param dataframe:df
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
#diğer fonksiyon düzgün çalışmadığı için bunu getirdim


#ÇÖZÜM 3: TAHMİNE DAYALI ATAMA İLE DOLDURMA

#makine öğrenmesi var

df = load()
df.head()

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first= True)

dff.head()

# değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns = dff.columns)
dff.head()


# knn in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors= 5)
# knn yöntemi kısaca: komşunu söyle sana kim olduğunu söyliyim
dff = pd.DataFrame(imputer.fit_transform(dff),columns = dff.columns)
dff.head()

# doldurduğumuz verileri göremedik tam olarak (standartlaştırılmış olarak görüyoruz)

dff = pd.DataFrame(scaler.inverse_transform(dff), columns = dff.columns)
dff.head() # nasıl değiştiğğini göremedik yeni veriyi gördük ama

df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]


#######################
# Gelişmiş Analizler
#######################

msno.bar(df)
plt.show(block = True) # tam olan gözlemlerin sayılarını veriyor

msno.matrix(df)
plt.show(block = True)

msno.heatmap(df)
plt.show(block = True) # ısı tablosu

####################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
####################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

####################################
# LABEL ENCOING & BINARY ENCODING
###################################

df = load()
df.head()
df["Sex"].head() # binary encoder yapmak istiyoruz

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
# 0 1 leri alfabetik sıraya göre verir

# sıfır bir hangisi olduğunu unutursak
le.inverse_transform([0,1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]


for col in binary_cols:
    label_encoder(df,col)

df.head()

df = load_application_train()
df.shape


binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]


df[binary_cols].head()


for col in binary_cols:
    label_encoder(df, col)
# eksik değerleri de doldurmuş 2 ile


df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique()) # unique de boş değerleri de kategori olaraak alıyorr

######################
# One- Hot Encoding
######################

df = load()
df.head()
df["Embarked"].value_counts() # burada 3 değişken var ve aralarında değer farkı yok

pd.get_dummies(df, columns= ["Embarked"]).head() # seçili olan 1 diğerleri 0 olacak şekilde gözüküyor

pd.get_dummies(df, columns= ["Embarked"],drop_first= True).head() # değişkenler birbirinin üzerinden türemesin diye koyduk

pd.get_dummies(df, columns= ["Embarked"], dummy_na= True).head() # eksiklikleri de bir sınıfa dönüştürüyor

pd.get_dummies(df, columns= ["Sex"],drop_first= True).head() # il değişken olan maleyi aldı diğeri male ise 1 değilse 0 yazıd

pd.get_dummies(df, columns= ["Embarked", "Sex"], drop_first= True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first= drop_first)
    return dataframe

df = load()

# num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

###################
# Rare Encoding
##################

# 1 kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
# 2 Rare Kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
# 3 Rare encoder yazacağız

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()


cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary ( dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts()/ len(dataframe)}))
    print("################################################")
    if plot:
        sns.countplot(x = dataframe[col_name], data= dataframe)
        plt.show(block = True)

for col in cat_cols:
    cat_summary(df, col)


#2
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

# 3

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET",cat_cols)

#########################################
# Feature Scaling (özellik ölçeklendirme)
#########################################

###############
# StandartScaller: klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya bol. z = (x - u) / s
###############

df = load()

ss = StandardScaler()
df["Age_standart_scaller"] = ss.fit_transform(df[["Age"]])

df.head()
# aykırı değerlerden etkilenirler

###############
#  RobustScaller: Medyanı çıkar iqr a böl
###############

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T
# çok fazla tercih edilmiyor. aykırı değerlerden etkilenmiyor

################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
################

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col,plot = False):
    quantiles  = [0.05, 0.10,  0.20,  0.30,  0.40,  0.50, 0.60,  0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins = 20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

for col in age_cols:
    num_summary(df,col,plot = True)
    # değişkenlere eşit yaklaşılması demek onları bozmak değildir. yapılarını koruyacak şekilde ifade ediliş tarzlarını değiştirdik


####################
# Numeric to Cotegorical: sayısal değişkenlerş kategorik değişkenlere çevirme
# binning
###################

df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()

#################################
# Feature Extraction ( özellik çıkarımı)
##################################

###################################
# Binary Features: Flag, Bool, True- False
####################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int64") # dolu mu diyoruz ve int çeviriyoru
# dolu olanlara 1 boş olanlara 0 yazdık

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})
# kabin bilgisi olanların hayatta kalması daha yüksek çıktı

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                               df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value = 0.00 olursa anlamlı bir farklılık oluyor

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"

df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                               df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))


######################################
# Text'ler Üzerinden Özellik üretmek
######################################
df = load()
df.head()

######## Letter Count ########

df["NEW_NAME_COUNT"] = df["Name"].str.len()

######### Word Count ###########

df["NEW_NAME_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

########## Özel Yapıları Yakalamak ##########

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len( [x for x in x.split()if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]}) # doktor olanların ve olmaynaların hayatta kalma oranlarını kıyasladık
# frekansa da bakmak lazım

################
# Regex ile değişken türetmek
################

df = load()
df.head()

