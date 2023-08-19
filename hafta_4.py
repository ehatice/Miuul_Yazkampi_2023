import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#########################################
# Sales Prediction with Linear Regression
#########################################
pd.set_option('display.float_format', lambda x: '%.2f' % x) # virgülden sonra 2 basamak göster
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("dataset/advertising.csv")
df.shape

X = df[["TV"]]
Y = df[["sales"]]

###### MODEL ######

reg_model = LinearRegression().fit(X, Y)

# y_hat = b + w * x

# sabit (b - bias)(intercept)

reg_model.intercept_[0]


# tv nin katsayısı (w) coef diye de geçer

reg_model.coef_[0][0]

#############
# Tahmin
############
# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir

reg_model.intercept_[0] + reg_model.coef_[0][0] *150

# 500 birimlik TV harcaması olsa ne kadar satış olması beklenir

reg_model.intercept_[0] + reg_model.coef_[0][0] *500


df.describe().T



 ## MODELİN GÖRSELLEŞTİRİLMESİ ##

g = sns.regplot( x= X, y = Y, scatter_kws= {'color': 'b', 's': 9},
                ci = False, color = "r")
# ci de güven aralığı ekleme dedik

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0],2)} + TV * {round(reg_model.coef_[0][0],2 )} ")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom = 0) # sıfırdan başla
plt.show(block = True)

#################
# Tahmin başarısı
#################

#MSE
y_pred = reg_model.predict(X)
mean_squared_error(Y, y_pred) # gerçek ve tamini verdik ve bize hatayı hesapladı
# düşük değer olması iyi demek
Y.mean()
Y.std() # iyi mi kötü mü olduğunu anlamak için ortalamasına bakıyoruz ve standart sapması

#RMSE
np.sqrt(mean_squared_error(Y, y_pred))
#3.24

#MAE
mean_absolute_error(Y, y_pred)
# son ikisi daha küçük çıktı diye daha iyi olduğu anlamına gelmiyor
#2.54

# R- KARE
reg_model.score(X,Y) # bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir
# tv nin satışı açıklama oluyor burada
# değişken sayısı arttıkça r kare şşişmeye meğillidir
# istatiksel olarak bakmıyoruz konuya makine öğrenimi tarafından bakıyoruz

#############################
# Multiple Linear Regression ( çoklu değişken)
############################

X = df.drop("sales", axis = 1) # bağımsız değişkenleri seçtik

y = df[["sales"]]

### model ###

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.20, random_state= 1)
# burada model seti ve test seti olarak böldük %20 test yaptık

X_train.shape # 160 tanesi model oldu %80lik dilimden dolayı
y_train.shape

X_test.shape
y_test.shape # bunları yapmamıza gerek yok sadece görüp anlamak için yaptık



reg_model = LinearRegression().fit(X_train, y_train)

reg_model.intercept_
reg_model.coef_

##### Tahmin ######

# aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# Tv: 30
# radio: 10
# newspaper: 40

# bias: 2.90
# 0.0468431 , 0.17854434, 0.00258619

# sales = 2.90 + TV * 0.04 + radio * 0.17 + newspaper * 0.002

yeni_veri = [[30],[10],[40]]

yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)
# veride olmayan verileri de tahmin ettik buldurduk

#################################
# Tahmin Başarısını Değerlendirme
#################################

#Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# train rkare
reg_model.score(X_train, y_train) # %61 den &89 çıkardık tahmin etmeyi
# yeni değişken eklediğimizde hatanın düşmesinden dolayı tahmin etme yükseldi


# TEST RMSE
y_pred = reg_model.predict(X_test)
np.sqrt((mean_squared_error(y_test, y_pred)))
# 1.41 aslında test hatasının daha yüksek çıkmasını bekleriz. burda tam tersi yani çok iyi bir durum


#TEST RKARE
reg_model.score(X_test,y_test)

# 10 katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X,  y,cv = 10, scoring = "neg_mean_squared_error")))
# 1.69

##############################################
# Diabetes Prediction with Logistic Regression
##############################################

# iş problemi:
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirebilir misiniz
#Veri seti ABD deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
#parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
#Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
#bağımsız değişkenden oluşmaktadır. Hedef değişken ”outcome” olarak belirtilmiş OLUP ; 1 diyabet test sonucunun
# 1 pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# değişkenler
# Pregnancies: hamilelik sayısı
# glucose: glikoz
# bloodPressure: kan basıncı
# skinThickness: cilt kalınlığı
# ınsulin
# BMI: beden kitle indexi
# diabetesPedigreeFunction: soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
# age: yas
# outcome: kişinin diyabet olup olmadığını bilgisi. hastalığa sahip(1) ya da değil(0)

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7 Prediction for A New Observation

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix,classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantil_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantil_range
    low_limit = quartile1 - 1.5 * interquantil_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return  True
    else:
        False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("dataset/diabetes.csv")
df.head()
df.shape

####################
# Target' ın Analizi
####################

df["Outcome"].value_counts()

sns.countplot(x = "Outcome", data = df)
plt.show(block = True) # count görselleştirdik

100 * df["Outcome"].value_counts() / len(df)  # yüzdelik olarak gösterdik

#######################
# Feature'ların Analizi
#######################

df.describe().T  # sayısal featureların özetinş aldık

df["BloodPressure"].hist(bins = 20)
plt.xlabel("BloodPressure")
plt.show(block = True)


df["Glucose"].hist(bins = 20)
plt.xlabel("Glucose")
plt.show(block = True)

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist( bins = 20)
    plt.xlabel(numerical_col)
    plt.show(block = True)


cols = [col for col in df.columns if "Outcome" not in col]
# outcome bir target olduğu için çıkarttık
for col in cols:
    plot_numerical_col(df, col)

######################
# Target ve Features
######################

df.groupby("Outcome").agg({"Pregnancies": "mean"}) # hamilelik sayısı için diyabet hastası olan

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end = "\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)

#####################################
# Data Preprocessing (veri ön işleme)
#####################################

df.isnull().sum() # eksik değer olup olmadığına baktık

df.describe().T  # eksik veriler 0 larla değiştirilmiş gibi
# eksik değer yokmuş gibi davranacağız

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")  # insulindeki aykırı değerleri hesaplamış olduğumuz eşik değerleri ile değiştirdil

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])  # robust aykrıı değerlerden etkilenmiyor

df.head()

######################
# Model & Prediction
######################

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_  # fonksiyondaki b ifadesi

log_model.coef_  # fonksiyondaki w değerleri

y_pred = log_model.predict(X)  # x (bağımsız değişkenleri) veriyoruz ve bize y ( bağımlı değişkeni) elde et diyoruz

y_pred[0:10]
y[0:10]

##########################
# Model Evaluation
##########################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm,annot = True, fmt = ".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size = 10)
    plt.show(block = True)

plot_confusion_matrix(y,y_pred)

print(classification_report(y, y_pred))