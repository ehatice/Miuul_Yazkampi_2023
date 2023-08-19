##################
#    NUMPY
##################


import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

# numpy olmadan yapmaya çalışırsak tek tek gezip çarpıp daha sonra yeni liste oluştırıp atmamız lazım
ab = []
for i in range(0, len(a)):
    ab.append(a[i] * b[i])
ab

# numpy ile yaptığımızda

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

np.array([1, 2, 3, 4, 5])
np.zeros(10, dtype=int)

np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))  # ortalaması standar satma 3 *4 array

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)
# reshape: yeniden boyutlandırıyoruz. boyuta bölünebilir olmalı


#### Index seçimi (index selection)

a = np.random.randint(10, size=10)
a[0]
a[0:5]

m = np.random.randint(10, size=(3, 5))
m[0, 0]  # ilk satırı  2. sütunu ifade ediyor

# numpy tek bir tip bilgisi tutar bu yüzden hızlı

##  Fancy index

v = np.arange(0, 30, 3)  # 0 dan 30 hariç üçer üçer artarak

catch = [1, 2, 3]

v[catch]

# Numpy da koşullu işlemler

v = np.array([1, 2, 3, 4, 5])
v<3
v[v < 3]

## Matematiksel işlemler

v = np.array([1, 2, 3, 4, 5])

np.add(v , 1)
np.mean(v)
np.sum(v)

#iki bilinmeyenli denklem çözümleri

# 5 *x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

########################
##### PANDAS ###########
########################

#pandas series

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.size
s.ndim

#VERİ OKUMA

df = pd.read_csv("dataset/advertising.csv")
import seaborn as sns
df = sns.load_dataset("titanic")
df.head(5)
df.columns
df.tail(3)
df.index
df.describe().T #sayısal olan değişkenlerin özet bilgileri geldi
df.isnull().values.any() # eksik değer var mı
df.isnull().sum() # eksik değerleri hesapladı
df["sex"].value_counts()


#değişkeni indexe çevirmek

df["age"].head()
df.age.head()
df.index = df["age"] #indexe yaş değişkenini ekledik
df.drop("age",axis = 1, inplace = True) #sütundan yaş değişkenini sildik
df.head()


#indexi değişkene çevirmek

df["age"] = df.index # birinci yol

df = df.reset_index() #indexten sildi ve sütuna ekledi
df.head()




"age" in df #df içinde var mı diye soruyoruz

df["age"].head()
type(df["age"].head())


#####################################################
## Pandas' da Seçim işlemleri (selection in Pandas)##
#####################################################

df.drop(0, axis = 0).head() #geçici oldu bir değşkene atamadığımız için

delete_index = [1,3,5,7]
df.drop(delete_index,axis = 0).head(10)#silinicekleri bir listeye de atıp da silebilirliz
# yine kalıcı değil. eğer kalıcı olmasını istersek:
# df = diyip yeniden üstteki işlemi yapabiliriz,
# df.drop(delete_index, axis = 0,inplace = True) eklersek kalıcı olmuş olur

#################################
# Değişkenler üzerindeki işlemler
#################################

pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic") #bütün columns görmek için
df.head()

type(df["age"]) #pandas series geldi

type(df[["age"]])# DataFrame geldi. Dataframe de yapmak istiyorsak iki köşeli parantez kullan

col_names = ["age", "who", "alive"]
df[col_names] #type dataframe

#yeni bir sütun eklemek
df["age2"] = df["age"] ** 2
df.head()

#sütun silmek

sil = ["age2"]
df.drop(sil, axis = 1, inplace = True)
df.head()

df["age2"] = df["age"] ** 2
df.head()
df.loc[:,df.columns.str.contains("age")].head(10)
# eğer df.columns başına ~ koyrsak içinde age olan süyunları siler,


###################
# iloc & loc yapısı
###################
df.head()
#iloc: integer based selection

df.iloc[0:3] #0 1 2 indexlerini verir
df.iloc[0, 0]

#loc: label based selection

df.loc[0:3] # 0 1 2 3 indexlerini alır


df.iloc[0:3, "age"] # hata verir çünkü integer based diğer kısma str değil int vermeliydik

df.loc[0:3, "age"] #hata vermeden çalışır
# age yerine bir liste verip de girebilirdik sorun çıkmazdı


########################################
# Koşullu seçim ( conditional selection)
########################################

df[df["age"] > 50].head()
df[df["age"] > 50].count() # bir değişken seçmeden yaptığımız için hepsini vermiş
df[df["age"] > 50]["age"].count() # age değişkenini verdik


secmek = ["class","age"]
df.loc[df["age"] > 50, secmek].head() # iki koşul için & ifadesi kullanıyoruz and kapısı

###################################################
# Toplulaştırma & Gruplama (Aggregation & Grouping)
###################################################

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

#####cinsiyete göre yaşın ortalaması########

df["age"].mean() # bize genel yaş ortalamasını verdi cinsiyet yok

df.groupby("sex")["age"].mean() #asıl cevap bu

df.groupby("sex").agg({"age": "mean"})  #  bu da cevap bu daha iyi bunu kullanmaya çalış

df.groupby("sex").agg({"age": ["mean", "sum"]})  # toplamayı da ekledik agg daha avantajlı

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"}) # hayatta kalanların yüzdeliğini vermiş oldu

df.groupby(["sex","embark_town"]).agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex","embark_town","class"]).agg({
    "age": "mean",
    "survived": "mean",
    "sex": "count"})

##############
# Pivot Table
##############

df.pivot_table("survived","sex", "embarked") #kesişimde yer alanlar ortalamadır

df.pivot_table("survived","sex", "embarked", aggfunc="std") # kesişim standart sapma oldu

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90]) # veriyi nereden bölünmwsi gerektiğini bilmiyorsan qcut fonkdiyonu otomatik olarak yapar

df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex", ["new_age", "class"])
#pd.set_option("display.width",500) yaparsak çıktıyı bölmeden görebiliriz



#########################
# Apply ve Lambda
#########################

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

(df["age"]/10).head()


df.drop("new_age", axis = 1, inplace = True) # alttaki kod new_age yüzünden hata vermişti onu kaldırdım

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())
        # df[col] = df[col]/10 eğer df ye kayıt etmek istersek

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()


df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x- x.mean())/ x.std()).head()


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# df.loc[:, "age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()
#kalıcı yapabilmek için


###########################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
###########################################

# MATPLOTLİB

# kategorik değişken: sütun grafik. countplot bar
# sayısal değişken: hist, boxplot

##### kategorik değişken görselleştirme #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind = "bar")
plt.show(block = True) # çalışmıyor showu algılamıyor

plt.hist(df["age"])
plt.show(block=True)

plt.boxplot(df["fare"])
plt.show(block = True)

#Matplotlib özellikler

#plot

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show(block = True)

plt.plot(x, y, 'o')
plt.show(block = True)

#marker

y = np.array([13, 28, 11, 100])
plt.plot(y, marker = '*')
plt.show(block = True)

#line

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle = "dashed", color = "r")
plt.show(block = True)

#multiple lines


x = np.array([1, 8, 12, 45])
y = np.array([0, 150, 178, 200])
plt.plot(x)
plt.plot(y)
plt.title(" bu ana başlık")
plt.xlabel("x ekseni isimlendirmesi")
plt.ylabel("bu y ekseni isimlendirmesi")
plt.grid() #arkasınıı ızgara şekline getirdi
plt.show(block = True)

#subplots

x = np.array([10,29,39,40,50,79,89,100, 120,123])

y = np.array([15,29,31,40,56,99,89,100, 120,200])
plt.subplot(1,2,1) # bir satırlık 2 sütunluk bri grafik ve bunun birincisi bu

plt.title("1")
plt.plot(x,y)
plt.show(block = True)

a = np.array([10, 20, 30, 40, 50, 60, 70])
b = np.array([1, 2, 3, 4, 5, 6, 7])
plt.subplot(1,2,2)
plt.title("2")
plt.plot(a, b)
plt.show(block = True)

#Seaborn

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x =df["sex"])
plt.show(block = True)

sns.boxplot(x = df["total_bill"])
plt.show(block = True)

###########################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ
###########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df = sns.load_dataset("titanic")
df[["sex","survived"]].groupby("sex")

def check_df(dataframe, head = 5):
    print("##########################shape#############################")
    print(dataframe.shape)
    print("##########################types#############################")
    print(dataframe.dtypes)

check_df(df)


df = sns.load_dataset("flights")#load_dataset in içinde hazır olan bir set kullandık.
df.head()
check_df(df)


########### KATEGORİK DEĞİŞKEN ANALİZİ ##############

df["embarked"].value_counts()
df["sex"].unique()
df["sex"].nunique()

# bütün kategorik değişkenleri seçelim. gizlenenleri ve katg gibi görünüp öyle olmayanları
# bool, category ve object birer kategirk ddeğişken
# survived değişkeni int gözüken ama kategorik değişkendir

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"] ]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat # hepsi kategori

# eğer cat_but_car dolu olsaydı onu da çıkarmamız gerekecekti
# cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique() # kontrol ettik kategorik değişken

def cat_summary ( dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts()/ len(dataframe)}))
    print("################################################")
cat_summary(df,"sex")


for col in cat_cols:
    cat_summary(df, col)
    #hepsini okutmuş olduk

def cat_summary ( dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts()/ len(dataframe)}))
    print("################################################")
    if plot:
        sns.countplot(x = dataframe[col_name], data= dataframe)
        plt.show(block = True)

#cat_summary(df, "sex", plot = True)
for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
    else:
        cat_summary(df, col, plot= True)
# döngüleri fonksiyonun içine yazarsan karışır. ayrı yaz her zaman önemli olan basitlik
#hata aldık (if else yokken) çünkü görselleştiremeyecek şeyler var mesela bool tipi (adult_male)

df["adult_male"].astype(int) #true false ları 0 ve 1 lere çevirdi


##########################
# SAYISAL DEĞİŞKEN ANALİZİ
##########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df[["age","fare"]].describe().T


#bu setteki nümerik değişkenleri nasııl seçerim
num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]

#buradaki bazı değişkenler sayısal gözüküp öyle olmayan değişkenler

num_cols = [col for col in num_cols if col not in cat_cols]
# şimdi tam olarak ulaştık

def num_summary(dataframe, numerical_col,plot = False):
    quantiles  = [0.05, 0.10,  0.20,  0.30,  0.40,  0.50, 0.60,  0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)
num_summary(df, "age",plot =True)

for col in num_cols:
    num_summary(df,col,plot= True)

#docstring: fonksiyona tanım yazma
# yukarıda ayırdıklarımızı bir fonksiyonda tutmuş olduk
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

#hedef değişken analizi

df["survived"].value_counts()

cat_summary(df, "survived")
# survied 1 olanlar acaba neden hayatta kalmkş diye sorguluyoruz

###########HEDEF DEĞİŞKENİN KATEGORİK DEĞİŞKENLER İLE ANALİZİ############

df.groupby("sex")["survived"].mean() # kadınların daha çok hayatta kaldığını görmüş olduk
df.groupby("survived").agg({"age":  "mean"}) # bence daha iyi bir çıktı verdi


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"survived", col)


