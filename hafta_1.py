'''SANAL ORTAM VE PAKET YÖNETİMİ
(terminal bölümünden erişiyoruz)
-Sanal ortamların listelenmesi: conda env list
-Sanal ortam oluşturma:conda create -n ecem
-sanal ortamı aktif etme : conda activate ecem
-yüklü paketlerin listelenmesi: conda list
-paket yüklenmesi: conda install numpy
-paket silme: conda remove package_name
-belirli bir versiyona göre yükleme:conda install numpy = 1.20.2
-paket yükseltme: conda upgrade numpy
-bütün paketlerin yükseltilmesi: conda upgrade -all
-pip: python package index: paket yönetim aracı
-paket yükleme: pip install paket_Adı
-paket yükleme versiyona göre: pip install pandas == 1.2.1 (daha öncekini silip istediğimiz versiyonu yüklüyor)
-conda sadece sanal ortam işlemleri için kullanabiliriz(pip ve conda ile yükleme yapılabiliri)

##çalıştığımız paket ve sürümleri görüp başka bir prije ya da ortama eklemek için
conda env export > environment.yaml
ls dir(göstermek için)


#DEMET(TUPLE)
-değiştirilemez
-sıralı
-kapsayıcı
--liste gibi tek fark değiştirilemez demeti list e çevirip işlem yapabiliriz(değişmiş olur)
#SET(KÜME)
-değiştirilebilir
-sırasız + eşsiz
-kapsayıcıdır
#difference() : iki kümenin farkı


'''
set1 = set([1,3,5])
set2 = set([1,2,3])

#set1 de olup set 2 de olmayan değer
set1.difference(set2)
set1 -set2

#set2 de olup set1 de olmayanlar
set2.difference(set1)
set2-set1

set1.symmetric_difference(set2)#iki kümede birbirşne göre olmayan -değiştirsek de setlerin değerini aynı değer gelir fark yok

set1.intersection(set2) #iki kümenin kesişimi

set1.union(set2)# iki setin birleşimi

set1.isdisjoint(set2) #iki kümenin kesşimi boş mu?

#bazı koddaki kodları konsol bölünümne yazıp da kullanabiliriz

'''FONKSİYONLAR
-parametre= fonk tanımlanmasında iafede edilen değişkendir
-argüman = bu fonksiyonlar çağrıldığında parametre değerlerine karşılık girilen değerlerdir
-genellikle hepsine argüman diyoruz
konsola ?print yaz (sep end vs birer srgüman oluyor)
'''


def calculate(x):
    print(x*2)


calculate(5)

#iki argümanlı fonk


def summer(arg1,arg2):
    print(arg1 +arg2)


summer(8,7)

''' DOCSTRİNG
-fonksiyonlara nilgi notu ekleme 
'''
def summer(arg1,arg2):
    '''
    sum of two numbers

    :param arg1: int, float
    :param arg2: int,float
    :return: int,float

    '''
    # üç tırnak ve enter olduğunda çıkıyor
    print(arg1 +arg2)

#FONKSİYONLARIN STATEMENT/BODY BÖLÜMÜ#

def say_hi(string):
    print(string)
    print("hi")
    print("hello")

say_hi("miuul")


def multiplication(a, b):
    c = a * b
    print(c)

multiplication(10,9)

#girilen değerleri bir liste içinde saklayacak fonksiyon

liste = []

def add_element(a,b):
    c = a * b
    liste.append(c)
    print(liste)

add_element(1,8)
add_element(5,8)


#ÖZEL TANIMLI ARGÜMANLAR/PARAMETRELER (DEFAULT PARAMETERS/ARGUMENTS)

def divide(a,b):
    print(a/b)

divide(1,2)

def say_hi(string = "merhaba"):
    print(string)
    print("hi")
    print("hello")

say_hi()
say_hi("selamm")

#RETURN : FONKSİYON ÇIKTILARINI GİRDİN OLARAK KULLANMAK

def calculate(varm,moisture,charge):
    varm = varm *2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture)/charge
    return varm,moisture,charge,output

 varm,moisture,charge,output =calculate(98,12,78)

#FONKSİYON İÇİRİSİNDEN FONKSİYON ÇAĞIRMAK

#UYGULAMA-MÜLAKAT SORUSUU#
#amaç: aşağıdaki şekide string string değişen fonksiyon yazmak istiyorum
#before: "hi my name is john and i am learning python"
#after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

cumle = "hi my name is john and i am learning python"
def degistir(string):
    new_string = ""
    for i in range(len(string)):
        if i % 2 == 0:
            new_string += string[i].upper()
        else:
            new_string += string[i].lower()
    return new_string

degistir(cumle)
#################################################################
#####ENUMERATE: otomatik counter/indexer ile for loop
#############################################################
studentss = ["John","Mark","Venessa","Mariam"]


for index, student in enumerate(studentss):
    print(index,student)


ciftli = []
tek = []
for index,student in enumerate(studentss):
    if index % 2 == 0:
        ciftli.append(student)
    else:
        tek.append(student)

print(ciftli)
print(tek)

#########################
#UYGULAMA- MÜLAKAT SORUSU
#########################
#divide_students fonksiyonu yazınız.
#Çift indexte yer alan öğrencileri bir listeye alınız.
#Tek indexte yer alan öğrencileri başka bir listeye alınız.
#Fakat bu iki liste tek bir liste olarak return olsun.

students = ["John","Mark","Venessa","Mariam"]

def divide_students(students):
    groups = [[],[]]
    for index,student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    return groups

divide_students(students)
########
#zip
######## ayrı listleri bir araya getime
#list(zip(,,,))


############################
#lambda, map, filter, reduce
############################

new_sum = lambda a,b: a +b
new_sum(3,4) #aynı fonksiyon tanımlama gibi

#map
salaries = [1000, 2800, 3000, 4000, 5080]
def new_salary(x):
    return x * 20 / 100 + x
for salary in salaries:
    print(new_salary(salary))

list(map(new_salary,salaries))

list(map(lambda x: x * 20 /100 +x,salaries))#tek satıra sığdırmış olduk

list_store =[1, 2, 3, 4, 5, 6, 7, 8, 9, 18]
list(filter(lambda x: x % 2 == 0, list_store))

wages =[700, 800, 900, 1000]

[wage*1.1 if wage> 950 else wage*1.2 for wage in wages]


###################
#LIST COMPREHENSİON
###################

