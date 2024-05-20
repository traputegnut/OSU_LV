#LV1--------------------------------------------
#1.4.1.
print("unesite broj radnih sati")
hour=int(input())
print("unesite satnicu")
perHour=float(input())
print("iznos:",+ float(hour*perHour))

def CalculateWage(hour,perHour):
    return hour*perHour

print(CalculateWage(hour,perHour),'eura')

#1.4.2.
try:
    print("unesi broj:")
    broj=float(input())
except:
    print("nije unesen broj")
if broj<0.0 or broj>1.0:
    print("broj izvan intervala")
elif broj<0.6:
    print("F")
elif broj>0.6 and broj<0.7:
    print("D")
elif broj>0.7 and broj<0.8:
    print("C")
elif broj>0.8 and broj<0.9:
    print("B")
else:
    print("A")

#1.4.3.
def averageValue(brojevi):
    sum=0
    for broj in brojevi:
        sum+=sum
    return sum/len(brojevi)

brojevi=[]
while True:
    unos=input()
    if(unos="Done"):
        break
    try:
        broj=int(unos())
    except:
        print("nije unesen broj")
        continue
    else:
        brojevi.append(broj)

print("Korisnik je unio", len(brojevi), "brojeva.")
print("Srednja vrijednost:", srednja_vrijednost(brojevi))
print("Min:", min(brojevi))
print("Max:", max(brojevi))
brojevi.sort()
print(brojevi)


#1.4.4.
words=[]
file=open("song.txt")
for line in file:
    current_words=line.split()
    for word in current_words:
        word.lower()
        words.append(word)
file.close()

dictionary=dict()

for word in words:
    dictionary[word]=words.count(word)

count=0
for word in dictionary:
    if dictionary[word]==1:
        count+=1
        print(word)

print("ima",count,"rijeci koje se pojavljuje jednom")

#1.4.5.
def prosjecan_broj_rijeci(messages):
    count=0
    length=0
    for message in messages:
        length+=len(message)
        words=message.split()
        count+=len(words)
    return length/count

hams=[]
spams=[]

file=open("SMSspamCollection.txt")
for line in file:
    line=line.rsplit(" ")
    if line[0]=="ham":
        hams.append(line[1])
    else:
        spams.append(line[1])

print("prosjecan broj rijeci ham:",prosjecan_broj_rijeci(hams))
print("prosjecan broj rijeci spams",prosjecan_broj_rijeci(spams))

count=0
for spam in spams:
    if spam.endswith("!"):
        count+=1
    
    print(count," spam poruka zavrsava s ukslicnikom")


#LV2-----------------------------------------------------------------------------------------------------------------------------
#2.4.1 crtanje slike
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,3,1])
y=np.array([1,2,2,1,1])
plt.plot(x,y,'b',linewidth=1,marker=".",markersize=10)
plt.exis([0.0,4.0,0.0,4.0])
plt.xlabel("x os")
plt.ylabel("y os")
plt.title("primjer")
plt.show


#2.4.2 data.csv sadrži mjerenja visine i mase na M i Ž.Učitaj podatke u numpy polje
import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt("LV2\data.csv",delimiter=",",names=True)

#a Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja?
print("mjerenja izvrsena na ",len(data),"osoba.")
#b Prikažite odnos visine i mase osobe pomocu naredbe ´ matplotlib.pyplot.scatter
plt.scatter(data["Height"],data["Weight"],mareker=".",linewidths=0.1)
plt.xlabel("visina u cm")
plt.ylabel("masa [kg]")
plt.title("odnos visine i mase")
plt.show()
#c  Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici
plt.scatter(data["height"][0::50],data["weight"][0::50],marker=".",linewidths=0.1)
plt.xlabel("visina u cm")
plt.ylabel("masas u kg")
plt.title("odnos visinei mases")
plt.show()
#d Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu
print("minimalna visina",np.min(data["height"]))
print("max",np.max(data["height"]))
print("mean",np.mean(data["height"]))
#e Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. 
ind_m=(data[:,0]==1)
ind_z=(data[:,0]==0)
print("muskarci:")
print("min",np.min(data["height"],where=ind_m))
print("max",np.max(data["height"],where=ind_m))
print("mean",np.mean(data["height"],where=ind_m))
print("zene")
print("min",np.min(data["height"],where=ind_z))
print("max",np.max(data["height"],where=ind_z))
print("mean",np.mean(data["height"],where=ind_z))


#2.4.3 Skripta zadatak_3.py ucitava sliku ’ ˇ road.jpg’.
import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("LV2\\road.jpg")
img=img[:,:,0].copy()

plt.figure()

#a posvijetliti sliku
plt.imshow(img,cmap="gray",alpha=0.5)
plt.show()
#b prikazati samo drugu cetvrtinu slike po širini
plt.imshow(img[:,160:320:],cmap="gray")
plt.show()
#c zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
plt.imshow(np.rot90(img),cmap="gray")
plt.show()
#d zrcaliti sliku
plt.imshow(np.flip(img,axis=1),cmap="gray")
plt.show()

#2.4.4 Napišite program koji ce kreirati sliku koja sadrži ´ cetiri kvadrata crne odnosno bijele boje 
import numpy as np
import matplotlib.pyplot as plt

black=np.zeroes((50,50))
white=np.ones((50,50))

first=np.hstack([black,white])
second=np.hstack([black,white])

img=np.vstack([first,second])

plt.figure()
plt.imshow(img,cmap="gray")
plt.show()






#LV8-------------------------------------------------------------
# učitavanje podataka:
data_df = pd.DataFrame(data, columns=['num_pregnant', 'plasma', 'blood_pressure',
                       'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['diabetes']).to_numpy()
y = data_df['diabetes'].copy().to_numpy()

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# a)
model = keras.Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Dense(units=12, activation="relu"))
model.add(layers.Dense(units=8, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.summary()

# b)
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy", ])

# c)
history = model.fit(X_train, y_train, batch_size=10,
                    epochs=150, validation_split=0.1)


# d)
model.save('Model/')

# e)
model = load_model('Model/')
score = model.evaluate(X_test, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

# f)
y_predictions = model.predict(X_test)
y_predictions = np.around(y_predictions).astype(np.int32)
cm = confusion_matrix(y_test, y_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
# izvrši predikciju mreže
