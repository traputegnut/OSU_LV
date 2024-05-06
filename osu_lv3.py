#LV3---------------------------------

#3.4.1 
import pandas as pd
data=pd.read_csv("OSU_LV\LV3\data_co2_emission.csv")
#a Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili 
#  duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veličine konvertirajte u tip category.
print("broj mjerenja:",len(data))
print(data.info())
print(data.dropna())
print(data.drop_duplicates())

data["Make"]=data["Make"].astype("category")
data["Model"]=data["Model"].astype("category")
data["vehicle class"]=data["vehicle class"].astype("category")
data["Transmission"]=data["Transmission"].astype("category")
data["Fuel type"]=data["Fuel type"].astype("category")

print(data.info())

#b Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
#  ime proizvodaca, model vozila i kolika je gradska potrošnja.
df=data.sort_values(by="Fuel consumption City (L/100km)",ascending=False)
print(df[["Make","Model","Fuel consumption city (L/100km)"]].head(3))
print(df[["make","Model","Fuel consumption city (L/100km)"]].tail(3))

#c Koliko vozila ima velicinu motora između 2.5 i 3.5 L? Kolika je prosjecna C02 emisija plinova za ova vozila?
df=data[(data['Engine Size (L)']>=2.5) & (data['Engine Size (L)'] <=3.5)]
print("broj vozila sa zapreminom motora vecom od 2.5,a manjom od 3.5L je ",len(df))
print("prosjecna emisija co2 za ova vozila je ",df["c02 emissions (g/km)"].mean())

#d  Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara? 
df=data[data['Make']=='Audi']
print(df)
print("broj audi vozila:",len(df))
df=df[df['Cylinders']==4]
print("prosjecna emisija co2 audi vozila s 4 cilindra ",df['CO2 emissions (g/km)'].mean())

#e Koliko je vozila s 4,6,8... cilindara?Kolika je prosječna emisija CO2 s obzirom na broj cilindara
df=data.groupby('Cylinders')
print(df.agg('count')[['Model']])
print(df['CO2 emissions (g/km)'].mean())

#f Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
df=data.groupby('Fuel type')
print(df['Fuel consumption city (L/100km)'].mean())
print(df['Fuel Consumption City (L/100km)'].median())

#g Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?
df=data[(data['Fuel type']=='D') & (data['Cylinders']==4)]
print(df.sort_values(by='Fuel consumption city (L/100km)',ascending=False).head(1)[["Make","Model","Fuel consumption city (L/100km)"]])

#h Koliko vozila ima ručni mjenjač
df=data[(data["Transmission"]=='M5') | (data["Transmission"]=='M6') | (data["Transmission"]=='M7')]
print("broj vozila s rucnim mjenjacem:", len(df))

#i Izracunajte korelaciju izmedu numerickih velicina.
print(data.corr(numeric_only=True))

#3.4.2--Prikazi sljedece vizualizacije
import pandas as pd
import matplotlib.pyplot as plt

#a Pomocu histograma prikažite emisiju C02 plinova
plt.figure()
data['CO2 Emissions (g/km)'.plot(kind='hist',bins=20)]
plt.show()

#b Pomocu dijagrama raspršenja prikažite odnos izmedju gradske potrošnje goriva i emisije C02 plinova
df1=data[data['Fuel type']=='E']
df2=data[data['Fuel type']=='D']
df3=data[data['Fuel type']=='Z']
df4=data[data['Fuel type']=='X']
plt.scatter(df1["Fuel consumption city (L/100km)"], df1['CO2 Emissions (g/km)'],c='blue',s=0.5)
plt.scatter(df2["Fuel Consumption City (L/100km)"], df2['CO2 Emissions (g/km)'], c='black', s=0.5)
plt.scatter(df3["Fuel Consumption City (L/100km)"], df3['CO2 Emissions (g/km)'], c='green', s=0.5)
plt.scatter(df4["Fuel Consumption City (L/100km)"], df4['CO2 Emissions (g/km)'], c='red', s=0.5)
plt.show()

#c Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva
data.boxplot(column=['Fuel Consumption hwy (L/100km)',by='Fuel Type'])
plt.show()

#d Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu groupby.
df=data.groupby("Fuel type")
df.agg('count')[['Model']].plot(kind="bar")
plt.show()

#e Pomocu stupcastog grafa prikažite na istoj slici prosjecnu C02 emisiju vozila s obzirom na broj cilindara.
df=data.groupby('Cylinders')
df.agg('mean')[['CO2 emissions (g/km)']].plot(kind="bar")
plt.show()


