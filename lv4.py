#LV4----------------------------------------

#4.5.1
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

data = pd.read_csv("LV4\data_C02_emission.csv")

X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]

y = data[['CO2 Emissions (g/km)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

plt.scatter(X_train[['Fuel Consumption Comb (L/100km)']], y_train, marker='.', color="blue", s=1)
plt.scatter(X_test[['Fuel Consumption Comb (L/100km)']], y_test, marker='.', color="red", s=1)
plt.show()

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

f, (ax1, ax2) = plt.subplots(2,1)
ax1.hist(X_train['Fuel Consumption Comb (L/100km)'], bins=20)
ax2.hist(X_train_n[4], bins=20)
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)

y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p, marker='.', s=1)
plt.show()

MSE = mean_squared_error(y_test, y_test_p)
RMSE = sqrt(MSE)
MAE = mean_absolute_error(y_test, y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2 = r2_score(y_test, y_test_p)

print("MSE:", MSE, " RMSE:", RMSE, " MAE:", MAE, " MAPE:", MAPE, " R2:", R2)



#4.5.2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
import numpy as np

data = pd.read_csv("LV4\data_C02_emission.csv")

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data['Fuel Type'] = X_encoded

X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel Type']]


y = data[['CO2 Emissions (g/km)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

y_test_p = linearModel.predict(X_test)

max_pogreska = 0
max_pogreska_loc = 0

y_true = np.array(y_test)

for i in range(len(y_test)):
    pogreska = abs(y_true[i] - y_test_p[i])
    if(pogreska > max_pogreska):
        max_pogreska = pogreska
        max_pogreska_loc = i

print("Najveca pogreska(g/km):", max_pogreska, data.iloc[max_pogreska_loc][['Make']], data.iloc[max_pogreska_loc][['Model']])