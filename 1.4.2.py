
try:
    grade=float(input("Unesi broj od 0.0 do 1.0: "))
except:
        print("Nisi unio broj")
if grade<0.0 or grade>1.0:
    print("ocjena nije ispravna")  

while(grade<=0.0 or grade>=1.0):
    grade=float(input("ponovno unesite ocjenu:"))

if(grade>=0.9):
    print("A")
elif(grade>=0.8 and grade<0.9):
    print("B")
elif(grade>=0.7 and grade<0.8):
    print("C")
elif(grade>=0.6 and grade<0.7):
    print("D")
else: 
    print("F")