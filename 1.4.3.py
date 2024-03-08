
list=[]
while True:
    x=input("Unesi broj: ")
    if(x=="Done"):
        break
    try:
        x=int(x)
    except:
        print("Unesi broj!!")
        continue


    list.append(x)
print("Unešeno je: ", len(list), "brojeva")

print("Minimalna vrijednost: ", min(list))
print("Maksimalna vrijednost: ", max(list))

print("Srednja: ",sum(list)/len(list) )
list.sort()
print(list)