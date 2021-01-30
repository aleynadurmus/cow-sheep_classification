import cv2
import numpy as np

def resmiklasordenal(dosyaadi):
    resim=cv2.imread('%s'%dosyaadi)
    return resim



girisverisi=np.array([])

for i in range(2610):
    klasordenalinmisresim=0
    i=i+1
    string='animalset/%s.jpg' %i
    klasordenalinmisresim=resmiklasordenal(string)

    boyutlandirilmisresim=cv2.resize(klasordenalinmisresim,(80,80))
    print(girisverisi)
    girisverisi=np.append(girisverisi,boyutlandirilmisresim)
    print(i+1)

girisverisi=np.reshape(girisverisi,(-1,80,80,3))
np.save("animalset",girisverisi)
print(girisverisi.shape)
