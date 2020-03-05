
import weakref

fagKoder =[['informasjon vitenskap', 'INF'], ['økonomi', 'ECO'], ['geologi', 'GEO']]
Emner=[]
#### Klasser ####
class emneClass:

    instances=[]
    
    def __init__(self, navn, kar=''):
        self.__class__.instances.append(weakref.proxy(self))
        Emner.append(self)
        self.navn=navn
        self.karakter=kar

    def getnavn(self):
        return self.navn
        
#### Filbehandling ####

fil=open('Semester.txt', encoding='utf-8')
temp=fil.readline().strip()
tempListe=[]
while temp != '':
    tempListe.append(temp)
    temp=fil.readline().strip()
for i, a in enumerate(tempListe):
    if a not in ('A', 'B', 'C', 'D', 'E', 'F'):
        if tempListe[i+1] in ('A', 'B', 'C', 'D', 'E', 'F'):
            tempListe[i] = emneClass(a, tempListe[i+1])
        else:
            tempListe[i] = emneClass(a)
fil.close
    #### Emne-funksjoner ####
    # Lånt fra løsningsforslag
    
def leggTilEmne():
    e=input('Nytt emne: ')
    temp=[]
    temp.append(e)
    for i, a in enumerate(temp):
        temp[i] = emneClass(a)

def sjekkEmne(emne):
    for e in Emner:
        if e.navn == emne:
            return True
    return False        

def lesEmne():
    e=input('Emne: ')
    while not sjekkEmne(e):
        e=input('Oppgi et korrekt emne (? for emneliste): ')
        if e=='?':
            for i in Emner: print(i.navn)
    return e

def skrivEmner(emneliste):
    for e in sorted(emneliste):
        print(e)
        
def emnenivå(emne):
        n=len(emne)
        return emne[n-3]+'00'

def finnEmner(fag='', nivå=''):

    def sjekkFagområde(emne):
        return(fag=='' or (fag==fagområde(emne)[0] or fag[:3]==fagområde(emne)[1]))

    def sjekkNivå(emne):
        return(nivå=='' or nivå==emnenivå(emne))

    resultat=[]
    
    for e in Emner:
        if sjekkFagområde(e.navn) and sjekkNivå(e.navn):
            resultat.append(e.navn + ' ' + e.karakter)
    return resultat

    #### fag-funksjoner ####
    # Lånt fra løsningsforslag
    
def fagkode(emne):
        return emne[0]+emne[1]+emne[2]

def fagområde(emne):
    if not sjekkEmne(emne):
        print (emne, 'finnes ikke')
        return None
    f=fagkode(emne)
    for e in fagKoder:
        if e[1]==f: return e

    #### Karakter-funksjoner ####
    # Lånt fra løsningsforslag
            
def lesKarakter(melding='Karakter: '):
    k=input(melding)
    while not (k=='' or k=='A' or k=='B' or k=='C' or k=='D' or k=='E' or k=='F'):
        k=input('Oppgi en bokstav A-F: ')
    return k

def settKarakter(emne):
    k=input('Sett ny karakter (<enter for å slette>): ')
    while not (k=='' or k=='A' or k=='B' or k=='C' or k=='D' or k=='E' or k=='F'):
        k=input('Oppgi en bokstav A-F: ')
    for i in Emner:
        if i.navn == emne:
            setattr(i, 'karakter', k)

    #### Snitt-beregning ####
    # Lånt fra løsningsforslag
    
def poengSnitt(emneliste):
    antall=0
    sum=0
    for e in Emner:
        k=e.karakter
        if k!='':
            antall=antall+1
            sum=sum+karakterTilPoeng(k)
    if antall==0:
        return 0
    return sum/antall

def snittKarakter(emneliste):
    return poengTilKarakter(poengSnitt(emneliste))

def karakterTilPoeng(karakter):
    if karakter=='A': return 5
    elif karakter=='B': return 4
    elif karakter=='C': return 3
    elif karakter=='D': return 2
    elif karakter=='E': return 1
    else: return 0

def poengTilKarakter(poeng):
    p=round(poeng)
    if p==0: return 'F'
    elif p==1: return 'E'
    elif p==2: return 'D'
    elif p==3: return 'C'
    elif p==4: return 'B'
    elif p==5: return 'A'

    #### Lagre fil ####
def lagre():
    fil=open('Semester.txt', 'w', encoding='utf-8')
    for i in Emner:
        fil.write(i.navn + '\n')
        if i.karakter != '':
            fil.write(i.karakter + '\n')
    fil.close

#### hovedprogram ####
# Lånt fra løsningsforslag

def meny():
    print('--------------------')
    print(' 1 Emneliste ')
    print(' 2 Legg til emne ')
    print(' 3 Sett karakter ')
    print(' 4 Karaktersnitt ')
    print(' 5 Avslutt ')
    print('--------------------')

def start():
    meny()
    kommando=''
    while kommando!='5':
        kommando=input('Velg handling (0 for meny)> ')
        if kommando=='0':
            meny()
        elif kommando=='1':
            print('Velg fag og/eller emnenivå (<enter> for alle)')
            f=input(' - Fag: ')
            n=input(' - Nivå: ')
            skrivEmner(finnEmner(f, n))
        elif kommando=='2':
            leggTilEmne()
        elif kommando=='3':
            e=lesEmne()
            settKarakter(e)
        elif kommando=='4':
            print('Velg fag og/eller emnenivå (<enter> for alle)')
            f=input(' - Fag: ')
            n=input(' - Nivå: ')
            e=finnEmner(f, n)
            if e==[]:
                print('Tomt emneutvalg')
            else:
                print('Snitt:', snittKarakter(e))
        elif kommando=='5':
            svar=input('Vil du lagre endringer? (j for ja): ')
            if svar == 'j':
                lagre()
            else:
                pass
        else: print('Ukjent handling')
start()
