
## Har arbeidet/hjulpet Nils med denne oppgaven

from random import shuffle
import os.path
import os

spar='\u2660'
ruter='\u2666'
kløver='\u2663'
hjerter='\u2665'

bunker = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[]}

kortStokk = []

cls = lambda: os.system('cls')

#### Kort class ####
## Brukte tutorial for kort classen og kort funksjoner https://medium.com/@anthonytapias/build-a-deck-of-cards-with-oo-python-c41913a744d3

class Kort:

    def __init__(self, kortType, verdi, bunke = ''):
        self.kortType = kortType
        self.verdi = verdi
        self.bunke = bunke

    def vis(self):
        if self.kortType == 'spar':
            suit = '\u2660'
        elif self.kortType == 'ruter':
            suit = '\u2666'
        elif self.kortType == 'kløver':
            suit = '\u2663'
        elif self.kortType == 'hjerter':
            suit = '\u2665'
        return str((self.verdi, suit))

#### Kort funksjoner ####
    
def byggNyttSpill(stokk):
    for t in ['spar', 'ruter', 'kløver', 'hjerter']:
        for v in ('A', '7' ,'8', '9', '10', 'J', 'Q', 'K'):
            stokk.append(Kort(t, v))

def byggLagret(bunke, verdi, kortType):
    kortStokk.append(Kort(kortType, verdi, bunke))
    
def stokke(kort):
    shuffle(kort)

def trekkKort(kort):
    return kortStokk.pop()

#### Bunkefunksjoner ####

def lagBunker():
    global bunker
    bunker = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[]}
    for key in bunker:
        for kort in range(1, 5):
            bunker[key].append(trekkKort(kort))
    return bunker

def lagLagret():
    global bunker
    kortStokk.reverse()
    bunker = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[]}
    while len(kortStokk) > 0:
        for key in bunker:
            try:
                if kortStokk[-1].bunke == key:
                    bunker[key].append(trekkKort(kortStokk))
            except:
                break
    
def visBunker():
    for key in bunker:
        print(f'{key:^11}', end='')
    print('\n')
    for key in bunker:
        if len(bunker[key]) != 0:
            print(f'{(bunker[key])[-1].vis():^11}', end='')
        else:
            print('          ', end='')
    print('\n')
    for key in bunker:
        print(f'{len(bunker[key]):^11}', end ='')

#### Spillfunksjoner ####

def sjekkVinn():
    x = 0
    for key in bunker:
        if len(bunker[key]) == 0:
            x += 1
    if x == 8:
        return True


def sjekkSpill():
    for key in bunker:
        if len(bunker[key]) != 0:
            for key2 in bunker:
                if key != key2:
                    if len(bunker[key2]) != 0:
                        if (bunker[key])[-1].verdi == (bunker[key2])[-1].verdi:
                            return True
    return False

def sjekkLikeKort(en, to):
    if (bunker[en])[-1].verdi == (bunker[to])[-1].verdi:
        return True


def fjernKort(en, to):
    try:
        if sjekkLikeKort(en, to):
            bunker[en].pop()
            bunker[to].pop()
        else:
            print('Kortene er ikke like, velg på nytt')
    except:
        print('Ikke mulig valg')
        
def spilles():
    print('Trekk to kort (x - avbryt')
    while sjekkSpill():
        cls()
        visBunker()
        kommando = input('\n Velg bunker: ')
        if kommando == 'x':
            meny()
            break
        else:
            try:
                en = kommando[0]
                to = kommando[1]
                fjernKort(en.upper(), to.upper())
            except:
                print('Ikke mulig valg')
        if sjekkSpill() is False:
            visBunker()
            if sjekkVinn() is True:
                print('\n Du vant!')
                kommando = input('1 for hovedmeny, x for å avslutte: ')
                if kommando == '1':
                    meny()
                    break
                elif kommando == 'x':
                    kommando = 5
                    return kommando
            print('\n Ingen flere mulige trekk, du tapte')
            kommando = input('1 for hovedmeny, x for å avslutte: ')
            if kommando == '1':
                meny()
                break
            elif kommando == 'x':
                kommando = 5
                return kommando

#### Lagre og hente funksjoner ####

def lagre():
    fil = open('lagretspill.txt', 'w')
    tempListe = []
    for key in bunker:
        for k in range(len(bunker[key])): 
            tempListe.append(key)
            tempListe.append(bunker[key][k].verdi)
            tempListe.append(bunker[key][k].kortType)
    for temp in tempListe:
        fil.write(temp + '\n')
    fil.close()

def hent():
    if os.path.exists('lagretspill.txt'):
        fil = open('lagretspill.txt', 'r',)
        temp1 = fil.readline().strip()
        temp2 = None
        temp3 = None
        while temp1 != '':
            try:
                temp2 = fil.readline().strip()
                temp3 = fil.readline().strip()
                byggLagret(temp1, temp2, temp3)
            except:
                pass
            temp1 = fil.readline().strip()
        fil.close()
        lagLagret()
        spilles()
    else:
        print('Ingen lagret spill')
            
#### hoved ####

def regler():
    print('Finn 2 like kort')
    print('Velg bunkene ved å skrive navnet på begge bunkene, som er de øverste bokstavene, så trykk enter')

def meny():
    print('--- Wish Kabal -------')
    print('1 - Start nytt spill')
    print('2 - Lagre spill')
    print('3 - Hent lagret spill')
    print('4 - Hvordan spille')
    print('5 - Avslutt')
    print('----------------------')

def start():
    meny()
    kommando = ''
    while kommando != '5':
        kommando = input('Velg >')
        if kommando == '0':
            meny()
        elif kommando == '1':
            byggNyttSpill(kortStokk)
            stokke(kortStokk)
            lagBunker()
            spilles()
        elif kommando == '2':
            lagre()
        elif kommando == '3':
            hent()
        elif kommando == '4':
            regler()
        elif kommando == '5':
            break
        else:
            print('Ukjent handling')

start()
