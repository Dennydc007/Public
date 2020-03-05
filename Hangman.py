import os

cls = lambda: os.system('cls')

### ---Regler--- ###
def nyttOrd(riktigOrdListe):
    nyttOrd = input('Skriv et ord: ')
    for i in nyttOrd:
            riktigOrdListe.append(i)
    return riktigOrdListe

def gjetting(bokstav):
    bokstav = input('Gjett en bokstav: ')
    return bokstav

def riktigBokstav(bokstav, riktigOrdListe):
    if bokstav in riktigOrdListe:
        return True
    else:
        return False

### ---ASCI--- ###
def tapprint(tap):
    if tap == 1:
        print(' _ ___  ')
        print('/     | ')
    elif tap == 2:
        print('  |     ')
        print('  |     ')
        print('  |     ')
        print('  |     ')
        print('  |     ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 3:
        print('  |---  ')
        print('  |     ')
        print('  |     ')
        print('  |     ')
        print('  |     ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 4:
        print('  |---  ')
        print('  |  |  ')
        print('  |     ')
        print('  |     ')
        print('  |     ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 5:
        print('  |---  ')
        print('  |  |  ')
        print('  |  o  ')
        print('  |     ')
        print('  |     ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 6:
        print('  |---  ')
        print('  |  |  ')
        print('  |  o  ')
        print('  |  |  ')
        print('  |     ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 7:
        print('  |---  ')
        print('  |  |  ')
        print('  |  o  ')
        print('  | /|  ')
        print('  |     ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 8:
        print('  |---  ')
        print('  |  |  ')
        print('  |  o  ')
        print('  | /|| ')
        print('  |     ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 9:
        print('  |---  ')
        print('  |  |  ')
        print('  |  o  ')
        print('  | /|| ')
        print('  | /   ')
        print(' _|___  ')
        print('/     | ')
    elif tap == 10:
        print('  |---  ')
        print('  |  |  ')
        print('  |  o  ')
        print('  | /|| ')
        print('  | / | ')
        print(' _|___  ')
        print('/     | ')

### ---Print--- ###
def skriv(lengde, brukte):
    for i in lengde:
        if i not in brukte:
            print('_', end=' ')
        else:
            print(i, end=' ')
    print('\n', brukte)
    print('\n')

### ---Spill--- ###
def spilles():
    riktigOrd = ''
    riktigOrdListe = []
    bokstav = ''
    brukte = []
    tap = 0
    feilBokstaver = []
    riktigBokstaver = []
    if riktigOrd == '':
        riktigOrdListe = nyttOrd(riktigOrdListe)
    print(riktigOrd)
    while tap != 10 and riktigBokstaver != riktigOrdListe:
        cls()
        tapprint(tap)
        skriv(riktigOrdListe, brukte)
        if bokstav == '' or bokstav in brukte:
            bokstav = gjetting(bokstav)
            if riktigBokstav(bokstav, riktigOrdListe):
                if bokstav not in riktigBokstaver:
                    riktigBokstaver.append(bokstav)
            elif bokstav not in brukte:
                feilBokstaver.append(bokstav)
                tap += 1
            if bokstav not in brukte:
                brukte.append(bokstav)
    if sorted(riktigBokstaver) == sorted(riktigOrdListe):
        cls()
        print('Du vant!')
    else:
        cls()
        tapprint(tap)
        print('Du tapte')
def meny():
    print('--------------')
    print('1 Nytt spill')
    print('2 Avslutt')
    print('--------------')

def smeny():
    cls()
    kom = ''
    while kom != '2':
        meny()
        kom = input('Valg >')
        if kom == '1':
            spilles()
        elif kom == '2':
            break
        else:
            print('Ukjent valg')

smeny()

