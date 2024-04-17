import numpy as np

# Citirea parametrilor din fisier
def citeste_parametrii_din_fisier(nume_fisier):
    parametrii = {}
    with open(nume_fisier, 'r') as fisier:
        for linie in fisier:
            cheie, valoare = linie.strip().split(':')
            parametrii[cheie.strip()] = valoare.strip()
    return parametrii

# Convertirea si validarea parametrilor
def parseaza_parametrii(parametrii):
    dimensiune_populatie = int(parametrii["Dimensiunea Populatiei"])
    domeniu = np.array(parametrii["Domeniul de Definitie"].split(','), dtype=float)
    coeficienti = np.array(parametrii["Coeficienti Polinom"].split(','), dtype=float)
    precizia = int(parametrii["Precizia"])
    prob_recombinare = float(parametrii["Probabilitate Recombinare"])
    prob_mutatie = float(parametrii["Probabilitate Mutatie"])
    numar_generatii = int(parametrii["Numar Generatii"])
    return (dimensiune_populatie, domeniu, coeficienti, precizia, prob_recombinare, prob_mutatie, numar_generatii)

# Initializarea populatiei
def genereaza_populatie_initiala(dimensiune, domeniu, precizia):
    numar_biti = int(np.ceil(np.log2((domeniu[1] - domeniu[0]) * (10 ** precizia))))
    return np.random.randint(2, size=(dimensiune, numar_biti))

# Decodificarea cromozomului
def decodifica_cromozom(cromozom, domeniu, precizia):
    numar_decimal = int("".join(cromozom.astype(str)), 2)
    max_val = 2 ** len(cromozom) - 1
    return domeniu[0] + (numar_decimal / max_val) * (domeniu[1] - domeniu[0])

# Calculul fitness-ului
def calculeaza_fitness(cromozom, domeniu, coeficienti, precizia):
    x = decodifica_cromozom(cromozom, domeniu, precizia)
    return coeficienti[0] * x**2 + coeficienti[1] * x + coeficienti[2]

# Evaluarea populatiei
def evalueaza_populatia(populatie, domeniu, coeficienti, precizia):
    return np.array([calculeaza_fitness(cromozom, domeniu, coeficienti, precizia) for cromozom in populatie])

# Selectia elitista si selectia generala cu cautare binara
def selectie(populatie, fitness, dimensiune):
    # Selectie elitista
    index_elit = np.argmax(fitness)
    elit = populatie[index_elit]

    # Calcularea probabilitatilor de selectie
    probabilitati_selectie = fitness / np.sum(fitness)
    probabilitati_cumulate = np.cumsum(probabilitati_selectie)

    # Selectia bazata pe cautare binara
    selectati = np.array([elit])
    while len(selectati) < dimensiune:
        u = np.random.random()
        index = np.searchsorted(probabilitati_cumulate, u)
        selectati = np.append(selectati, [populatie[index]], axis=0)
    return selectati

# Crossover cu un punct de taietura
def crossover(populatie, prob_recombinare):
    parents=[]
    for i in range(len(populatie)):
            if np.random.random() < prob_recombinare:
                parents.append(i)  
    
    np.random.shuffle(parents)

    for i in range(0, len(parents) - 1, 2): 
        parent1, parent2 = populatie[parents[i]], populatie[parents[i+1]]
        punct_taietura = np.random.randint(1, len(parent1))  
        child1 = np.concatenate((parent1[:punct_taietura], parent2[punct_taietura:]))
        child2 = np.concatenate((parent2[:punct_taietura], parent1[punct_taietura:]))
        populatie[parents[i]], populatie[parents[i+1]] = child1, child2
    return populatie

# Mutatia
def mutatie(populatie, prob_mutatie):
    for i in range(len(populatie)):
        for j in range(len(populatie[i])):
            if np.random.random() < prob_mutatie:
                populatie[i][j] = 1 - populatie[i][j]
    return populatie

# Algoritmul genetic
def algoritm_genetic(nume_fisier_intrare, nume_fisier_iesire):
    parametrii = citeste_parametrii_din_fisier(nume_fisier_intrare)
    dimensiune_populatie, domeniu, coeficienti, precizia, prob_recombinare, prob_mutatie, numar_generatii = parseaza_parametrii(parametrii)
    
    populatie = genereaza_populatie_initiala(dimensiune_populatie, domeniu, precizia)
    fitness = evalueaza_populatia(populatie, domeniu, coeficienti, precizia)
    
    with open(nume_fisier_iesire, 'w') as fisier:
        # Populatia initiala
        fisier.write("Populatia initiala\n")
        for idx, crom in enumerate(populatie):
            x_val = decodifica_cromozom(crom, domeniu, precizia)
            fit_val = calculeaza_fitness(crom, domeniu, coeficienti, precizia)
            fisier.write(f"  {idx+1:2}: {''.join(map(str, crom))} x= {x_val: .{precizia}f} f={fit_val}\n")

        # Probabilitati selectie
        fisier.write("\nProbabilitati selectie\n")
        probabilitati_selectie = fitness / np.sum(fitness)
        for idx, prob in enumerate(probabilitati_selectie):
            fisier.write(f"cromozom {idx+1:4} probabilitate {prob}\n")

        fisier.write("\nIntervale probabilitati selectie\n")
        probabilitati_cumulate = np.cumsum(probabilitati_selectie)
        interv_selectie = ' '.join([f"{cum_prob}" for cum_prob in probabilitati_cumulate])
        fisier.write(interv_selectie + "\n")

        # Dupa selectie
        populatie_selectata = selectie(populatie, fitness, dimensiune_populatie)
        fisier.write("\nDupa selectie:\n")
        for idx, crom in enumerate(populatie_selectata):
            x_val = decodifica_cromozom(crom, domeniu, precizia)
            fit_val = calculeaza_fitness(crom, domeniu, coeficienti, precizia)
            fisier.write(f"  {idx+1:2}: {''.join(map(str, crom))} x= {x_val: .{precizia}f} f={fit_val}\n")

        parents = []
        fisier.write("\nCromozomii care participa la crossover:\n")
        for i in range(len(populatie_selectata)):
            if np.random.random() < prob_recombinare:
                parents.append(i)  

        np.random.shuffle(parents) 
        for i in range(0, len(parents) - 1, 2): 
            parent1, parent2 = populatie_selectata[parents[i]], populatie_selectata[parents[i+1]]
            punct_taietura = np.random.randint(1, len(parent1))  
            child1 = np.concatenate((parent1[:punct_taietura], parent2[punct_taietura:]))
            child2 = np.concatenate((parent2[:punct_taietura], parent1[punct_taietura:]))
            fisier.write(f"Crossover intre {parent1} si {parent2} la punctul {punct_taietura}:\n")
            fisier.write(f"Parent1: {''.join(map(str, parent1))}\n")
            fisier.write(f"Parent2: {''.join(map(str, parent2))}\n")
            fisier.write(f"Child1: {''.join(map(str, child1))}\n")
            fisier.write(f"Child2: {''.join(map(str, child2))}\n")
            populatie_selectata[parents[i]] = child1
            populatie_selectata[parents[i+1]] = child2

        # After crossover
        fisier.write("\nDupa crossover:\n")
        for idx, crom in enumerate(populatie_selectata):
            x_val = decodifica_cromozom(crom, domeniu, precizia)
            fit_val = calculeaza_fitness(crom, domeniu, coeficienti, precizia)
            fisier.write(f"  {idx+1}: {''.join(map(str, crom))} x= {x_val: .{precizia}f} f={fit_val}\n")


        # Mutatie
        fisier.write("\nDupa mutatie:\n")
        for i in range(len(populatie_selectata)):
            original = populatie_selectata[i].copy()
            for j in range(len(populatie_selectata[i])):
                if np.random.random() < prob_mutatie:
                    populatie_selectata[i][j] = 1 - populatie_selectata[i][j]
                    fisier.write(f"Mutatie la cromozomul {i+1:2}, gena {j+1:2}: de la {'0' if original[j] == 0 else '1':2} la {populatie_selectata[i][j]:2}\n")
            x_val = decodifica_cromozom(populatie_selectata[i], domeniu, precizia)
            fit_val = calculeaza_fitness(populatie_selectata[i], domeniu, coeficienti, precizia)

        # Dupa mutatie
        fisier.write("\nDupa mutatie:\n")
        for i in range(len(populatie_selectata)):
            x_val = decodifica_cromozom(populatie_selectata[i], domeniu, precizia)
            fit_val = calculeaza_fitness(populatie_selectata[i], domeniu, coeficienti, precizia)
            fisier.write(f"  {i+1:2}: {''.join(map(str, populatie_selectata[i]))} x= {x_val: .{precizia}f} f={fit_val}\n")

        fisier.write("\nEvolutie fitness:\n")
        for generatie in range(1, numar_generatii + 1):
            populatie_selectata = selectie(populatie_selectata, fitness, dimensiune_populatie)
            populatie_selectata = crossover(populatie_selectata, prob_recombinare)
            populatie_selectata = mutatie(populatie_selectata, prob_mutatie)
            fitness = evalueaza_populatia(populatie_selectata, domeniu, coeficienti, precizia)
            max_fitness = max(fitness)
            mean_fitness = np.mean(fitness)  
            fisier.write(f"Generatia {generatie:4}: Max Fitness = {max_fitness:20}, Mean Fitness = {mean_fitness:20}\n")

    return populatie_selectata, fitness

# Utilizare:
populatie_finala, fitness_final = algoritm_genetic('entry.txt', 'output.txt')