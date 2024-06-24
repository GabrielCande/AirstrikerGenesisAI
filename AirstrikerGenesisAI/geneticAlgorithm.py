# -------------------------------------------------------------------------------------------------
# import required packages/libraries
# -------------------------------------------------------------------------------------------------

import numpy as np
import random
import retro
import pygame
import timeit
import logging

# Dimensões da janela de renderização do Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 900
# Configuração do logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# -------------------------------------------------------------------------------------------------
# A class for a Genetic Algorithm 
# -------------------------------------------------------------------------------------------------

class GeneticAlgorithm:
    
    # Attributes
    population = []
    # Number of Generations to execute
    numberOfGenerations = 100
    # Population size
    populationSize = 50
    # Mutation rate
    mutationRate = 0.025
    # Crossover rate
    cross_rate = 0.75
    # Best individual returned after the GA is executed
    bestIndividual = []
    # Best fitness value obtained by the best individual
    bestFitness = []
    # Elitism
    elitism = 4
    # Define se a população inicial irá começar com um indivíduo salvo, ou totalmente aleatória
    choice = 0
    
    # Constructor
    def __init__(self):
        # Defina a seed
        np.random.seed(404)
    
    # Generate the initial population
    def generateInitialPopulation(self):
        if self.choice == 0:
            # População inicial é totalmente aleatória
            population = [[np.random.choice([0, 1, 2, 3]) for _ in range(10000)] for _ in range(self.populationSize)]
            logging.info('Started a new population.')
        else:
            # População inicial tem um indivíduo bom que foi salvo anteriormente
            population = [[np.random.choice([0, 1, 2, 3]) for _ in range(10000)] for _ in range(self.populationSize-1)]
            loaded_arr = np.load('GeneticAgent_bestInd.npy')
            population.append(loaded_arr)
            logging.info('Started a new population and added the loaded individual.')
        return population
    
    # Fitness function to evaluate an individual
    def fitnessFunction(self, individual):
        time_temp = timeit.default_timer()
        vida = 3
        ponto = 0
        xmen = individual.copy()
        # Resultado da função run_with_strategy baseado no individuo
        env = retro.make(game='Airstriker-Genesis')
        obs = env.reset()
            
        key_to_action = {
            0: [1, 0, 0, 0, 0, 0, 0, 0],  # Ação para atirar
            1: [0, 0, 0, 0, 0, 0, 1, 0],  # Ação para mover à esquerda
            2: [0, 0, 0, 0, 0, 0, 0, 1],  # Ação para mover à direita  
            3: [0, 0, 0, 0, 0, 0, 0, 0]   # Ação para descansar
        }

        fitness = 0
        reward = 0
        remaining_moves = 0
        shoot_toggle = True

        while True:
            # Usa o individuo para determinar as direções
            if remaining_moves < len(xmen):
                action_index = int(xmen[remaining_moves])
                if action_index == 0:
                    # Garante que o indivíduo atire
                    action = key_to_action[0] if shoot_toggle else key_to_action[3]
                    shoot_toggle = not shoot_toggle
                else:
                    action = key_to_action[action_index]
                remaining_moves += 1
            else:
                break
            
            obs, rew, done, info = env.step(action)
            reward += rew
            score = info.get('score', 0)
            fitness = score

            # Verifica se passou de fase
            if info['lives'] > vida:
                # Recompensa por passar de fase
                ponto += 100
                vida = info['lives']
            else:
                vida = info['lives']

            if done:
                time_temp = (time_temp - timeit.default_timer())
                fitness = ((fitness * 1.2) + ponto + (reward * 0.4)) * 0.8 + (1 / time_temp) * 0.2
                break

        env.close()
        return fitness     

    # Função que seleciona os pais por torneio
    def selectParents(self, population_fitness, K):
        # Seleciona K indivíduos aleatórios
        contenders = np.random.choice(len(population_fitness), K)
        # Pega o fitness destes K indivíduos selecionados
        fitnessContenders = population_fitness[contenders]
        # Seleciona o indivíduo que possui o maior fitness entre os K
        selected = contenders[np.argmax(fitnessContenders)]
        # Retorna a posição deste indivíduo na população
        return(selected)
    
    # Função de crossover de um ponto
    def crossover(self, parent1, parent2):
        # Gera um número aleatório para o ponto de corte
        crossover_point = np.random.randint(1, len(parent1)-1)
        # Gera um número aleatório entre 0 e 1
        cr = np.random.rand()
        # Verifica se o número gerado aleatoriamente entre 0 e 1 é maior que 0.75 (cross_rate)
        if(cr > self.cross_rate):
            child1 = parent1.copy()
            child2 = parent2.copy()
        else:
            # Realiza o crossover de um ponto para criar os filhos    
            child1 = np.concatenate([parent1[: (crossover_point-1)],parent2[(crossover_point - 1):]])
            child2 = np.concatenate([parent2[: (crossover_point-1)],parent1[(crossover_point - 1):]])
        return child1, child2
    
    # Função de mutação uniforme
    def mutationOperator(self, individual):
        ind_copy = individual.copy()
        for i in range(len(individual)):
            # Gera um número aleatório de 0 a 1
            prob = np.random.rand()
            # Verifica se o número gerado é menor ou igual ao mutationRate
            if prob <= self.mutationRate:
                # Substitui o movimento no individuo de maneira aleatória
                ind_copy[i] = random.choice([0, 1, 2, 3])
        return ind_copy
    
    # Função que coordena o algoritmo genético
    def execute(self):
        logging.info('Initializing the genetic algorithm.')
        # Inicializando Loading Screen
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH/3, SCREEN_HEIGHT/3))
        pygame.display.set_caption('Loading Screen')
        clock = pygame.time.Clock()
        # Exibindo Loading Screen
        self.show_loading_screen(screen, clock)
        # Cria a população inicial
        population = self.generateInitialPopulation()
        population = np.array(population) 
        # Cria o vetor de fitness da população
        population_fitness = np.array([self.fitnessFunction(individual) for individual in population])
        # Controle de erro, caso exista um None em alguma posição do vetor de fitness, troca para 0
        population_fitness = np.where(population_fitness == None, 0, population_fitness)
        # Cria o vetor de fitness médio das gerações e salva os valores médios na posição controlada por 'gene' que representa a geração
        gene = 0
        # Salva os fitness médios por geração
        mean_of_g = np.zeros((self.numberOfGenerations + 1,))
        mean_of_g[gene] = round(np.mean(population_fitness), 2)
        for generation in range(self.numberOfGenerations):
            logging.info(f'------Starting Generation {generation}------.')
            child_pop = np.zeros_like(population)
            # Incrementa um na geração
            gene += 1
            # Cria os filhos da próxima geração - a quantidade de elitismo
            for k in range((self.populationSize - self.elitism)//2):
            # Pega a posição do pai1 e do pai2 que "venceram" o torneio de Seleção
                id_parent1 = self.selectParents(population_fitness, K=10)
                id_parent2 = self.selectParents(population_fitness, K=10)
                # ch1 e ch2 recebem os filhos gerados a partir do CrossOver entre o pai1 e o pai2
                ch1, ch2 = self.crossover(population[id_parent1], population[id_parent2])
                # Recebe cada filho após passar pela função de Mutação
                child1 = self.mutationOperator(ch1)
                child2 = self.mutationOperator(ch2)
                # Adiciona cada filho gerado na população de filhos
                child_pop[2*k] = child1
                child_pop[2*k+1] = child2
            # Verifica a posição dos maiores fitness para propagar para a próxima geração (elitismo)
            elites = np.argpartition(population_fitness, -self.elitism)[-self.elitism:]
            # Adiciona ao final da população de filhos os melhores individuos da população anterior
            child_pop[-self.elitism:] = population[elites]
            # População de filhos se torna a população
            population = child_pop
            # Verifica o fitness da nova população
            population_fitness = np.array([self.fitnessFunction(individual) for individual in population])
            population_fitness = np.where(population_fitness == None, 0, population_fitness)
            # Adiciona o valor do fitness médio na posição da geração
            mean_of_g[gene] = round(np.mean(population_fitness), 2)
        # Pega a posição do melhor valor de fitness
        best_id = np.argmax(population_fitness)
        # Pega o melhor indivíduo da população a partir da posição achada acima
        self.bestIndividual = population[best_id]
        # Pega o melhor valor do fitness
        self.bestFitness = population_fitness[best_id]
        # Cria um arquivo de texto com o vetor de fitness médio das gerações
        np.savetxt('GeneticFitness_mean.txt', mean_of_g, fmt='%f')
        logging.info('Fitness mean per generation saved.')
        # Salva o melhor indivíduo e a última população
        np.save('GeneticAgent_bestInd.npy', self.bestIndividual)
        logging.info('Best individual saved.')
        np.save('GeneticLastPop.npy', population)
        logging.info('Last population saved.')
        print("Last Population Fitness: \n", population_fitness)
        print("Best Fitness: \n", self.bestFitness)
        logging.info('Genetic algorithm finished.')
        # Finalizando a Loading Screen
        pygame.quit()
        return self.bestIndividual
    
    # Função que da tela de loading
    @staticmethod
    def show_loading_screen(screen, clock):
        """
        Simple Loading Screen
        """
        font = pygame.font.Font(None, 80)
        loading_text = font.render("Loading...", True, (255, 255, 0))
        text_rect = loading_text.get_rect(center=((SCREEN_WIDTH/3) // 2, (SCREEN_HEIGHT/3) // 2))
        screen.fill((0, 0, 0))
        screen.blit(loading_text, text_rect)
        pygame.display.flip()

        # Espera um momento para simular carregamento
        pygame.time.wait(1000)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------