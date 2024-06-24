import retro
import numpy as np
import neat
import pickle
import timeit
import pygame

# Dimensões da janela de renderização do Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 900

# Parâmetros de configuração do NEAT
CONFIG_PATH = "configuration_neat.ini"

class NeatAlgorithm:
    # Número de gerações a serem executadas
    generations = 10
    g_fitness = []
    gene = 0
    mean_of_g = np.zeros((generations,))
    # Função para jogar e avaliar o desempenho do agente
    def fitness(self, genomes, config):
        for genome_id, genome in genomes:
            # Cria a rede neural para o genoma atual
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Reinicia o ambiente do jogo
            env = retro.make(game='Airstriker-Genesis')
            obs = env.reset()
            
            done = False
            score = 0
            reward = 0
            vida = 3
            ponto = 0
            shoot_toggle = True
            time_temp = timeit.default_timer()
            while not done:
                # Redimensiona a observação para um tamanho menor, mantendo a proporção 4:3
                obs_resized = np.resize(obs, (5, 10, 3))
                obs_flatten = obs_resized.flatten()
                action = net.activate(obs_flatten)
                action_index = np.argmax(action)
                
                # Mapeia a saída da rede neural para ações específicas
                key_to_action = {
                    0: [1, 0, 0, 0, 0, 0, 0, 0],  # Atirar
                    1: [0, 0, 0, 0, 0, 0, 1, 0],  # Esquerda
                    2: [0, 0, 0, 0, 0, 0, 0, 1],  # Direita  
                    3: [0, 0, 0, 0, 0, 0, 0, 0]   # Descansar
                }

                # Usa o individuo para determinar as direções
                if action_index == 0:
                    # Garante que o individuo atire
                    action = key_to_action[0] if shoot_toggle else key_to_action[3]
                    shoot_toggle = not shoot_toggle
                else:
                    action = key_to_action[action_index]
                
                # Executa a ação e obtém os retornos
                obs, rew, done, info = env.step(action)
                reward += rew
                score = info.get('score', 0)
                fitness = score
                # Verifica se o agente passou de fase
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
            self.g_fitness.append(fitness)
            genome.fitness = fitness
            env.close()

        self.mean_of_g[self.gene] = round(np.mean(self.g_fitness), 2)
        self.g_fitness = []
        self.gene += 1

    # Configuração do NEAT
    def run_neat(self, config_file, generations):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_file)
        
        # Cria a população com a configuração
        population = neat.Population(config)
        
        # Adiciona repórteres para exibir informações do treinamento
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Executa o treinamento do NEAT
        best_genome = population.run(self.fitness, generations)
        
        # Salva o melhor genoma em um arquivo
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(best_genome, f)
        
        return best_genome

    def execute(self):
        # Inicializando Loading Screen
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH/3, SCREEN_HEIGHT/3))
        pygame.display.set_caption('Loading Screen')
        clock = pygame.time.Clock()
        # Exibindo Loading Screen
        self.show_loading_screen(screen, clock)
        best_genome = self.run_neat(CONFIG_PATH, self.generations)
        
        # Carrega a configuração do NEAT para criar a rede neural
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    CONFIG_PATH)
        
        # Cria a rede neural para o melhor genoma
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        
        # Converte a rede neural para uma lista de ações para rodar em playInd()
        individual = []
        env = retro.make(game='Airstriker-Genesis')
        obs = env.reset()
        done = False
        while not done:
            # Redimensiona a observação para um tamanho menor
            obs_resized = np.resize(obs, (5, 10, 3))
            obs_flatten = obs_resized.flatten()
            action = net.activate(obs_flatten)
            action_index = np.argmax(action)
            individual.append(action_index)
            
            key_to_action = {
                0: [1, 0, 0, 0, 0, 0, 0, 0],  # Atirar
                1: [0, 0, 0, 0, 0, 0, 1, 0],  # Esquerda
                2: [0, 0, 0, 0, 0, 0, 0, 1],  # Direita  
                3: [0, 0, 0, 0, 0, 0, 0, 0]   # Descansar
            }
            action = key_to_action[action_index]
            obs, rew, done, info = env.step(action)
            
            if done:
                break

        env.close()

        # Finalizando a Loading Screen
        pygame.quit()

        # Salva o array em um arquivo de texto
        np.savetxt('NeatFitness_mean.txt', self.mean_of_g, fmt='%f')
        np.save('NeatAgent_bestInd.npy', individual)
        return individual
    
    # Exibe uma tela de loading enquanto o NEAT ainda está em execução
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
