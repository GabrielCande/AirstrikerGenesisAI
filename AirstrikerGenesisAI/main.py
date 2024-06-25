# Alunos: Gabriel Candelária Wiltgen Barbosa, 2271958
#         Camila Costa Durante, 2270196

import retro
import pygame
import numpy as np
from geneticAlgorithm import GeneticAlgorithm
from neatAlgorithm import NeatAlgorithm
from ppo_agent import PPOAgent
import time
import logging

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PARÂMETRO DE CONTROLE 
controller = 3
# 0 -> Roda o jogo para jogar
# 1 -> Roda o agente do algoritmo genético
# 2 -> Roda o agente do algoritmo neat
# 3 -> Roda o agente do PPO
# 4 -> Mostra o indivíduo recebido na variável a seguir:
indLoader = np.load('NeatAgent_bestInd.npy') # Precisa ser um numpy array em formato de arquivo .npy

# Dimensões da janela de renderização do Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 900
# Configuração do logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------Métodos---------------------------------------------------------------------------------------------

def gameplay():
    # Criando o ambiente do jogo Retro
    env = retro.make(game='Airstriker-Genesis')

    # Inicializando o Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Airstriker-Genesis')

    # Reiniciando o ambiente e obtendo a observação inicial
    obs = env.reset()

    clock = pygame.time.Clock()

    # Dicionário para mapear teclas do Pygame para ações do Retro
    key_to_action = {
        pygame.K_LEFT: 6,    # Ação para mover à esquerda
        pygame.K_RIGHT: 7,   # Ação para mover à direita
        pygame.K_z: 0        # Ação para atirar
    }

    # Inicializando a ação com zeros (sem ação)
    action = np.zeros(env.action_space.shape[0], dtype=np.int8)

    # Array para salvar os movimentos jogados
    recorded_individual = []

    # Loop principal do jogo
    while True:
        # Processamento de eventos do Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action[key_to_action[event.key]] = 1  # Ativa a ação
            elif event.type == pygame.KEYUP:
                if event.key in key_to_action:
                    action[key_to_action[event.key]] = 0  # Desativa a ação

        # Salva os movimentos do jogador
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            recorded_individual.append(0)
        if np.array_equal(action, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]):
            recorded_individual.append(1)
        if np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]):
            recorded_individual.append(2)
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]):
            recorded_individual.append(0)
            recorded_individual.append(1)
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]):
            recorded_individual.append(0)
            recorded_individual.append(2)
        if np.array_equal(action, [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]):
            recorded_individual.append(3)
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]):
            recorded_individual.append(0)
            recorded_individual.append(3)
        if np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            recorded_individual.append(3)

        # Executando a ação escolhida no ambiente e obtendo as informações de retorno
        obs, rew, done, info = env.step(action)
        score = info.get('score', 0)  # Verifica se 'score' está presente em info, caso contrário retorna 0
        # Convertendo a imagem de observação para formato Pygame
        frame = obs[:, :, :]  # Convertendo de BGR para RGB
        frame = np.rot90(frame)  # Rotacionando a imagem para a orientação correta
        frame = pygame.surfarray.make_surface(frame)  # Convertendo para Surface do Pygame

        # Redimensionando e desenhando a imagem na janela do Pygame
        screen.blit(pygame.transform.flip(pygame.transform.scale(frame, (SCREEN_WIDTH, SCREEN_HEIGHT)), True, False), (0, 0))


        pygame.display.flip()

        # Limitando a taxa de atualização
        clock.tick(60)

        # Verificando se o episódio (jogo) terminou
        if done:
            break

    time.sleep(2)     
    # Fechando a janela principal do jogo
    pygame.display.quit()
    
    # Abrindo uma nova janela para exibir o score final
    final_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Final Score') 

    # Exibindo o score final
    font = pygame.font.Font(None, 80)
    final_score_text = font.render(f'Final Score: {score}', True, (255, 255, 0))
    text_centralize = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    final_screen.blit(final_score_text, text_centralize)
    pygame.display.flip() 

    # Salva as ações da sessão do jogo em um arquivo .npy
    np.save('recorded_gameplay.npy', recorded_individual)
    print("Fim de Jogo")
    time.sleep(5)

    # Fechando o ambiente
    obs = env.close()  
    # Finalizando o Pygame
    pygame.quit()

# Roda o melhor indivíduo do AG e do Neat
def playInd(individual):
    # Criando o ambiente do jogo Retro
    env = retro.make(game='Airstriker-Genesis')

    # Inicializando o Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Airstriker-Genesis')

    # Reiniciando o ambiente e obtendo a observação inicial
    obs = env.reset()

    clock = pygame.time.Clock()

    # Dicionário para mapear teclas do Pygame para ações do Retro
    key_to_action = {
        0: [1, 0, 0, 0, 0, 0, 0, 0],  # Ação para atirar
        1: [0, 0, 0, 0, 0, 0, 1, 0],  # Ação para mover à esquerda
        2: [0, 0, 0, 0, 0, 0, 0, 1],  # Ação para mover à direita  
        3: [0, 0, 0, 0, 0, 0, 0, 0]   # Ação para descansar
    }

    # Loop principal do jogo
    remaining_moves = 0
    shoot_toggle = True
    
    while True:
        # Usa o individuo para determinar as direções
        if remaining_moves < len(individual):
            action_index = int(individual[remaining_moves])
            if action_index == 0:
                # Garante que o indivíduo atire
                action = key_to_action[0] if shoot_toggle else key_to_action[3]
                shoot_toggle = not shoot_toggle
            else:
                action = key_to_action[action_index]
            remaining_moves += 1
        else:
            break
        
        # Executando a ação escolhida no ambiente e obtendo as informações de retorno
        obs, rew, done, info = env.step(action)

        # Acessando o score do jogo (se disponível)
        score = info.get('score', 0)  # Verifica se 'score' está presente em info, caso contrário retorna 0

        # Convertendo a imagem de observação para formato Pygame
        frame = obs[:, :, :]  # Convertendo de BGR para RGB
        frame = np.rot90(frame)  # Rotacionando a imagem para a orientação correta
        frame = pygame.surfarray.make_surface(frame)  # Convertendo para Surface do Pygame

        # Redimensionando e desenhando a imagem na janela do Pygame
        screen.blit(pygame.transform.flip(pygame.transform.scale(frame, (SCREEN_WIDTH, SCREEN_HEIGHT)), True, False), (0, 0))
        pygame.display.flip()

        # Limitando a taxa de atualização
        clock.tick(60)

        # Verificando se o episódio (jogo) terminou
        if done:
            break

    time.sleep(2)     
    # Fechando a janela principal do jogo
    pygame.display.quit()
    
    # Abrindo uma nova janela para exibir o score final
    final_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Final Score') 

    # Exibindo o score final
    font = pygame.font.Font(None, 80)
    final_score_text = font.render(f'Final Score: {score}', True, (255, 255, 0))
    text_centralize = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    final_screen.blit(final_score_text, text_centralize)
    pygame.display.flip() 

    print("Fim de Jogo")
    time.sleep(5)

    obs = env.close()  
    # Finalizando o Pygame
    pygame.quit()

# Roda o agente PPO
def playIndPPO(individual):
    # Criando o ambiente do jogo Retro
    env = retro.make(game='Airstriker-Genesis')

    # Inicializando o Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Airstriker-Genesis')

    # Reiniciando o ambiente e obtendo a observação inicial
    obs = env.reset()

    clock = pygame.time.Clock()

    # Loop principal do jogo
    remaining_moves = 0
    
    while True:
        # Usa o individuo para determinar as direções
        if remaining_moves < len(individual):
            action = individual[remaining_moves]
            remaining_moves += 1
        else:
            break
        
        # Executando a ação escolhida no ambiente e obtendo as informações de retorno
        obs, rew, done, info = env.step(action)

        # Acessando o score do jogo (se disponível)
        score = info.get('score', 0)  # Verifica se 'score' está presente em info, caso contrário retorna 0

        # Convertendo a imagem de observação para formato Pygame
        frame = obs[:, :, :]  # Convertendo de BGR para RGB
        frame = np.rot90(frame)  # Rotacionando a imagem para a orientação correta
        frame = pygame.surfarray.make_surface(frame)  # Convertendo para Surface do Pygame

        # Redimensionando e desenhando a imagem na janela do Pygame
        screen.blit(pygame.transform.flip(pygame.transform.scale(frame, (SCREEN_WIDTH, SCREEN_HEIGHT)), True, False), (0, 0))
        pygame.display.flip()

        # Limitando a taxa de atualização
        clock.tick(60)

        # Verificando se o episódio (jogo) terminou
        if done:
            break

    time.sleep(2)     
    # Fechando a janela principal do jogo
    pygame.display.quit()
    
    # Abrindo uma nova janela para exibir o score final
    final_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Final Score') 

    # Exibindo o score final
    font = pygame.font.Font(None, 80)
    final_score_text = font.render(f'Final Score: {score}', True, (255, 255, 0))
    text_centralize = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    final_screen.blit(final_score_text, text_centralize)
    pygame.display.flip() 

    print("Fim de Jogo")
    time.sleep(5)

    obs = env.close()  
    # Finalizando o Pygame
    pygame.quit()

if __name__ == "__main__":
    if controller == 0:
        logging.info('Initializing gameplay module.')
        gameplay()

    if controller == 1:
        game = GeneticAlgorithm()
        logging.info('Initializing Genetic agent.')
        # Executa o algoritmo genético
        bestInd = game.execute()
        # Roda o melhor indivíduo no playInd()
        playInd(bestInd)

    if controller == 2:
        game = NeatAlgorithm()
        logging.info('Initializing Neat agent.')
        # Executa o Neat
        bestInd = game.execute()
        # Roda o melhor indivíduo no playInd()
        playInd(bestInd)

    if controller == 3:   
        agent = PPOAgent()
        logging.info('Initializing PPO agent.')

        # Para iniciar e treinar um novo agente
        agent.train()

        # Para carregar um agente
        #agent.load_model("ppo_agents/ppo_airstriker_9h")

        # Para continuar o treinamento
        #agent.continue_training("ppo_agents/ppo_airstriker_9h", additional_timesteps=100000)

        # Pega as ações do agente treinado e roda em playIndPPO
        bestInd, score = agent.get_best_individual()
        playIndPPO(bestInd)


    if controller == 4:
        logging.info('Initializing Loaded agent.')
        # Roda o indivíduo salvo que foi carregado
        playInd(indLoader)

