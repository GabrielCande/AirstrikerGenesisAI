import retro
import gym
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Configuração do logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PPOAgent:
    def __init__(self, game='Airstriker-Genesis', timesteps=100000):
        self.game = game
        self.timesteps = timesteps
        self.env = DummyVecEnv([lambda: retro.make(game=self.game)])
        self.model = PPO('CnnPolicy', self.env, verbose=1)

    # Função de treinamento do agente PPO 
    def train(self):
        logging.info(f'Starting training for {self.timesteps} timesteps.')
        # Inicia o treinamento
        self.model.learn(total_timesteps=self.timesteps)
        # Salva o agente no arquivo nomeado como ppo_airstriker.zip
        self.model.save("ppo_airstriker")
        logging.info('Training completed and model saved.')

    # Função que testa o agente treinado para 10 episódios
    def evaluate(self, num_episodes=10):
        logging.info(f'Evaluating agent over {num_episodes} episodes.')
        episode_rewards = []
        rewards_mean = 0
        for _ in range(num_episodes):
            obs = self.env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _states = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward[0]
            # Salva a recompensa obtida no episódio
            episode_rewards.append(total_reward)
            logging.debug(f'Episódio {_+1}: recompensa = {total_reward} timesteps = {self.model.num_timesteps}')
        # Retorna a média de recompensa por episódio do agente
        rewards_mean = np.mean(episode_rewards)
        logging.debug(f'Recompensa média por episódio = {rewards_mean}')

    # Função que carrega o agente salvo no arquivo ppo_airstriker.zip
    def load_model(self, path="ppo_airstriker"):
        self.model = PPO.load(path, env=self.env)
        logging.info(f'Model loaded from {path}.')

    # Função que treina novamente o agente que foi salvo
    def continue_training(self, path, additional_timesteps):
        self.load_model(path)
        logging.info(f'Continuing training for {additional_timesteps} timesteps.')
        # Treina novamente o agente com uma quantidade de timestep adicional
        self.model.learn(total_timesteps=additional_timesteps)
        # Salva o agente que foi treinado novamente
        self.model.save("ppo_airstriker_12h")
        logging.info('Continued training completed and model saved.')

    def get_best_individual(self):
        logging.info('Iniciando avaliação do melhor indivíduo.')
        obs = self.env.reset()
        done = False
        total_reward = 0
        actions = []
        while not done:
            action, _states = self.model.predict(obs)
            # Garantir que action seja um array de múltiplos valores
            action = np.array(action, dtype=int)[0]
            # Salva as ações do agente para rodar depois em playIndPPO()
            actions.append(action)
            obs, reward, done, info = self.env.step([action])
            total_reward += reward[0]
        self.env.close()
        logging.info(f'Recompensa total: {total_reward}')
        return actions, total_reward


