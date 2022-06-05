import time
ttm = time.time() # время начала выполнения, [с] 
import numpy as np
def SARSA(env, lr=0.01, num_episodes=10000, gamma=0.95, 
 eps=0.3, eps_decay=0.00005):
 """
 Функция, реализующая метод RL-обучения SARSA
 Параметры:
 env - объект класса окружения (среда "Taxi-v3")
 lr - скорость обучения
 num_episodes - максимальное число итераций (игр, эпизодов)
 gamma - коэффициент дисконтирования
 eps - нач.вероятн.выбора случ.действ. при eps-жадном алг.
 eps_decay - декремент величины eps
 Возвращает:
 обученную Q-матрицу полезности действия
 """
 nA = env.action_space.n # количество возможных действий
 nS = env.observation_space.n # количество возможных состояний
# Инициализация Q-матрицы [nS,nA]: строки - состояния; столбцы - действия
 Q = np.zeros((nS, nA))
 
 games_reward = [] # список наград в эпизодах
 test_rewards = [] # результаты тестирования
 for ep in range(num_episodes):
 
     state = env.reset() # сброс среды в нач. состояние
     tot_rew = 0 # награда в текущем эпизоде
     # Уменьшение eps в eps-жадном алг.
     if eps > 0.01:
         eps -= eps_decay
     action = eps_greedy(Q, state, eps) # выбор eps-жадного действия
 # Основной цикл SARSA-обучения
     done = False
     while not done:
 
         # Выполнение очередного шага игры 
         next_state, rew, done, _ = env.step(action) 
         # выбор следующего eps-жадного действия (для SARSA)
         next_action = eps_greedy(Q, next_state, eps)
         # SARSA-обновление
         Q[state][action] = Q[state][action] + lr*(rew + 
         gamma*Q[next_state][next_action] -
         Q[state][action])
         state = next_state
         action = next_action
         tot_rew += rew
         
         if done:
             games_reward.append(tot_rew)
 # Через каждые 300 эпизодов тестируем агента и печатаем результат
 if (ep % 300) == 0:
     test_rew = run_episodes(env, Q, 1000)
     print("Episode:{:5d} Eps:{:2.4f} Rew:{:2.4f}".
     format(ep, eps, test_rew))
     test_rewards.append(test_rew)
 return Q, games_reward, test_rewards
def eps_greedy(Q, s, eps=0.1):
 """
 Функция, выбирающая действие согласно eps-жадной стратегии 
 Параметры:
 Q - матрица полезности действия;
 s - текущее состояние;
 eps - вероятн.выбора случ.действ. при eps-жадном алг.
 Возвращает:
 eps-жадное действие
 """ 
 if np.random.uniform(0,1) < eps: # случ.число с равн.зак.распр.
     return np.random.randint(Q.shape[1])
 else:
     return greedy(Q, s) # выбор действия по жадной стратегии
def greedy(Q, s):
 """
 Функция, выбирающая действие согласно жадной стратегии 
 Параметры:
 Q - матрица полезности действия;
 s - текущее состояние;
 Возвращает:
 жадное действие
 """
 return np.argmax(Q[s])
def run_episodes(env, Q, num_episodes=100, to_print=False):
 """
 Функция, запускающая несколько эпизодов игры для оценки стратегии 
 Параметры:
 env - объект класса окружения (среда "Taxi-v3")
 Q - матрица полезности действия (определяющая стратегию);
 num_episodes - количество запускаемых эпизодов;
 to_print - флаг вывода нга печать
 Возвращает:
 среднее вознаграждение в серии эпизодов
 """
 tot_rew = [] # список наград в эпизодах
 state = env.reset() # сброс среды в нач. состояние
 for _ in range(num_episodes):
     done = False
     game_rew = 0 # награда в текущем эпизоде
     while not done:
         # Выполнение очередного шага игры при жадной стратегии
         next_state, rew, done, _ = env.step(greedy(Q, state))
         state = next_state
         game_rew += rew 
         if done:
             state = env.reset()
             tot_rew.append(game_rew)
 if to_print:
     print('Mean score: %.3f of %i games!'%(np.mean(tot_rew), num_episodes))
 return np.mean(tot_rew)
import gym
env = gym.make("Taxi-v3") # создаём окружение (огр.кол-ва 
nA = env.action_space.n # количество возможных действий
nS = env.observation_space.n # количество возможных состояний
print("\n Cостояний:", nS, "\n Действий: ", nA)
env.reset()
env.render() # изображение игрового поля
 # Вызово функции, реализующей метод RL-обучения SARSA
Q_sarsa, games_reward_sarsa, test_rewards_sarsa = SARSA(
 env, lr=.1, num_episodes=5000, eps=0.4, gamma=0.95, eps_decay=0.001)
env.close() # закрываем среду
import matplotlib.pyplot as plt
fig1 = plt.figure(figsize=[8, 4])
plt.plot(games_reward_sarsa, 'c')
plt.ylabel('sum(Rewards)')
plt.xlabel('Episodes')
plt.show()
fig2 = plt.figure(figsize=[8, 4])
plt.scatter(range(len(test_rewards_sarsa)), test_rewards_sarsa, color = 'r')
plt.ylabel('mean(Test_Rewards)')
plt.xlabel('Test_Episodes')
plt.show()
print ("\n Время выполнения: %6.2f c" % (time.time()-ttm))
