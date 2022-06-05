import gym 
import numpy as np 
import time 
t = time.time() # время начала выполнения, [с] 
env = gym.make("Taxi-v3").env # создаём окружение 
state = env.reset() # инициализация среды 
env.render() # Визуализация среды 
n_states = env.observation_space.n 
n_actions = env.action_space.n 
print("\n Cостояний:", n_states, "\n Действий: ", n_actions) 
 # Таблица наград: {action: [(probability, nextstate, reward, done)]}
print('\n Таблица наград: \n', 
 env.P[state],'\n') 
 # [South,North,West,East,Pickup,Dropoff] 
possible_actions = [0, 1, 2, 3, 4, 5] 
 # СОЗДАЁМ "РАВНОМЕРНУЮ" СТРАТЕГИЮ с вероятностным распределением p(s,a) 
 # в виде 2-D массива (500x6) = (n_states x n_action) 
policy = np.array( 
 [[1./n_actions for _ in range(n_actions)] 
 for _ in range(n_states)]) 
 # ГЕНЕРИРУЕМ ИГРОВУЮ СЕССИЮ с такой стратегией 
def generate_session(policy, t_max=10**4): 
 states, actions = [],[] 
 total_reward = 0 
 s = env.reset() 
 for t in range(t_max): 
 
 # выбираем действие с вероятностью, указанной в стратегии 
 #~~~~~~~~ Управляющий код здесь ~~~~~~~~~~~
     a = np.random.choice(possible_actions, 1, p=policy[s])[0] 
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     new_s, r, done, info = env.step(a) 
     
     # запоминаем состояния, действия и вознаграждение 00
     states.append(s) 
     actions.append(a) 
     total_reward += r
     s = new_s
     if done:
         break 
 
 return states, actions, total_reward 
s, a, r = generate_session(policy) 
 # ОПРЕДЕЛЯЕМ ЛУЧШИЕ ПАРЫ СОСТОЯНИЕ-ДЕЙСТВИЕ (с max вознаграждением)
 # среди серии игровых сессий 
 # (X-я процентиль -> порог, ниже кот. распол. X% значений выборки)
def select_elites(states_batch, actions_batch, 
 rewards_batch, percentile=50): 
 # Порог вознеграждения для попадания в элиту 
 reward_threshold = np.percentile(rewards_batch, percentile) 
 
 # Отбор лучших пар состояние-действие 
 elite_states, elite_actions = [], [] 
 for i in range(len(rewards_batch)): 
     if rewards_batch[i] >= reward_threshold: 
         elite_states = elite_states + states_batch[i] 
         elite_actions = elite_actions + actions_batch[i] 
 
 return elite_states, elite_actions 
 # ОБНОВЛЯЮЩАЯСЯ СТРАТЕГИЯ - частота вхождений пар состояние-действие 
 # в лучшие (элитные) игровые сессии
def update_policy(elite_states,elite_actions): 
 
 new_policy = np.ones([n_states,n_actions])/n_actions 
 for state in range(n_states): # перебор всех состояний 
     nes = elite_states.count(state) # кол-во появл. данного 
     # сост.в элит.сессиях 
     if nes: 
         k=0 
         for _ in range(nes): 
             k = elite_states.index(state,k) 
             a = elite_actions[k] # элит.дейст.в эт.сост. 
             new_policy[state, a] += 1 # улучш. стратегии 
             k += 1 
 
 # Нормирование вероятностей действий для данного состояния 
         new_policy[state] = new_policy[state] / sum(new_policy[state]) 
 
 return new_policy 
 # МОДЕЛИРОВАНИЕ ОБУЧЕНИЯ 
""" (При n_series = 10; n_sessions = 60; t_max=300 
 время выполнения около 24 c) 
""" 
n_series = 30 # количество серий (эпох) 
n_sessions = 250 # количество сессий в серии (для сэмплир.) 
t_max=300 # максимальное кол-во шагов в сессии 
percentile = 50 # процентиль для выбора элитных сессий 
learning_rate = 0.5 # скорость обучения 
log = [] 
for i in range(n_series): # i-я серия сессий 
 
 # Генерируем некоторое каличество сессий: 
 states_batch = [] # последоват. сост. в сессиях серии 
 actions_batch = [] # посл. действий в сессиях серии 
 rewards_batch = [] # подкрепления в сессиях серии 
 len_sess = [] # кол-во шагов в сессиях 
 for i in range(n_sessions): 
     s, a, r = generate_session(policy, t_max) 
     states_batch.append(s) 
     actions_batch.append(a) 
     rewards_batch.append(r) 
     len_sess.append(len(s)) 
 # Отбираем лучшие действия и состояния 
 elite_states, elite_actions = select_elites(states_batch, 
 actions_batch, rewards_batch, percentile) 
 # Обновляем стратегию 
 new_policy = update_policy(elite_states,elite_actions) 
 policy = learning_rate*new_policy + (1-learning_rate)*policy 
 # Протокол обучения 
 # средняя накрада в серии сессий 
 mean_reward = np.mean(rewards_batch) 
 # порог отбора элитных сессий 
 threshold = np.percentile(rewards_batch, percentile) 
 # средняя продолжительность серии сессий 
 mean_len_sess = np.mean(len_sess) 
 
 n_fail = len_sess.count(t_max)/t_max 
 
 log.append([mean_reward, threshold, mean_len_sess, n_fail]) 
 print("mean reward = %.3f, threshold=%.3f,mean len_sess = %.3f (n_fail=%.3f)" % (mean_reward, threshold, mean_len_sess, n_fail)) 
 # ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ МОДЕЛИРОВАНИЯ 
 
import matplotlib.pyplot as plt 
""" Функция zip() на каждой итерации возвращает кортеж, 
 содержащий элементы последовательностей, которые расположены 
 на одинаковом смещении. Функция возвращает объект, поддерживающий 
 итерации. Чтобы превратить его в список, следует результат 
 передать в функцию list(). 
 Если есть список кортежей и необходимо разделить элементы каждого 
 кортежа на независимые последовательности можно использовать zip() 
 вместе с оператором распаковки *. 
""" 
mrw, trh, mls, nfl = list(zip(*log)) 
fig1 = plt.figure(figsize=[12, 4]) 
plt.subplot(1, 3, 1) 
plt.plot(range(n_series), mrw, label='Mean rewards') 
plt.plot(range(n_series), trh, label='Reward thresholds') 
plt.legend(loc=4) 
plt.grid() 
plt.subplot(1, 3, 2) 
plt.plot(range(n_series), mls, label='mean_len_sess') 
plt.legend(loc=3) 
plt.grid() 
plt.subplot(1, 3, 3) 
plt.plot(range(n_series), nfl, label='n_fail') 
plt.legend(loc=3) 
plt.grid() 
plt.show() 
print ("Время выполнения: %6.2f c" % (time.time()-t)) 
env.close() # закрываем среду
