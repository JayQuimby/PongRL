import numpy as np
import tensorflow as tf
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, Input
from keras.optimizers import AdamW
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    from keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
import random
import os
from static import *
from concurrent.futures import ThreadPoolExecutor
from memory import AgentMemory

class PongAgent:
    def __init__(self, train, max_games):
        self.train = train
        self.memory = AgentMemory()
        self.epsilon = BASE_EPSILON if train else 0.0
        if train:
            self.decay = MIN_EPSILON**(self.epsilon/max_games)
            self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.stats = {
            'train_loss': [],
            'win_rate': []
        }
        
        if os.path.exists(LOAD_PATH):
            self.load(LOAD_PATH)
        self.step = 0
        

    def _conv_block(self, model, filters, kernel_size, k_stride, pool_size, p_stride, dropout_rate=DROPOUT_RATE):
        model.add(Conv3D(filters, kernel_size, strides=k_stride, activation=ACTIVATION, padding='same'))
        model.add(MaxPooling3D(pool_size=pool_size, strides=p_stride))
        model.add(Dropout(dropout_rate))

    def _dense_block(self, model, units):
        model.add(Dense(units, activation=ACTIVATION))
        model.add(Dropout(DROPOUT_RATE))

    def _build_model(self) -> Sequential:
        model = Sequential()
        model.add(Input(shape=INPUT_SHAPE))
        
        # More efficient progression
        model.add(Conv3D(32, (17,9,3), strides=(7,3,1), activation=ACTIVATION, padding='same'))
        model.add(MaxPooling3D((3,3,1), strides=(2,2,1)))
        model.add(Dropout(DROPOUT_RATE))
        
        model.add(Conv3D(64, (7,7,3), strides=(2,2,1), activation=ACTIVATION, padding='same'))
        model.add(MaxPooling3D((3,3,1), strides=(2,2,1)))
        model.add(Dropout(DROPOUT_RATE))
        
        model.add(Conv3D(64, (3,3,3), strides=(2,2,1), activation=ACTIVATION, padding='same'))
        model.add(MaxPooling3D((2,2,1)))
        model.add(Dropout(DROPOUT_RATE))
        
        model.add(Flatten())
        
        # Simpler dense layers (RL typically doesn't need huge dense layers)
        model.add(Dense(512, activation=ACTIVATION))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(256, activation=ACTIVATION))
        model.add(Dropout(DROPOUT_RATE))
        
        model.add(Dense(ACTION_SIZE, activation=OUTPUT_ACTIV))
        
        optim = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model.compile(loss=LOSS_FUNC, optimizer=optim, metrics=[METRIC])
        
        return model

    def __call__(self, state):
        self.step_itter()
        act = [0,0,0,0,0]
        if random.random() < self.epsilon:
            act[random.randint(0, 4)] = 1
        else:
            act[np.argmax(self._optimized_predict(state))] = 1
        return act

    @tf.function
    def _optimized_predict(self, state):
        return self.model(state, training=False)[0]

    def _process_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.vstack(states)
        next_states = np.vstack(next_states)

        # Get Q-values for current states and next_states
        target = self.model.predict(states, verbose=0) # Nx5
        q_val = self.target_model.predict(next_states, verbose=0) # Nx5
        
        # This is the Q-Value update step where we are approximating the current reward and future rewards
        # Rewards is given by the state of the board at next_state
        # The discount factor increases as time goes on so the values learned over time get more important
        # The Q-Value gets compounded more into the future as epsilon decreases
        # GG - if the next_state was a game over there is no q_val for the future
        #       |  R  | + |Dis| * |       Q Val       | * |         GG        |
        q_val = rewards + GAMMA * np.max(q_val, axis=1) * (1 - np.array(dones))
        
        # Update Q-values for the actions taken
        for t, q, a in zip(target, q_val, actions):
            i = np.argmax(a)
            # weight the update value based on current and next state prediction
            t[i] *= (1-Q_VAL_RATIO)
            t[i] += Q_VAL_RATIO * q

        return states, target

    def replay(self, percent):
        num_memories = int(len(self.memory) * percent)
        if num_memories < BATCH_SIZE:
            return

        num_batches = num_memories // BATCH_SIZE  # Number of batches
        num_sub_batches = 4  # Adjust based on your needs
        sub_batch_size = BATCH_SIZE // num_sub_batches

        #print(f'Training on {num_memories} memories.')
        losses = []
        for _ in range(num_batches):
            minibatch = self.memory(percent) 

            futures = []
            for i in range(0, BATCH_SIZE, sub_batch_size):
                sub_batch = minibatch[i:i+sub_batch_size]
                futures.append(self.thread_pool.submit(self._process_batch, sub_batch))

            all_states = []
            all_targets = []
            for future in futures:
                states, targets = future.result()
                all_states.append(states)
                all_targets.append(targets)

            all_states = np.vstack(all_states)
            all_targets = np.vstack(all_targets)

            loss = self.model.fit(all_states, all_targets, epochs=1, verbose=0, batch_size=BATCH_SIZE)
            losses.extend(loss.history[METRIC])
        self.stats['train_loss'].append(sum(losses)/len(losses))

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember((state, action, reward, next_state, done))

    def apply_decay(self):
        if self.train and self.epsilon > MIN_EPSILON:
            self.epsilon *= self.decay

    def step_itter(self):
        self.step = (self.step + 1) % MODEL_SAMPLE_RATE
        return self.step

    def reset(self):
        #self.memory.clear()
        self.step = 0

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

