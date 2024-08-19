import numpy as np
import tensorflow as tf
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout,  Flatten, Conv3D,  MaxPooling3D
from keras.optimizers import Adam
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
import random
import os
from global_vars import *
from concurrent.futures import ThreadPoolExecutor
from memory import AgentMemory

class PongAgent:
    def __init__(self, train, max_games):
        self.train = train
        self.memory = AgentMemory()
        self.epsilon = 0.55 if train else 0.0
        self.decay = MIN_EPSILON**(1/max_games)
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.stats = {
            'train_loss': [],
            'win_rate': []
        }
        if os.path.exists(BASE_PATH):
            self.load(BASE_PATH)
        self.step = 0
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.last_act = [0,0,0,0,1]

    def conv_block(self, model, filters, kernel_size, k_stride, pool_size, p_stride, dropout_rate=DROPOUT_RATE):
        model.add(Conv3D(filters, kernel_size, strides=k_stride, activation=ACTIVATION, padding='same'))
        model.add(MaxPooling3D(pool_size=pool_size, strides=p_stride))
        model.add(Dropout(dropout_rate))

    def dense_block(self, model, units):
        model.add(Dense(units, activation=ACTIVATION))
        model.add(Dropout(DROPOUT_RATE))

    def _build_model(self) -> Sequential:
        model = Sequential()
        kernels = 128
        
        # 3D Convolution block
        model.add(Conv3D(kernels, kernel_size=(17,9,3), strides=(7,3,1), activation=ACTIVATION, padding='same', input_shape=INPUT_SHAPE))
        model.add(MaxPooling3D(pool_size=(3,3,1), strides=(1,1,1)))
        model.add(Dropout(DROPOUT_RATE))
        
        kernels *= 2
        self.conv_block(model, kernels, (7,7,3), (2,2,1), (3,3,1), (2,2,1), DROPOUT_RATE)
        
        kernels //= 2
        self.conv_block(model, kernels, (3,3,3), (2,2,1), (3,3,1), (1,1,1), DROPOUT_RATE)
        
        # Flatten the 3D tensor
        model.add(Flatten())
        
        # Dense layers
        units = 2**11
        for _ in range(3):
            self.dense_block(model, int(units))
            units //= 4
        
        # Output layer
        model.add(Dense(ACTION_SIZE, activation=OUTPUT_ACTIV))
        
        model.compile(
            loss=LOSS_FUNC, 
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=[METRIC]
        )
        model.summary(100)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember((state, action, reward, next_state, done))#.append()

    def __call__(self, state):
        self.step_itter()
        act = [0,0,0,0,0]
        if random.random() < self.epsilon:
            if random.random() > self.epsilon / 2.5:
                act[random.randint(0, 4)] = 1
            else:
                act = self.last_act
        else:
            if self.step == 0:
                act[np.argmax(self._optimized_predict(state))] = 1
            else:
                act = self.last_act
        self.last_act = act
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
        
    def apply_decay(self):
        if self.train and self.epsilon > MIN_EPSILON:
            self.epsilon *= self.decay

    def step_itter(self):
        self.step = (self.step + 1) % MODEL_SAMPLE_RATE
        return self.step

    def reset(self):
        #self.memory.clear()
        self.step = 0
        self.last_act = [0,0,0,0,0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def update_target(self):
        self.target_model = clone_model(self.model)

