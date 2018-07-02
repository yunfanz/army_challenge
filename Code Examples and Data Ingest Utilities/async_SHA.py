from multiprocessing import Process, Queue, Event
import multiprocessing as mp
import numpy as np
from time import sleep
from inception import *

class TrainWorker(Process):
    def __init__(self, data, task_q: Queue, result_q, stop_event: Event, name):
        super().__init__()
        self.task_q = task_q
        self.result_q = result_q
        self.stop_event = stop_event
        self.name = name
        self.x_train, self.y_train, self.x_val, self.y_val = data
        self.train_size, self.val_size = self.x_train.shape[0], self.x_val.shape[0]

    def train_model(self, theta, k, resource, num_classes=5):
        import keras
        from keras.datasets import mnist
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from keras import backend as K
        import tensorflow as tf
        import numpy as np
        import os
        from inception import googleNet, get_pdict
        
        
        
        
        if resource < 1:
            train_size, val_size = int(resource*self.train_size), int(resource*self.val_size)
            train_idx = np.random.choice(range(self.train_size), train_size)
            val_idx = np.random.choice(range(self.val_size), val_size)
            x_train, y_train = self.x_train[train_idx], self.y_train[train_idx]
            x_val, y_val = self.x_val[val_idx], self.y_val[val_idx]
        else:
            x_train,y_train,x_val, y_val = self.x_train, self.y_train, self.x_val, self.y_val
        epochs = max(1, int(resource))
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.name)
        if self.name == 2:
            batch_size = 128
        else:
            batch_size = 256
        print("starting on", self.name, 'rung', k, 'epochs', epochs, 'train_size', x_train.shape[0])
        
        input_img = Input(shape=(1024,2))
        out = googleNet(input_img,data_format='channels_last', pdict=theta, num_classes=num_classes)
        model = Model(inputs=input_img, outputs=out)
        #model.summary()
        print(theta)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        #print('model compiled')
        for e in range(epochs):
            x_train = augment(x_train)
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=0,
                      validation_data=(x_val, y_val))
        score = model.evaluate(self.x_val, self.y_val, verbose=0)
        return score

    def run(self):
        while not self.stop_event.is_set():
            if self.task_q.empty():
                sleep(0.1)
            else:
                #print("Received task", self.name)
                idx, theta, k, resource = self.task_q.get()
                score = self.train_model(theta, k, resource)
                print('putting result', (idx, k))
                self.result_q.put((idx, theta, k, score[0]))
                
class JobManager(Process):
    def __init__(self, train_worker, task_q: Queue, result_q: Queue, ladder, stop_event: Event):
        super().__init__()
        self.train_worker = train_worker
        self.result_q = result_q
        self.task_q = task_q
        self.ladder = ladder
        self.stop_event = stop_event
        self.eta = 3
        self.KMAX = 6
        self.resource_min = 1
        self.bracket = 0
        self.idx = 0
        while not self.task_q.full():
            if self.idx == 0:
                theta = {'depths': np.array([2, 2, 1, 0, 2, 1]), 'features': np.array([3, 2, 3, 3, 3, 3, 3]), 'dr': 0.67561133072930946}
            elif self.idx== 1:
                theta = {'depths': np.array([2, 2, 0, 3, 0, 2]), 'features': np.array([1, 3, 1, 2, 2,  2, 2]), 'dr': 0.24749480935162974}
            elif self.idx == 2:
                theta = {'depths': np.array([1, 1, 0, 0, 0, 1]), 'features': np.array([3, 1, 2, 2, 3, 3, 3]), 'dr': 0.54753203788931493}
            else:
                theta = get_pdict(mode='orig')
            
            self.task_q.put((self.idx, theta, 0,self.resource_min*self.eta**(0+self.bracket)))
            self.update_ladder(k=0)
            self.idx += 1
            sleep(1)
            
    def update_ladder(self, k, idx=None, value=None):
        if idx is None:
            idx = self.idx
        if len(self.ladder) <= k:
            self.ladder.append({})
            print("### New Rung Reached  ###")
            for k, rung_dict in enumerate(self.ladder):
                print(list(rung_dict.keys()))
                if k == len(self.ladder)-2:
                    print(rung_dict)
        self.ladder[k][idx] = value

    def get_job(self):
        kmax = len(self.ladder)-1
        job = None
        if kmax> self.KMAX:
            self.stop_event.set()
            print("stopping event")
        for k in range(kmax+1)[::-1]:
            rung = self.ladder[k]  #dictionary{idx:(theta, loss), idx2:None}
            rung = [(key, val) for key, val in rung.items() if val is not None]
            if len(rung) < self.eta: continue
            rung = sorted(rung, key=lambda x:x[1][1])
            promotable = rung[:int(float(len(rung))/self.eta)]
            if k == kmax:
                best = promotable[0]
            else:
                best = None
                for p in promotable:
                    if p[0] not in list(self.ladder[k+1].keys()):
                        best = p
                        break
            if best is not None:
                job = (best[0], best[1][0], k+1)
                break
        
        if job is None:
            #print("Nothing promotable", self.ladder[-1])
            job = (self.idx, get_pdict(mode='prior'), 0)
            self.idx += 1
        print("submit job", job)
        
        return job
    
    def run(self):
        while not self.stop_event.is_set():
            if self.result_q.empty():
                sleep(1)
            else:
                idx, theta, k, loss = self.result_q.get()
                print('Received result', idx, k, loss)
                self.update_ladder(k, idx=idx, value=(theta, loss))
                idx, theta, k = self.get_job()
                self.update_ladder(k, idx=idx)
                resource = self.resource_min*self.eta**(k+self.bracket)
                self.task_q.put((idx, theta, k, resource))
                

class async_SHA:
    def __init__(self, data, ngpu=2):
        self.stop_event = Event()
        self.result_q = Queue(4)
        self.ladder = []
        self.ngpu = ngpu
        self.task_q = Queue(ngpu)
        
    def run(self):
        self.train_worker = {pid: TrainWorker(data, self.task_q, self.result_q, self.stop_event, name=str(pid)) for pid in range(self.ngpu)}
        for w in self.train_worker.values():
            w.start()
        self.job_manager = JobManager(self.train_worker, self.task_q, self.result_q, self.ladder, self.stop_event)
        self.job_manager.start()
        if self.stop_event.is_set():
            print(self.ladder)
            for w in self.train_worker.values():
                w.join()
                #w.terminate()
            self.job_manager.join()
            #self.job_manager.terminate()
            
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Asynchronous hyperoptim')
    parser.add_argument('--ngpu', type=int, default=2,
                    help='number of available gpus')
    parser.add_argument('--data_dir', dest='data_dir', type=str)
    args = parser.parse_args()
    x_train, y_train, x_val, y_val = get_data(mode='time_series',
                                         BASEDIR=args.data_dir,
                                         load_mods=['CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB', 'GFSK_5KHz'],
                                         files=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    data = (x_train, y_train, x_val, y_val)
    a = async_SHA(data, ngpu=args.ngpu)
    a.run()
