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
        self.x_train, self.y_train, self.x_test, self.y_test = data

    def train_model(self, theta, k, resource):
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
        
        
        batch_size = 128
        num_classes = 24
        epochs = int(resource)
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.name)
        print("starting on", self.name, 'rung', k, 'epochs', epochs)
        
        input_img = Input(shape=(1024,2))
        out = googleNet(input_img,data_format='channels_last', pdict=theta)
        model = Model(inputs=input_img, outputs=out)
        #model.summary()
        print(theta)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        #print('model compiled')
        model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(self.x_test, self.y_test))
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        return score

    def run(self):
        while not self.stop_event.is_set():
            if self.task_q.empty():
                sleep(0.1)
            else:
                #print("Received task", self.name)
                idx, theta, k, resource = self.task_q.get()
                score = self.train_model(theta, k, resource)
                print('putting result', (idx, theta, k))
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
        self.KMAX = 4
        self.resource_min = 1
        self.bracket = 0
        self.idx = 0
        while not self.task_q.full():
            theta = get_pdict(mode='orig')
            theta['dr'] = np.random.uniform(0.3, 0.7, size=1)[0]
            self.task_q.put((self.idx, theta, 0,self.resource_min*self.eta**(0+self.bracket)))
            self.update_ladder(k=0)
            self.idx += 1
            
    def update_ladder(self, k, idx=None, value=None):
        if idx is None:
            idx = self.idx
        if len(self.ladder) <= k:
            self.ladder.append({})
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
#                 print("### Updating Ladder ###")
#                 for k, rung_dict in enumerate(self.ladder):
#                     print("Rung#", k, rung_dict)
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
    x_train, y_train, x_val, y_val = get_data(mode='time_series',
                                         BASEDIR="/home/mshefa/training_data/",
                                         files=[0])
    data = (x_train, y_train, x_val, y_val)
    a = async_SHA(data, ngpu=2)
    a.run()
