import threading
import time
import random

num_philosophers = 5
num_forks = 5

#semephores for forks(1 indicate the lock is available)
forks = [threading.Semaphore(1) for i in range(num_forks)]
#semephore for mutex(1 indicate the lock is available) -> ensure mutual exclusion when acquiring forks
mutex = threading.Semaphore(1)

def philosopher(index):
    while True:
        print("philosopher ",index,"is thinking")
        time.sleep(random.randint(1,5))#sleep for random time 1-5 seconds
        
        mutex.acquire()#acquire mutex

        left_fork = index
        right_fork = (index+1) % num_forks
        forks[left_fork].acquire()#acquire left fork
        forks[right_fork].acquire()#acquire right fork

        mutex.release()#release mutex after getting the forks
        print("philosopher ",index,"is eating")
        time.sleep(random.randint(1,5))#sleep for random time 1-5 seconds

        forks[left_fork].release()#release left fork
        forks[right_fork].release()#release right fork

philosopher_threads = []
for i in range(num_philosophers):#for each philosopher create a thread
    philosopher_threads.append(threading.Thread(target=philosopher, args=(i,)))

for thread in philosopher_threads:
    thread.start()#start threads

for thread in philosopher_threads:
    thread.join()#wait for threads to finish