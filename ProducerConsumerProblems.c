#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define BUFFER_SIZE 5

int buffer[BUFFER_SIZE];
int indexIn = 0;
int indexOut = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t full = PTHREAD_COND_INITIALIZER;
pthread_cond_t empty = PTHREAD_COND_INITIALIZER;

void *producer(void *arg) {
    int item;
    for (int i = 0; i < BUFFER_SIZE * 2; i++) {
        item = rand() % 100; // Generate a random item to produce

        pthread_mutex_lock(&mutex);//lock the mutex
        if ((indexIn + 1) % BUFFER_SIZE == indexOut) {
            pthread_cond_wait(&empty, &mutex);//wait for the buffer to be empty
        }

        //produce the item
        buffer[indexIn] = item;
        indexIn = (indexIn + 1) % BUFFER_SIZE;
        printf("Produced: %d\n", item);
        pthread_cond_signal(&full);//signal that the buffer is full
        pthread_mutex_unlock(&mutex);//unlock the mutex
    }
    pthread_exit(0);
}

void *consumer(void *arg) {
    int item;
    for (int i = 0; i < BUFFER_SIZE * 2; i++) {
        pthread_mutex_lock(&mutex);//lock the mutex
        if (indexIn == indexOut) {
            pthread_cond_wait(&full, &mutex);//wait for the buffer to be full
        }

        item = buffer[indexOut];
        indexOut = (indexOut + 1) % BUFFER_SIZE;
        printf("Consumed: %d\n", item);
        pthread_cond_signal(&empty);//signal that the buffer is empty
        pthread_mutex_unlock(&mutex);//unlock the mutex
    }
    pthread_exit(0);
}

int main() {
    pthread_t producerThread, consumerThread;

    srand(time(NULL));

    pthread_create(&producerThread, NULL, producer, NULL);
    pthread_create(&consumerThread, NULL, consumer, NULL);

    pthread_join(producerThread, NULL);
    pthread_join(consumerThread, NULL);

    return 0;
}