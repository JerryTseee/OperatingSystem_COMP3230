//the problem is about managing access to shared data
/*
Readers: muiltiple readers can read data simultaneously
Writers: only one writer can access the data
*/

//using semaphore!!!
#include<semaphore.h>
#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>

sem_t readMutex;
sem_t semMutex;
int readCount = 0;

pthread_t reader_thread[100];
pthread_t writer_thread[100];

//for reader
void startRead(){
    sem_wait(&readMutex);//lock the readMutex
    readCount++;
    if (readCount == 1)
    {
        sem_wait(&semMutex);//wait until the writer finishes
    }
    sem_post(&readMutex);//release the lock
}

void endRead(){
    sem_wait(&readMutex);//lock the readMutex
    readCount--;
    if (readCount == 0)
    {
        sem_post(&semMutex);//release the mutex lock so that writer can access
    }
    sem_post(&readMutex);//release the lock
}

void *reader(void *arg){
    startRead();
    //reading
    printf("reader is reading\n");
    endRead();
    pthread_exit(NULL);
}


//for writer
void startWrite(){
    sem_wait(&semMutex);//lock the semMutex
}

void endWrite(){
    sem_post(&semMutex);//release the lock
}

void *writer(void *arg){
    startWrite();
    printf("writer is writing\n");
    endWrite();
    pthread_exit(NULL);
}


int main(){

    //initialize the values to 1
    sem_init(&readMutex, 0, 1);
    sem_init(&semMutex, 0, 1);

    //two readers, one writer
    pthread_create(&reader_thread[0], NULL, reader, NULL);
    pthread_create(&reader_thread[1], NULL, reader, NULL);
    pthread_create(&writer_thread[0], NULL, writer, NULL);


    pthread_join(reader_thread[0], NULL);
    pthread_join(reader_thread[1], NULL);
    pthread_join(writer_thread[0], NULL);

    return 0;
}