#include <pthread.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <unistd.h> 

pthread_t tid[2]; 
pthread_mutex_t lock;
int counter; 

//a thread function
void* trythis(void* arg) 
{ 
	pthread_mutex_lock(&lock);//lock the mutex

	unsigned long i = 0; 
	counter += 1; // this is the critical section!

	printf("\n Job %d has started\n", counter); 
	for (i = 0; i < (0xFFFFFFFF); i++) 
		; 
	printf("\n Job %d has finished\n", counter); 

	pthread_mutex_unlock(&lock);//unlock the mutex


	return NULL; 
} 

int main(void) 
{ 
	int i = 0; 
	int error; 

	//because it has lock, so it will first finish job 1, then job 2
	while (i < 2) { 
		error = pthread_create(&(tid[i]), NULL, &trythis, NULL); //creating thread
		if (error != 0) 
			printf("\nThread can't be created : [%s]", strerror(error)); 
		i++; 
	} 

	pthread_join(tid[0], NULL); //join the first thread
	pthread_join(tid[1], NULL); //join the second thread
	pthread_mutex_destroy(&lock); //destroy the mutex
	return 0; 
} 
