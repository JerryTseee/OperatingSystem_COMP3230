#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>

int N = 8;// the number of chair
int customer_num = 0;//it represents customer number
pthread_cond_t stove; // the state of the stove
pthread_cond_t burger; // the state of the burger
pthread_mutex_t mutex; // the mutex lock

void prepare_burger(){
    printf("chef is preparing burger\n");
}

void place_burger(){
    printf("chef is placing burger\n");
}

void get_burger(){
    printf("customer is getting burger\n");
}

void go_away(){
    printf("customer is going away\n");
}

void *customer(void *arg) {
    pthread_mutex_lock(&mutex);
    if (customer_num < N) {
        customer_num += 1;
        printf("customer %d comes and sits\n", customer_num);
        pthread_cond_signal(&stove); // signal chef that customer is coming
        pthread_cond_wait(&burger, &mutex); // wait for burger, and release mutex lock
        get_burger(); // get the burger
    } else {
        go_away(); // customer goes away
    }
    pthread_mutex_unlock(&mutex); // release mutex lock
}


void *chef(void *arg){
    //because chef must always be ready to serve customer, it is an infinite loop
    while (1){
        pthread_mutex_lock(&mutex);
        if(customer_num == 0){
            //wait for customer to come
            pthread_cond_wait(&stove, &mutex);
        }
        prepare_burger();
        place_burger();
        pthread_cond_signal(&burger);//signal customer that one burger is ready
        customer_num -= 1;
        pthread_mutex_unlock(&mutex);//release mutex lock
    }
}

int main(){
    pthread_t chef_thread;
    pthread_t customer_thread[100];

    //initialize mutex lock and condition variables
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&stove, NULL);
    pthread_cond_init(&burger, NULL);

    pthread_create(&chef_thread, NULL, chef, NULL);
    for(int i = 0; i < 20; i++){
        //asume that there are total 20 customers!
        pthread_create(&customer_thread[i], NULL, customer, NULL);
    }

    pthread_join(chef_thread, NULL);
    for(int i = 0; i < 20; i++){
        pthread_join(customer_thread[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&stove);
    pthread_cond_destroy(&burger);

    return 0;
}