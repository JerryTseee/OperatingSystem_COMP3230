#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>

int main(){
    int count = 0;
    pid_t pid1;
    pid_t pid2;

    pid1 = fork();
    if(pid1 == 0){
        //inside the child 1 process
        pid2 = fork();
        count++;
        if (pid2 == 0)
        {
           //inside the child 2 process
           count+=10;
           exit(0);
        }
        else{
            //inside the child 1 process again
            count += 5;
        }
        
    }
    else{
        //inside the parent process
        count++;
    }
    printf("hello world!\n");
    printf("count = %d\n", count);
}