/*
* PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: parallel_3036065642.c
* NAME: TSE Wang Pok
* UID:  3036065642
* Development Platform: Workbench2
* Remark: (How much you implemented?) all
* How to compile separately: (gcc -o parallel parallel_[UID].c -O2 -lm -lpthread): make -B
*/

#include "common.h" // some common definitions

#include <unistd.h>       // for nearly everything :)
#include <stdio.h>        // for printf, sprintf, fgets
#include <stdlib.h>       // for malloc, calloc
#include <stdint.h>       // for uint8_t and uint64_t
#include <time.h>         // for time
#include <string.h>       // for memcpy and strcmp
#include <sys/resource.h> // for rusage collection

#include "model.h"// for Llama definitions -> no need to know

int pos = 0; // global position of generation
Transformer transformer; // transformer instance to be init
Tokenizer tokenizer;     // tokenizer instance to be init
Sampler sampler;         // sampler instance to be init

// YOUR CODE STARTS HERE
#include <pthread.h>
// #include <semaphore.h> // uncomment this line if you use semaphore
#include <stdbool.h>   // uncomment this line if you want true / false

// you may define global variables here
pthread_cond_t cond;
pthread_cond_t main_thread_cond;
pthread_mutex_t lock;
pthread_t tids[20];

int which_function = 0;
int termination_flag;
int running_thread;
int thr_count;

//for collecting resources
struct thread_stats{
    long user_time;
    long user_time_s;
    long system_time;
    long system_time_s;
};
struct thread_stats stat[20];
struct rusage main_usage;

typedef struct{
    float* out;
    QuantizedTensor* vec;
    QuantizedTensor* mat;
    int col;
    int row;

    float* q;
    float* key_cache;
    float* value_cache;
    float* att;
    int seq_len;
    int n_heads;
    int head_size;
    int kv_dim;
    int kv_mul;
}parameters;

parameters para;

// function executed by each thread to complete mat_vec_mul
// @note: please modify the signature to what you want
void mat_vec_mul_task_func(int id, float* out, QuantizedTensor* vec, QuantizedTensor* mat, int col, int row) {

	int rows_per_thread = (row + thr_count - 1) / thr_count;//rows maybe odd or even
    int start_row = id * rows_per_thread;
    //if it is the last id or if it is not the last id
    int end_row = (id == thr_count - 1)? row : id * rows_per_thread + rows_per_thread;

    for(int i = start_row; i < end_row; i++){
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * col;

        for(int j = 0; j <= col - GS; j += GS){
            for (int k = 0; k < GS; k++)
            {
                ival += ((int32_t) vec->q[j+k]) * ((int32_t) mat->q[in+j+k]);
            }
            val += ((float) ival)*mat->s[(in+j)/GS] * vec->s[j/GS];
            ival = 0;
        }
        out[i] = val;
    }
}

// function executed by each thread to complete multi_head_attn
// @note: please modify the signature to what you want
void multi_head_attn_task_func(int id, float* out, float* q, float* key_cache, float* value_cache, float* att, int seq_len, int n_heads, int head_size, int kv_dim, int kv_mul) {

    int heads_per_thread = n_heads / thr_count;//seems like it is even heads
    int start_head = id * heads_per_thread;
    //if it is the last id or if it is not the last id
    int end_head = (id == thr_count - 1)? n_heads : id * heads_per_thread + heads_per_thread;
    
    for(int h = start_head; h < end_head; h++){
        float* head_q = q + h * head_size; // query vector for this head
        float* head_att = att + h * seq_len; // attention scores for this head
        for (int t = 0; t <= pos; t++) { // iterate over all timesteps
            // get the key vector for this head and at this timestep
            float* head_k = key_cache + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++){
                score += head_q[i] * head_k[i]; // attention score := q dot k
            }
            score /= sqrtf(head_size); // normalize by head_size
            head_att[t] = score; // save to the attention buffer
        }
        softmax(head_att, pos + 1); // THREADS-SAFE SoftMax to normalize scores to weight
        float* head_out = out + h * head_size; // out vector for this head
        memset(head_out, 0, head_size * sizeof(float)); // clear buffer
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* head_v = value_cache + t * kv_dim + (h / kv_mul) * head_size;
            float a = head_att[t]; // attention weight for this timestep
            for (int i = 0; i < head_size; i++){
                head_out[i] += a * head_v[i]; // accumulate the weighted sum to head out
            }
        }
    }
}


// thread function used in pthread_create
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void *thr_func(void *arg) {
    int threadID = *((int *)arg);
    struct rusage usage;

    while(true){
        pthread_mutex_lock(&lock);
        running_thread--;
        //printf("running thread: %d\n",running_thread);
        if (running_thread == 0)
        {
            //signal to the main thread that all the threads go to sleep
            pthread_cond_signal(&main_thread_cond);
        }
        pthread_cond_wait(&cond,&lock);//go to sleep and wait for being wake up!
        pthread_mutex_unlock(&lock);

        //we dont need lock to lock the below execution!
        if(termination_flag == 1){
            running_thread--;

            //collect the resource usage
            getrusage(RUSAGE_THREAD, &usage);
            stat[threadID].user_time = usage.ru_utime.tv_sec;
            stat[threadID].user_time_s = usage.ru_utime.tv_usec;
            stat[threadID].system_time = usage.ru_stime.tv_sec;
            stat[threadID].system_time_s = usage.ru_stime.tv_usec;
            // printf("id: %d\n", threadID);
            // pthread_exit(NULL);//terminate the current thread, and collect the system usage
            break;
        }

        if(which_function == 1){
            //if flag = 1, then execute this head function
            multi_head_attn_task_func(threadID, para.out, para.q, para.key_cache, para.value_cache, para.att, para.seq_len, para.n_heads, para.head_size, para.kv_dim, para.kv_mul);
        }
        else if(which_function == 0){
            //if flag = 1, then execute this head function
            mat_vec_mul_task_func(threadID, para.out, para.vec, para.mat, para.col, para.row);
        }
        
    }
}

// function to initialize thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void init_thr_pool(int num_thr) {
    thr_count = num_thr;
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_cond_init(&main_thread_cond, NULL);

    //create the threads
    for(int i = 0; i < num_thr; i++){
        //pthread_t tid;
        int* threadID = malloc(sizeof(int));
        *threadID = i;
        pthread_create(&tids[i], NULL, thr_func, threadID);//i is the thread id
        running_thread++;
    }

    //wait until all the threads have been put to sleep
    pthread_mutex_lock(&lock);
    termination_flag = 0;
    pthread_cond_wait(&main_thread_cond, &lock);
    pthread_mutex_unlock(&lock);
}

// function to close thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void close_thr_pool() {

    double total_user_time = 0;
    double total_system_time = 0;

    pthread_mutex_lock(&lock);
    //wake up all sleeping threads, and then let them die!
    termination_flag = 1;
    running_thread = thr_count;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);
    for (int i = 0; i < thr_count; i++)
    {
        pthread_join(tids[i], NULL);
    }

    //print out the resources of each thread
    for (int i = 0; i < thr_count; i++)
    {
        printf("Thread %d has completed - user: %ld.%06ld s, system: %ld.%06ld s\n",i ,stat[i].user_time , stat[i].user_time_s, stat[i].system_time, stat[i].system_time_s);
        total_user_time += stat[i].user_time + stat[i].user_time_s/1000000.0;
        total_system_time += stat[i].system_time + stat[i].system_time_s/1000000.0;
    }
    //also for the main thread here
    getrusage(RUSAGE_THREAD, &main_usage);
    printf("main thread - user: %ld.%06ld s, system: %ld.%06ld s\n", main_usage.ru_utime.tv_sec, main_usage.ru_utime.tv_usec, main_usage.ru_stime.tv_sec, main_usage.ru_stime.tv_usec);

    total_user_time += main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec / 1000000.0;
    total_system_time += main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec / 1000000.0;
    printf("Whole process - user: %.4f s, system: %.4f s\n", total_user_time, total_system_time);


    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&cond);
    pthread_cond_destroy(&main_thread_cond);
}

// ----------------------------------------------------------------------------
// entry function for multi-threading matrix multiplication - > main thread
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void mat_vec_mul(float* out, QuantizedTensor *vec, QuantizedTensor *mat, int col, int row) {
    //pass the values to the variables
    para.out = out;
    para.vec = vec;
    para.mat = mat;
    para.col = col;
    para.row = row;

    //wake up all threads and execute mat_vec function
    pthread_mutex_lock(&lock);
    which_function = 0;//set flag as 0
    pthread_cond_broadcast(&cond);//wake up all waiting threads
    running_thread = thr_count;//now all the threads are running
    pthread_cond_wait(&main_thread_cond, &lock);//wait for all threads to finish
    pthread_mutex_unlock(&lock);
}

// ----------------------------------------------------------------------------
// entry function for multi-threading multi-head-attention -> main thread
// @note: YOU CAN NOT MODIFY FUNCTION SIGNATURE!!!
void multi_head_attn(
    float* out,         // output tensor [head, head_size]
    float* q,           // query tensor  [head, head_size]
    float* key_cache,   // cache of history key tensor   [kv_head, seq_len, head_size]
    float* value_cache, // cache of history value tensor [kv_head, seq_len, head_size]
    float* att,         // buffer for attention score [head, seq_len]
    int seq_len,
    int n_heads,
    int head_size,
    int kv_dim,
    int kv_mul) {
    //pass the values to variables
    para.out = out;
    para.q = q;
    para.key_cache = key_cache;
    para.value_cache = value_cache;
    para.att = att;
    para.seq_len = seq_len;
    para.n_heads = n_heads;
    para.head_size = head_size;
    para.kv_dim = kv_dim;
    para.kv_mul = kv_mul;

    //wake up all threads and execute head function
    pthread_mutex_lock(&lock);
    which_function = 1;//set flag as 1
    pthread_cond_broadcast(&cond);//wake up all waiting threads
    running_thread = thr_count;//now all the threads are running
    pthread_cond_wait(&main_thread_cond, &lock);//wait for all threads to finish
    pthread_mutex_unlock(&lock);
}
// YOUR CODE ENDS HERE

// ----------------------------------------------------------------------------
// forward Transformer, you're not allowed to modify this part
float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->q, &s->xq, w->wq + l, dim, dim);
        mat_vec_mul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        mat_vec_mul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        multi_head_attn(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, 
            p->seq_len, p->n_heads, head_size, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        mat_vec_mul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        mat_vec_mul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    mat_vec_mul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop, you're not allowed to modify this part
void generate(char *prompt) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+6) * sizeof(int)); // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // place holder for next token
    int token = prompt_tokens[0]; // place holder of prev token, kickoff as prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0; // to be lazy iniialzied
    while (pos < end_pos) {

        // forward the transformer to get logits for the next token
        float* logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos - start_pos + 1];
        } else if (pos == end_pos - 2) {
            // reaching the end, force it to close by <|im_end|>
            next = 2; // := <|im_end|>
        } else {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens) {
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0) { start_time = time_in_ms(); }
    }
    printf("\n");

    long end_time = time_in_ms();
    // \033[0;32m set color to green and \033[0m reset to default, they won't be generate by LLM
    fprintf(stdout, "\033[0;32mlength: %d, speed (tok/s): %.4f \033[0m\n", 
        pos, (pos - start_pos) / (float) (end_time - start_time) * 1000);
    
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_path     = "model.bin";  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature    = 0.6f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp           = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt         = NULL;  // prompt strings
    int num_prompt       = 0; // number of prompts
    uint64_t rng_seed    = 0; // seed rng with time by default
    int num_thr          = 0;

    if (argc == 4) {
        num_thr  = atoi(argv[1]);
        rng_seed = atoi(argv[2]);
        prompt   = argv[3];
    } else {
        fprintf(stderr, "Usage:   ./seq <num_thr> <seed> <prompt>\n");
        fprintf(stderr, "Example: ./seq 4 42 \"What is Fibonacci Number?\"\n");
        fprintf(stderr, "Note:    <prompt> must be quoted with \"\", only one prompt supported\n");
        exit(1);
    }

    // parameter validation/overrides
    if (num_thr <= 0 || num_thr > 16) { 
        fprintf(stderr, "num_thr must between 1 and 16 \n");
        exit(EXIT_FAILURE);
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // initialize thread pool
    init_thr_pool(num_thr);

    printf("user: %s \n", prompt);
    // perform multi-threading generation
    generate(prompt);
    
    // close thread pool
    close_thr_pool();

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}