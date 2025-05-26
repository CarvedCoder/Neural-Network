#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

double train [][2] ={
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},
    {5,10}
};

#define TRAIN_SIZE sizeof(train)/sizeof(train[0])

double rand_double(){
    return ((double)rand()/(double)RAND_MAX);
}

double square (double data){
    return data*data;
}

void eval(double w){
     for(size_t i = 0; i < TRAIN_SIZE;i++){
        double x = train[i][0];
        double y = x * w;

        printf("Actual : %f and Expected : %f\n",y,train[i][1]);
    }
}

double error_rate(double w){
     double result = 0.0f;
     for(size_t i = 0; i < TRAIN_SIZE;i++){
        double x = train[i][0];
        double y = x * w;
        double d = y - train[i][1];
        result += square(d);
    }
    return (result /= TRAIN_SIZE);
}

double derivative (double (*func)(double),double val){
    double h = 1e-6;
    return ((func(val+h)-func(val))/h);
}

int main(void){
        // y = x * w
    srand(1);
    double w = rand_double()*10;
    double b = rand_double()*10;
    eval(w);
    printf("error_rate : %f\n",error_rate(w));
    
    double lr = 1e-3;
    for(size_t i = 0; i < 500; i++){
        double grad_w = derivative(error_rate,w);
        double grad_b = derivative(error_rate,b);
        w -= lr * grad_w;
        b-= lr * grad_b;
        printf("error_rate : %f\n",error_rate(w));
    }

    printf("%.10f\n",w);
    eval(w);
}
