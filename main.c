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

void eval(double w,double b){
     for(size_t i = 0; i < TRAIN_SIZE;i++){
        double x = train[i][0];
        double y = x * w + b;
        printf("W = %f , B = %f\n",w,b);
        printf("Actual : %f and Expected : %f\n",y,train[i][1]);
    }
}

double error_rate(double w,double b){
     double result = 0.0f;
     for(size_t i = 0; i < TRAIN_SIZE;i++){
        double x = train[i][0];
        double y = x * w + b;
        double d = y - train[i][1];
        result += square(d);
    }
    return (result /= TRAIN_SIZE);
}

double grad_w (double w, double b){
    double sum = 0;
    for (size_t i =0; i < TRAIN_SIZE;i++){
        double x = train[i][0];
        double pred_y = x * w + b;
        sum += x*(pred_y - train[i][1]);
    }
    return (2.0/TRAIN_SIZE) * sum;
}

double grad_b (double w, double b){
    double sum = 0;
    for (size_t i =0; i < TRAIN_SIZE;i++){
        double x = train[i][0];
        double pred_y = x * w + b;
        sum += (pred_y - train[i][1]);
    }
    return (2.0/TRAIN_SIZE) * sum;
}


int main(void){
        // y = x * w
    srand(1);
    double w = rand_double()*10;
    double b = rand_double()*10;
    eval(w,b);
    printf("error_rate : %f\n",error_rate(w,b));
    double lr = 0.01;
    for(size_t i = 0; i < 5500; i++){
        double dw = grad_w (w,b);
        double db = grad_b (w,b);
        w -= lr * dw;
        b -= lr * db;
    }
    printf("error_rate : %f\n",error_rate(w,b));

    printf("%.10f\n",w);
    eval(w,b);
}
