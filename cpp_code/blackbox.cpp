#include "TinnSo_Cpp.h"
#include <cmath>
#include <algorithm>
class pi_nn p;

void pi_nn::act(float* input) {
    // observation
   // o = input;
    // hidden layer 1 with sigmoid

    for (int i = 0; i < sz_hid_1; ++i) {
        h1[i] = 0;
        for (int j = 0; j < sz_obs; ++j) { // weight
            h1[i] += pi_w1[i][j] * input[j];
        }
        h1[i] += pi_b1[i]; // bias
        h1[i] = 1 / (1 + std::exp((long double) -h1[i])); //sigmoid
    }
    // hidden layer 2 with sigmoid
    for (int i = 0; i < sz_hid_2; ++i) {
        h2[i] = 0;
        for (int j = 0; j < sz_hid_1; ++j) { // weight
            h2[i] += pi_w2[i][j] * h1[j];
        }
        h2[i] += pi_b2[i]; // bias
        h2[i] = 1 / (1 + std::exp((long double) -h2[i])); //sigmoid
    }

    /*// hidden layer 3 with relu
    for (int i = 0; i < sz_hid_3; ++i) {
        h3[i] = 0;
        for (int j = 0; j < sz_hid_2; ++j) { // weight
            h3[i] += pi_w3[i][j]*h2[j];
        }
        h3[i] += pi_b3[i]; // bias
        h3[i] = std::max(h3[i], (float) 0); // relu
    }*/
    // output layer with tanh
    for (int i = 0; i < sz_act; ++i) {
        a[i] = 0;
        for (int j = 0; j < sz_hid_2; ++j) { // weight
            a[i] += pi_w3[i][j] * h2[j];
        }
        a[i] += pi_b3[i]; // bias
        a[i] = std::tanh((long double) a[i]); // tanh
    }

    return;
};

pi_nn::pi_nn() {

}

extern "C" {


	void get_act(float* input, float* output) {
		p.act(input);
		output[0] = p.a[0];
		output[1] = p.a[1];
		return;
	}
}


class Simple {
public:
	int a;
	int b;
	float add();
	void set(float,float);
	Simple() :a(0),b(0) {}
};

float Simple::add()
{
    return a+b;
}


void Simple::set(float a_,float b_)
{
    a = a_;
    b = b_;
}

extern "C" {
	void deleteObj(Simple* ptr)
	{delete ptr;}

	Simple* init()
	{
		Simple* ptr = new Simple; // New object every initialization
		return ptr;
	}

	float add(Simple* obj)
	{return obj->add();}

	void set(Simple* obj, float a_, float b_)
	{obj->set(a_,b_);}
}


