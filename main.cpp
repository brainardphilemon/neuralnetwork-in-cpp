#include<bits/stdc++.h>
using namespace std;

long double genRand(double L , double R)
{
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution<long double> dist(L, R);
   long double x = dist(rng);

    return x;
}

long double relu (double preOut)  // using RELU max(0, x) as default activation function....
{
   double temp = max(0.00 , preOut);
   return temp;
}

class Neuron
{
    public:
    int inSize; // input size of neuron
    double bias = 0.01; // small bias to avoid inconsistency 
    vector <double> weight;


    Neuron(int inputSize): inSize(inputSize),weight(inSize) {
        // randomly initialise weights, bias... [-0.15 , +0.15]
        double L = -0.15;
        double R = 0.15;
        for(int i = 0; i < inputSize; i ++)
        {
            weight[i] = genRand(L,R);
            // weights initialised
        }
    }

    Neuron(int inputSize , string typ): inSize(inputSize),weight(inSize) {
        // input layer neurons...
        weight[0] = 1;
 
    }

    

    // forward function to calc W*x + B and apply activation...
    double forwardNeuron(vector <double> input)
    {
        long double preOut = 0;

        for(int i = 0; i < input.size(); i ++)
        {
            long double temp = weight[i] * input[i];       // wT * X + B
            preOut += temp;
        
        }
        preOut += bias;
       preOut = relu(preOut);

       return preOut;
    }
    


};

vector <Neuron> uidNeurons;

void init(int n)
{
    uidNeurons.reserve(n);
}



void inputLayer(int neuronCount, int startIndex)
{
 
    for(int i = startIndex; i < startIndex + neuronCount; i ++)
    {
      Neuron temp(1, "input");
      uidNeurons.emplace_back(temp);
    }

}

void hiddenLayer(int neuronCount , int prevLayerOut , int startIndex)
{

    int inputSize = prevLayerOut;
    for(int i = startIndex; i < startIndex + neuronCount; i ++)
    {
      Neuron temp(inputSize);
      uidNeurons.emplace_back(temp);
    }
   
}

void regressionOutputLayer(int prevLayerOut)
{
    int outSize = prevLayerOut;
    Neuron OutputNeuron(outSize);
    uidNeurons.emplace_back(OutputNeuron);

}
void classificationOutputLayer(int classCount, int prevLayerOut, int startIndex)
{
    int outSize = prevLayerOut;
    vector<Neuron> outNeurons;
    for(int i = startIndex; i < startIndex + classCount; i ++)
    {
      Neuron temp(outSize);
      uidNeurons.emplace_back(temp);
    }
    
}

map<int, int> layersMetadata;
int totalNeurons = 0;
double learningRate = 0.001;

void createNeuralNetwork(vector<vector<double>> features, vector<int> dimensions, string outputType , int classCount) // a vector with sizes of hidden layers and string as the task with feature vectors too
{
    int instanceCount = features.size();
    int featuresCount = features[0].size();
    for(int i = 0; i < dimensions.size(); i ++)
    {
        totalNeurons += dimensions[i];
    }
    // init(totalNeurons); // initialising size of neurons..

    int index = 0;
    layersMetadata[0] = 0;
    inputLayer(featuresCount, 0);
    index += featuresCount;
    for(int i = 1; i < dimensions.size(); i ++)
    {
            layersMetadata[i] = index;
            hiddenLayer(dimensions[i], dimensions[i - 1], index);
            index += dimensions[i];
        
    }
    
    cout << "Successfully Initialised!" << endl;

}

void MSE( double actual , double predicted) // Mean Squared Error...
{
  double loss = 0.5 * (actual - predicted) * (actual - predicted);
}

double gradientMSE(double actual , double predicted)
{
    double gradient = predicted - actual;
    return gradient;
}

void backProp(double gradient , vector<int> dimensions) 
// every neuron will get some gradient from forward.. (if more than one neuron in front, we just accumulate)
// should use that gradient to update the weights behind it.
{
   // store gradients at every node.
    reverse(dimensions.begin() , dimensions.end());
    vector<double> grads(totalNeurons);
    map <int,double> dp1;
    map <int,double> dp2;
    dp1[0] = gradient;
  int iter = 0;
  int idx = totalNeurons - 1;
  for(auto it = layersMetadata.rbegin(); it != layersMetadata.rend(); it ++)
 {
    for(int i = it->second; i < it->second + dimensions[iter]; i ++)
    {
        for(int j = 0; j < uidNeurons[i].weight.size(); j ++)
        {
            dp2[j] += uidNeurons[i].weight[j] * dp1[i - it->second];
            uidNeurons[i].weight[j] = uidNeurons[i].weight[j] - learningRate * (dp1[i - it->second]);   
            // optimiser a.k.a weight update formula
        }   
        grads[idx] = dp1[i - it->second];
        uidNeurons[i].bias = uidNeurons[i].bias - learningRate * (dp1[i - it->second]);
        idx --;
    }
    dp1 = dp2;
    iter ++;

 }
}

double forwardProp (vector<double> instance, vector<int>dimensions)
{

 int iter = 0;
    for(auto it = layersMetadata.begin(); it != layersMetadata.end(); it ++)
 {
    vector<double> temp;
    for(int i = it->second; i < it->second + dimensions[iter]; i ++)
    {
       if(it == layersMetadata.begin()) continue;
        double temp1;
        temp1 = uidNeurons[i].forwardNeuron(instance);
        temp.push_back(temp1);
        
    }
    iter++;
    instance = temp;

 }

return instance[0];
  
}
void train ( vector<vector<double>> features , vector<double> actualOutput, vector <int>dimensions)
{
    double mae = 0;
    for(int i = 0; i < features.size(); i ++)
    {
       double predict = forwardProp(features[i], dimensions);
       MSE(actualOutput[i], predict);
       mae += abs(actualOutput[i] - predict);
       double grad = gradientMSE(actualOutput[i], predict);
       backProp(grad, dimensions);
    }
  double N = (double)features.size();
    mae /= (double)N;
 cout << "MAE: " << mae << " ";
}


int main()
{

    vector<vector<double>> features =
{   {0.00632,18.0,2.31,0,0.538,6.575,65.2,4.0900,1,296.0,15.3,396.90,4.98},
    {0.02731,0.0,7.07,0,0.469,6.421,78.9,4.9671,2,242.0,17.8,396.90,9.14},
    {0.02729,0.0,7.07,0,0.469,7.185,61.1,4.9671,2,242.0,17.8,392.83,4.03},
    {0.03237,0.0,2.18,0,0.458,6.998,45.8,6.0622,3,222.0,18.7,394.63,2.94},
    {0.06905,0.0,2.18,0,0.458,7.147,54.2,6.0622,3,222.0,18.7,396.90,5.33},
    {0.02985,0.0,2.18,0,0.458,6.430,58.7,6.0622,3,222.0,18.7,394.12,5.21},
    {0.08829,12.5,7.87,0,0.524,6.012,66.6,5.5605,5,311.0,15.2,395.60,12.43},
    {0.14455,12.5,7.87,0,0.524,6.172,96.1,5.9505,5,311.0,15.2,396.90,19.15},
    {0.21124,12.5,7.87,0,0.524,5.631,100.0,6.0821,5,311.0,15.2,386.63,29.93},
    {0.17004,12.5,7.87,0,0.524,6.004,85.9,6.5921,5,311.0,15.2,386.71,17.10},

    {0.22489,12.5,7.87,0,0.524,6.377,94.3,6.3467,5,311.0,15.2,392.52,20.45},
    {0.11747,12.5,7.87,0,0.524,6.009,82.9,6.2267,5,311.0,15.2,396.90,13.27},
    {0.09378,12.5,7.87,0,0.524,5.889,39.0,5.4509,5,311.0,15.2,390.50,15.71},
    {0.62976,0.0,8.14,0,0.538,5.949,61.8,4.7075,4,307.0,21.0,396.90,8.26},
    {0.63796,0.0,8.14,0,0.538,6.096,84.5,4.4619,4,307.0,21.0,380.02,10.26},
    {0.62739,0.0,8.14,0,0.538,5.834,56.5,4.4986,4,307.0,21.0,395.62,8.47},
    {1.05393,0.0,8.14,0,0.538,5.935,29.3,4.4986,4,307.0,21.0,386.85,6.58},
    {0.78420,0.0,8.14,0,0.538,5.990,81.7,4.2579,4,307.0,21.0,386.75,14.67},
    {0.80271,0.0,8.14,0,0.538,5.456,36.6,3.7965,4,307.0,21.0,288.99,11.69},
    {0.72580,0.0,8.14,0,0.538,5.727,69.5,3.7965,4,307.0,21.0,390.95,11.28},

    {1.25179,0.0,8.14,0,0.538,5.570,98.1,3.7979,4,307.0,21.0,376.57,21.02},
    {0.85204,0.0,8.14,0,0.538,5.965,89.2,4.0123,4,307.0,21.0,392.53,13.83},
    {1.23247,0.0,8.14,0,0.538,6.142,91.7,3.9769,4,307.0,21.0,396.90,18.72},
    {0.98843,0.0,8.14,0,0.538,5.813,100.0,4.0952,4,307.0,21.0,394.54,19.88},
    {0.75026,0.0,8.14,0,0.538,5.924,94.1,4.3996,4,307.0,21.0,394.33,16.30},
    {0.84054,0.0,8.14,0,0.538,5.599,85.7,4.4546,4,307.0,21.0,303.42,16.51},
    {0.67191,0.0,8.14,0,0.538,5.813,90.3,4.6820,4,307.0,21.0,376.88,14.81},
    {0.95577,0.0,8.14,0,0.538,6.047,88.8,4.4534,4,307.0,21.0,306.38,17.28},
    {0.77299,0.0,8.14,0,0.538,6.495,94.4,4.4547,4,307.0,21.0,387.94,12.80},
    {1.00245,0.0,8.14,0,0.538,6.674,87.3,4.2390,4,307.0,21.0,380.23,11.98},

    {1.13081,0.0,8.14,0,0.538,5.713,94.1,4.2330,4,307.0,21.0,360.17,22.60},
    {1.35472,0.0,8.14,0,0.538,6.072,100.0,4.1750,4,307.0,21.0,376.73,13.04},
    {1.38799,0.0,8.14,0,0.538,5.950,82.0,3.9900,4,307.0,21.0,232.60,27.71},
    {1.15172,0.0,8.14,0,0.538,5.701,95.0,3.7872,4,307.0,21.0,358.77,18.35},
    {1.61282,0.0,8.14,0,0.538,6.096,96.9,3.7598,4,307.0,21.0,248.31,20.34},
    {0.06417,0.0,5.96,0,0.499,5.933,68.2,3.3603,5,279.0,19.2,396.90,9.68},
    {0.09744,0.0,5.96,0,0.499,5.841,61.4,3.3779,5,279.0,19.2,377.56,11.41},
    {0.08014,0.0,5.96,0,0.499,5.850,41.5,3.9342,5,279.0,19.2,396.90,8.77},
    {0.17505,0.0,5.96,0,0.499,5.966,30.2,3.8473,5,279.0,19.2,393.43,10.13},
    {0.02763,75.0,2.95,0,0.428,6.595,21.8,5.4011,3,252.0,18.3,395.63,4.32},
    {0.03359,75.0,2.95,0,0.428,7.024,15.8,5.4011,3,252.0,18.3,395.62,1.98},
    {0.12744,0.0,6.91,0,0.448,6.770,2.9,5.7209,3,233.0,17.9,385.41,4.84},
    {0.08826,0.0,6.91,0,0.448,6.417,6.6,5.7209,3,233.0,17.9,383.73,6.72},
    {0.15876,0.0,6.91,0,0.448,6.998,45.1,4.1850,3,233.0,17.9,394.63,2.94}
}; 
    vector<double> actualOutput = 
    {24.0,21.6,34.7,33.4,36.2,28.7,22.9,27.1,16.5,18.9,
    15.0,18.9,21.7,20.4,18.2,19.9,23.1,17.5,20.2,18.2,
    13.6,19.6,15.2,14.5,15.6,13.9,16.6,14.8,18.4,21.0,
    12.7,14.5,13.2,13.1,13.5,18.9,20.0,21.0,24.7,30.8,
    34.9,26.6,25.3,24.7};

 // take dimension of the NN


 string outType = "regression";
 cout << "Hey, This is Brainard Philemon's custom made Neural Network in C ++ without using any unusual(weird) libraries.. hehe" << endl;
 for(int i = 0; i < 5; i ++)
 cout << "." << endl;
 cout << "Could you please mention how many hidden layers: ";
 int hiddenCount;
 cin >> hiddenCount;
 cout << "Can you please let us know how many neurons in each hidden layer: " << endl;
 vector<int> dimensions(hiddenCount + 2);
 int featureCount = features[0].size();
 dimensions[0] = featureCount;
 for(int i = 0; i < hiddenCount; i ++)
 {
    cin >> dimensions[i + 1]; 
 }
 dimensions[hiddenCount + 1] = 1;
 

 createNeuralNetwork(features, dimensions, outType, 1);
 cout << "Mention The No of Epochs: " << endl;

 int epochs;
 cin >> epochs;

 for(int i = 0; i < epochs; i ++)
 {
    train(features,actualOutput,dimensions);

    cout << "| Epoch " << i + 1 << " done." << endl;
 }

 for(int i = 0; i < 5; i ++)
 cout << "." << endl;

 cout << "Training Done! Thank you cya !!! " << endl;











//  createNeuralNetwork({{1,2,3},{2,3,4}} , {2 ,4 ,6} , "regression" , 0);




}


// we are done with initialising...
// ig we are done with forward pass.. WE ARENT...
// backprop is the MAJOR thing...
// to keep it simple we will use normal gradient descent rather than stochastic gradient descent.
// done with forward and backwardprop
// for now let's keep the data constant :)
