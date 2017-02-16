//System parameters: they are model-specific
double tzeta=6e-5;
double tXi=5e-2;//Piezoelectric auto-coupling (scaling parameter)
double tF=1.;//Amplitud of the forcing (scaling parameter)
double tk=1.;//Piezoelectric  coupling (scaling parameter)
double tK=8e-2;//Spring constant (scaling parameter) (Hamiltonian coupling)
double beta=1.; //To introduce differences between the coupled oscillators
double pi=4.*atan(1.);
double omega=2.1;//*2/3;
double lambda=2e-1;
double eps=0;

double hini=0.05;
double hmin=1.0E-5;
double hmax=1.0;
double tol=1.0e-13;

//Gridsize:
size_t Ntheta=200;
size_t Nc=100;
double cini=0.1;
double cend=1.4;
double csimini=0.2;
double csimend=1.2;
double thetasimini=0;
double thetasimend=1;
