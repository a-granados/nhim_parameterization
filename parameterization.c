#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_interp2d.h>
#include <omp.h>

using std::cout;
//using std::cerr;
using std::endl;
//using std::setw;

double tzeta=6e-5;
//double tzeta=0.;
//double tzeta=1e-2;
//double tXi=0.1;
//double tXi=5e-1;
//double tXi=1e-2;
//double tXi=1e-4;
double tXi=0;
double tF=1.;
double tk=1.;
//double k=500;
double beta=1.;
double pi=4.*atan(1.);
double omega=2.1;//*2/3;
//double omega=2.*pi/7;
//double lambda=0.01;
double lambda=2e-1;
double eps=0;

double hini=0.05;
double hmin=1.0E-5;
double hmax=1.0;
double tol=1.0e-13;

//Gridsize:
size_t Ntheta=400;
size_t Nc=200;
double cini=0.1;
double cend=1.4;
double csimini=0.2;
double csimend=1.2;
double thetasimini=0;
double thetasimend=1;

//int ndim=5;
int count;

const gsl_interp2d_type *T=gsl_interp2d_bicubic;
//const gsl_interp2d_type *T=gsl_interp2d_bilinear;
#pragma omp threadprivate(tol)


#include "../piezo_library.c"

using namespace std;

void F_0(double *thetax,double **DF);
void F_eps(double theta, double c,double x, double y, double w, double *Fthetax);
void invF_eps(double theta, double c,double x, double y, double w, double *Fthetax);
void multMatrix(double **C,double **A, int fA, int cA, double **B,int fB,int cB);
void multMatrix_array(double *C,double *A, int fA, int cA, double *B,int fB,int cB);
//void P_0(double tau, double c, double **P0);
void P_0(double tau, double c, gsl_matrix *P0);
void D_uv_thetac(double u,double v, double alpha,double dalpha,double c, double *DinvPhi);
//void find_ini_Lambda(gsl_matrix *Lambdaaux);
void Lambda0_P0(double theta,double c,gsl_matrix *Lambdaaux,gsl_matrix *P0aux,gsl_matrix *invP0aux,double *f,double *wp);
void ini_functions(double **thetac,double **LambdaLvals, double **LambdaSvals, double *LambdaUvals, double **Kvals, double **Pvals, double **invPvals,double **fvals);
void compute_invf(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals);
void compute_E(double **thetac,double **Evals,double **fvals,double **Kvals,gsl_interp2d **K,double **FKvals);
void torus_correction_unstable(double **thetac,double *LambdaUvals,double *zetaUvals, double **fvals, double **Kvals,double **invPvals,double **Evals);
void torus_correction_stable(double **thetac,double **LambdaSvals,double **zetaSvals, double **invfvals, double **Kvals, double **invPvals,double **Evals,gsl_interp2d **K);
void innerdynamics_correction(double **thetac,double **Deltafvals,double **fvals,double **invPvals, double **Evals);
double modulo(double a, double b);
void compute_FK_DFK(double **Kvals,double **F, double **DF);
void F_DF_eps(double theta, double c,double x, double y, double w, double *Fthetax,double *DF);
void compute_FK_eps(double **Kvals,double **FKvals);
void compute_EredN(double **thetac,double **fvals,double **invPvals,double **Pvals,double **DFKvals,double **LambdaSvals,double *LambdaUvals,double **Eredvals);
void invertPvals(double **Pvals,double **invPvals);
void invertPvals2(double **invPvals, double *DeltaP,int k);
void QLS_correction(double **thetac,double **LambdaLvals,double **LambdaSvals,double **fvals,double **EredNvals,double **QLSvals);
void QLU_correction(double **thetac,double **LambdaLinvfvals, double *LambdainvfUvals, double **invfvals, double **EredNvals, double **QLUvals);
void QUS_correction(double **thetac, double **LambdaSvals, double *LambdaUvals,double **fvals,double **EredNvals, double **QUSvals);
void QSU_correction(double **thetac, double **LambdaSinvfvals,double *LambdaUinvfvals,double **invfvals,double **EredNvals,double **QSUvals);
void ini_functions2(double **thetac,double **LambdaLvals, double **LambdaSvals, double *LambdaUvals, double **Kvals, double **Pvals, double **invPvals,double **fvals,double **uvvals,double *alphavals,double *dalphavals);
void extendgrid(double **thetac,double **fvals,gsl_interp2d **f,double xpoint,double ypoint,double **extrathetac,double **extrafvals,gsl_interp2d **extraf,int xgrid,int ygrid);
void lift_data(double *data);
void correct_normal_bundle(double **thetac,double **fvals,double **invfvals,double **Kvals, double **DFKvals,double **Pvals, double **invPvals,double **LambdaLvals,double **LambdaSvals, double *LambdaUvals,double **EredNvals);
void iterate_inner_dynamics(double **thetac,double **fvals,gsl_interp2d **f,gsl_interp2d **K,double **Kvals,string filename);
void write_data_in_file(double *theta,double *c,double **data, int ncols, string filename);
void write_data_in_file2(double *theta,double *c,double *data, string filename);
void write_data_in_file3(double **fvals,double **data, int ncols, string filename);
double one_step_of_Newton(double **thetac,double **fvals,double **invfvals, double **Kvals,double **FKvals,double **LambdaLvals, double **LambdaSvals,double *LambdaUvals,double **Pvals, double **invPvals);
void clean_data(double **data,int ncols);
void clean_data2(double *data);
void compute_DFeps(double **thetac, double **uvvals, double *alphavals,double *dalphavals, double **Kvals, double **Evals);
void iterate_FK(double **thetac, double **Kvals);
void iterate_FK2(double **thetac, double **Kvals);
void predict_fKeps(double **thetac,double **LambdaSvals, double *LambdaUvals, double **invPvals, double **Pvals, double **Kvals,double **fvals);
void compute_DepsF(double **thetac,double **Kvals,double **EDepsFvals);
void shot_manifolds(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals,gsl_interp2d **invf,double **Pvals,double **Kvals);
void shot_manifolds2(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals,gsl_interp2d **invf,double **Pvals,double **Kvals);
void compute_invferror(double **thetac, double **fvals, gsl_interp2d **f,double **invfvals,gsl_interp2d **invf);

int main (){
  cout.precision(15);
  size_t i,j,k,l;

  //Grid:
  double **thetac=new double*[2];
  thetac[0]=new double[Ntheta];
  thetac[1]=new double[Nc];
  //The grid:
  for (i=0;i<Ntheta;i++){
    thetac[0][i]=double(i)/double(Ntheta-1);
  }
  for (i=0;i<Nc;i++){
    thetac[1][i]=cini+double(i)*(cend-cini)/double((Nc-1.));
  }

  //Functions that need to be evaluated outiside the grid values
  //F, K, P, zeta^u, zeta^s, LambdaS
  //F is given, the others need to be interpolated
  //invP is a 5x5 matrix whose components are 2dsplines
  //
  //P:
  double **Pvals=new double*[25];
  for (i=0;i<25;i++){
    Pvals[i]=new double[Ntheta*Nc];
  }
  //invP: (this will be used for the first iteration)
  double **invPvals=new double*[25];
  for (i=0;i<25;i++){
    invPvals[i]=new double[Ntheta*Nc];
  }
  //K:
  double **Kvals=new double*[5];
  for (i=0;i<5;i++){
    Kvals[i]=new double[Ntheta*Nc];
  }
  //FK:
  double **FKvals=new double*[5];
  for (i=0;i<5;i++){
    FKvals[i]=new double[Ntheta*Nc];
  }
  //LambdaS:
  double **LambdaSvals=new double*[4];
  for (i=0;i<4;i++){
    LambdaSvals[i]=new double[Ntheta*Nc];
  }
  //f:
  double **fvals=new double*[2];
  fvals[0]=new double[Ntheta*Nc];
  fvals[1]=new double[Ntheta*Nc];
  double **invfvals=new double*[2];
  invfvals[0]=new double[Ntheta*Nc];
  invfvals[1]=new double[Ntheta*Nc];
  //LambdaL:
  double **LambdaLvals=new double*[4];
  //LambdaU
  double *LambdaUvals=new double[Ntheta*Nc];
  for (i=0;i<4;i++){
    LambdaLvals[i]=new double[Ntheta*Nc];
  }
  double **uvvals=new double*[6];
  for (i=0;i<6;i++){
    uvvals[i]=new double[Ntheta*Nc];
  }
  double *alphavals=new double[Nc];
  double *dalphavals=new double[Nc];
  double err,Newtol=1e-7;
  int maxiter=8;
  string filename;

  //-------------------------------------------------------------------------------
  //-------------------------------Initialization------------------------------
  //-------------------------------------------------------------------------------
  eps=0;
  count=0;
  ofstream fperiods;
  fperiods.open("unperperiods0.tna");
  double ctmp;
  for (i=0;i<100;i++){
    ctmp=0.01+double(i)*(2.-0.01)/100.;
    fperiods<<ctmp<<" "<<periodporbit(sqrt(2.*ctmp))<<endl;
  }
  fperiods.close();
  cout <<"Initializing things..."<<endl;
  //ini_functions(thetac,LambdaLvals,LambdaSvals,LambdaUvals,Kvals,Pvals,invPvals,fvals);
  ini_functions2(thetac,LambdaLvals,LambdaSvals,LambdaUvals,Kvals,Pvals,invPvals,fvals,uvvals,alphavals,dalphavals);
  cout<<"Iterating F at K(theta,c).."<<endl;
  //iterate_FK(thetac,Kvals);
  //iterate_FK2(thetac,Kvals);
  cout <<"Done."<<endl;

  //cout<<"Now computing the inverse..."<<endl;
  //Initial values for the Newton to compute the inverse: (this is exact for eps=0)
  #pragma omp parallel for private(j)
  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      invfvals[0][j*Ntheta+i]=modulo(thetac[0][i]-(fvals[0][j*Ntheta+i]-thetac[0][i]),1.);
      invfvals[1][j*Ntheta+i]=thetac[1][j];
    }
  }
  lift_data(invfvals[0]);
  cout<<"Initialization done."<<endl;

  //Writing ini data:
  filename="Pvals";
  write_data_in_file(thetac[0],thetac[1],Pvals,25,filename);
  filename="invPvals";
  write_data_in_file(thetac[0],thetac[1],invPvals,25,filename);
  filename="LambdaLvals";
  write_data_in_file(thetac[0],thetac[1],LambdaLvals,4,filename);
  filename="LambdaSvals";
  write_data_in_file(thetac[0],thetac[1],LambdaSvals,4,filename);
  filename="LambdaUvals";
  write_data_in_file2(thetac[0],thetac[1],LambdaUvals,filename);
  filename="fvals";
  write_data_in_file(thetac[0],thetac[1],fvals,2,filename);
  filename="invfvals";
  write_data_in_file(thetac[0],thetac[1],invfvals,2,filename);
  filename="Kvals";
  write_data_in_file(thetac[0],thetac[1],Kvals,5,filename);


  eps=6e-2;
  //----------------------------
  //We start the Newton method:
  //----------------------------
  err=10*Newtol;
  count=1;
  //-----Continuation-------
  //cout<<"Computing continuation..."<<endl;
  //predict_fKeps(thetac,LambdaSvals,LambdaUvals,invPvals,Pvals,Kvals,fvals);
  //cout <<"Done"<<endl;
  while (err>Newtol && count<maxiter){
    cout<<"Iterating F at K(theta,c).."<<endl;
    //iterate_FK(thetac,Kvals);
    //iterate_FK2(thetac,Kvals);
    cout <<"Done."<<endl;
    err=one_step_of_Newton(thetac,fvals,invfvals,Kvals,FKvals,LambdaLvals,LambdaSvals,LambdaUvals,Pvals,invPvals);
    cout <<"Done."<<endl;
    /*
    cout <<"----------------------------- "<<endl;
    cout <<"Error in Current Newton step: "<<err<<endl;
    cout <<"----------------------------- "<<endl;
    */

    filename="LambdaSvals";
    write_data_in_file(thetac[0],thetac[1],LambdaSvals,4,filename);
    filename="LambdaLvals";
    write_data_in_file(thetac[0],thetac[1],LambdaLvals,4,filename);
    filename="LambdaUvals";
    write_data_in_file2(thetac[0],thetac[1],LambdaUvals,filename);
    filename="Kvals";
    write_data_in_file(thetac[0],thetac[1],Kvals,5,filename);
    filename="fvals";
    write_data_in_file(thetac[0],thetac[1],fvals,2,filename);
    filename="invfvals";
    write_data_in_file(thetac[0],thetac[1],invfvals,2,filename);
    count++;
  }

  //Free memory:
  //Data arrays:
  for (i=0;i<6;i++){
    delete[] uvvals[i];
  }
  delete[] alphavals;
  delete[] dalphavals;
  delete[] uvvals;
  delete[] fvals[0];
  delete[] fvals[1];
  delete[] fvals;
  delete[] invfvals[0];
  delete[] invfvals[1];
  delete[] invfvals;
  for (i=0;i<25;i++){
    delete[] Pvals[i];
    delete[] invPvals[i];
  }
  delete[] Pvals;
  delete[] invPvals;
  for (i=0;i<5;i++){
    delete[] Kvals[i];
    delete[] FKvals[i];
  }
  delete[] Kvals;
  delete[] FKvals;
  for (i=0;i<4;i++){
    delete[] LambdaLvals[i];
    delete[] LambdaSvals[i];
  }
  delete[] LambdaLvals;
  delete[] LambdaSvals;
  delete[] LambdaUvals;
  delete[] thetac[0];
  delete[] thetac[1];
  delete[] thetac;
}

void ini_functions2(double **thetac,double **LambdaLvals, double **LambdaSvals, double *LambdaUvals, double **Kvals, double **Pvals, double **invPvals,double **fvals,double **uvvals,double *alphavals,double *dalphavals){

  //Here we compute Lambda{L,S,U}vals, Kvals, Pvals,invPvals and fvals
  //for eps=0. Most of these functions do not depend on theta and c,
  //so they are computed only once and copied for all the grid. This
  //is the case of most compnents of P=(DK N), DF (and hence Lambda)
  //
  //Here we could also return P(f(theta,c)), as have to compute it
  //anyway and it is used in the torus correction in the unstable
  //direction (at least), although it is probably not worth because it
  //would only be useful for the first iteration, as we need to
  //interpolate it for the upcoming ones.
  int i,j,k,ndim;
  //uvvals contains u,v and D_{theta,c}uv. They will be used in compute_DFeps
  double h,t,t2,aux;
  double **Lambdavals=new double*[25];
  double **DFKvals=new double*[25];//This is just for control proposes.
  for (i=0;i<25;i++){
    Lambdavals[i]=new double[Ntheta*Nc];
    DFKvals[i]=new double[Ntheta*Nc];
  }
  cout <<"  Computing P0(theta,c), invP0(theta,c),K(theta,c) and f(theta,c)..."<<endl;
  //We first compute P0(theta,c),invP0(theta,c),K(theta,c),f(theta,c): 
  #pragma omp parallel for private(h,t,t2,i,ndim,k,aux)
  for (j=0;j<Nc;j++){
    double *uv=new double[6];
    double *uvw=new double[12];
    double *duvw=new double[12];
    double *duv=new double[2];
    double *z=new double[30];
    double *DPhi=new double[25];
    double *DinvPhi=new double[25];
    double *DF=new double[25];
    double *aux1=new double[25];
    double *aux2=new double[25];
    double *Dthetacuv=new double[4];
    alphavals[j]=periodporbit(sqrt(2.*thetac[1][j]));
    dalphavals[j]=dperiodporbit(sqrt(2.*thetac[1][j]));
    t=0;
    h=hini;
    uv[0]=0;
    uv[1]=sqrt(2.*thetac[1][j]);
    uv[2]=uv[5]=1.;
    uv[3]=uv[4]=0.;
    for (i=0;i<Ntheta;i++){
      //We compute f:
      //We need to interpolate fvals when computing invfvals, so we
      //need it continuous.
      fvals[0][j*Ntheta+i]=thetac[0][i]+2.*pi/(omega*alphavals[j]);
      fvals[1][j*Ntheta+i]=thetac[1][j];

      //We compute the first 4 components of K (5th is below):
      Kvals[0][j*Ntheta+i]=thetac[0][i];
      Kvals[1][j*Ntheta+i]=thetac[1][j];
      Kvals[2][j*Ntheta+i]=0.;
      Kvals[3][j*Ntheta+i]=0.;
      
      //We now go for P_0(theta,c):
      ndim=6;
      ini_rk78(ndim);
      h=hini;
      while (t<thetac[0][i]*alphavals[j]){
	rk78(&t,uv,&h,tol,hmin,hmax,ndim,vfieldUvar2);
      }
      h=-(t-thetac[0][i]*alphavals[j]);
      rk78(&t,uv,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar2);
      end_rk78(ndim);
      uvvals[0][j*Ntheta+i]=uv[0];
      uvvals[1][j*Ntheta+i]=uv[1];
      //D{theta,c}uv: (needed to compute DF)
      duv[0]=uv[1];
      duv[1]=0.5*beta*uv[0]*(1.-uv[0]*uv[0]);
      uvvals[2][j*Ntheta+i]=duv[0]*alphavals[j];
      uvvals[3][j*Ntheta+i]=(duv[0]*dalphavals[j]*thetac[0][i]+uv[3])/sqrt(2.*thetac[1][j]);
      uvvals[4][j*Ntheta+i]=duv[1]*alphavals[j];
      uvvals[5][j*Ntheta+i]=(duv[1]*dalphavals[j]*thetac[0][i]+uv[5])/sqrt(2.*thetac[1][j]);
      //We now compute w^p(u,v) and D_{u,v,w}w^p: (needed in DK)
      uvw[0]=uv[0];
      uvw[1]=uv[1];
      //w^p(u,v):
      uvw[2]=omegap_num(uvw[0],uvw[1]);
      Kvals[4][j*Ntheta+i]=uvw[2];
      for (k=3;k<12;k++){
	uvw[k]=0.;
      }
      uvw[3]=1.;
      uvw[7]=1.;
      uvw[11]=1.;
      //duvw could also be computed after integrating a time alpha,
      //but it should be the same due to periodicity.
      vfieldUvar(t,uvw,ndim,duvw);
      t2=0;
      h=hini;
      ndim=12;
      ini_rk78(ndim);
      while(t2<alphavals[j]){
	rk78(&t2,uvw,&h,tol,hmin,hmax,ndim,vfieldUvar);
      }
      h=-(t2-alphavals[j]);
      rk78(&t2,uvw,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar);
      end_rk78(ndim);
      /*
      cout<<endl<<endl;
      cout<<"uv for "<<thetac[0][i]<<" "<<thetac[1][j]<<endl;
      cout<<uv[0]<<" "<<uv[1]<<endl;
      cout<<endl<<"uvw:";
      for (k=0;k<12;k++){
	if (k%3==0){
	  cout<<endl;
	}
	cout<<left<<setw(10)<<uvw[k]<<" ";
      }
      */

      //With this we compute D_{theta,c}w^p using implicit function theorem to
      //the equation:
      //Pi_w(\varphi_{uvw}(alpha;u,v,w))-w=0
      //partial w^p/partial theta:
      aux=uvw[9]*uvvals[2][j*Ntheta+i]+uvw[10]*uvvals[4][j*Ntheta+i];
      aux=-aux/(uvw[11]-1.);
      Pvals[20][j*Ntheta+i]=aux;
      //partial w^p/partial c:
      aux=uvw[9]*uvvals[3][j*Ntheta+i]+uvw[10]*uvvals[5][j*Ntheta+i]+duvw[2]*dalphavals[j]/sqrt(2.*thetac[1][j]);
      aux=-aux/(uvw[11]-1.);
      Pvals[21][j*Ntheta+i]=aux;
      //The rest of DK:
      for (k=0;k<8;k++){
	if (k==0 ||k==3){
	  //Pvals[k/2*3+k][j*Ntheta+i]=1.;
	  Pvals[k/2*5+k%2][j*Ntheta+i]=1.;
	}
	else{
	  //Pvals[k/2*3+k][j*Ntheta+i]=0.;
	  Pvals[k/2*5+k%2][j*Ntheta+i]=0.;
	}
      }

      //Now N:
      //First we compute DF, which is also needed in Pvals[22] and Pvals[24].
      //Apanyu temporal:---------------------------------
      z[0]=z[1]=0.;
      z[2]=uvvals[0][j*Ntheta+i];
      z[3]=uvvals[1][j*Ntheta+i];
      z[4]=Kvals[4][j*Ntheta+i];
      for (k=5;k<30;k++){
	if ((k-5)%6==0){
	  z[k]=1.;
	}
	else{
	  z[k]=0.;
	}
      }
      t2=0.;
      h=hini;
      ndim=30;
      ini_rk78(ndim);
      while (t2<2.*pi/omega){
	rk78(&t2,z,&h,tol,hmin,hmax,ndim,vfieldunpertvar);
      }
      h=-(t2-2.*pi/omega);
      rk78(&t2,z,&h,tol,fabs(h),fabs(h),ndim,vfieldunpertvar);
      end_rk78(ndim);
      for (k=0;k<25;k++){
	DF[k]=z[k+5];
      }
      for (k=0;k<25;k++){
	DPhi[k]=0.;
      }
      DPhi[2]=1.;
      DPhi[8]=1.;
      DPhi[10]=uvvals[2][j*Ntheta+i];
      DPhi[11]=uvvals[3][j*Ntheta+i];
      DPhi[15]=uvvals[4][j*Ntheta+i];
      DPhi[16]=uvvals[5][j*Ntheta+i];
      DPhi[24]=1.;
      /*
      cout <<"Dphi2: ";
      for (k=0;k<25;k++){
	if (k%5==0){
	  cout <<endl;
	}
	cout<< DPhi[k]<<" ";
      }
      */
      multMatrix_array(aux1,DF,5,5,DPhi,5,5);
      D_uv_thetac(z[2],z[3],alphavals[j],dalphavals[j],thetac[1][j],Dthetacuv);
      for (k=0;k<25;k++){
	DinvPhi[k]=0.;
      }
      DinvPhi[2]=Dthetacuv[0];
      DinvPhi[3]=Dthetacuv[1];
      DinvPhi[7]=Dthetacuv[2];
      DinvPhi[8]=Dthetacuv[3];
      DinvPhi[10]=1.;
      DinvPhi[16]=1.;
      DinvPhi[24]=1.;

      multMatrix_array(DF,DinvPhi,5,5,aux1,5,5);
      //We know the value of some elements:
      DF[0]=1.;
      DF[2]=0;
      DF[3]=0;
      DF[4]=0;
      DF[5]=0;
      DF[6]=1.;
      DF[7]=0;
      DF[8]=0;
      DF[9]=0;
      DF[10]=0;
      DF[11]=0;
      DF[14]=0;
      DF[15]=0;
      DF[16]=0;
      DF[19]=0;
      for (k=0;k<25;k++){
	DFKvals[k][j*Ntheta+i]=DF[k];
      }
      /*
      //Alternative to DF:-----------
      //They coincide.
      for (k=0;k<25;k++){
	DFKvals[k][j*Ntheta+i]=0;
      }
      DFKvals[0][j*Ntheta+i]=1.;
      DFKvals[1][j*Ntheta+i]=-2.*pi/(omega*alphavals[j]*alphavals[j])*dalphavals[j]/sqrt(2*thetac[1][j]);
      DFKvals[6][j*Ntheta+i]=1.;
      DFKvals[12][j*Ntheta+i]=z[5];
      DFKvals[13][j*Ntheta+i]=z[6];
      DFKvals[17][j*Ntheta+i]=z[10];
      DFKvals[18][j*Ntheta+i]=z[11];
      //These two are wrong:
      DFKvals[20][j*Ntheta+i]=z[27];
      DFKvals[21][j*Ntheta+i]=z[28];
      //---
      DFKvals[22][j*Ntheta+i]=z[25];
      DFKvals[23][j*Ntheta+i]=z[26];
      DFKvals[24][j*Ntheta+i]=z[29];
      //Up to here the alternative to DF--------
      */
      /*
      cout <<endl<<"Vals= "<<thetac[0][i]<<" "<<thetac[1][j];
      for (k=0;k<25;k++){
	if (k%5==0){
	  cout <<endl;
	}
	cout <<DF[k]<<" ";
      }
      cout<<endl;
      */

      //Now we compute P0:
      for (k=0;k<6;k++){
	Pvals[k/3*5+k%3+2][j*Ntheta+i]=0.;
      }
      Pvals[12][j*Ntheta+i]=1.;
      Pvals[13][j*Ntheta+i]=0.;
      Pvals[14][j*Ntheta+i]=1.;
      Pvals[17][j*Ntheta+i]=-1./sqrt(2.);
      Pvals[18][j*Ntheta+i]=0.;
      Pvals[19][j*Ntheta+i]=1./sqrt(2.);
      Pvals[22][j*Ntheta+i]=(z[26]/sqrt(2.)-z[25])/(z[29]-exp(-1./sqrt(2.)*2.*pi/omega));
      Pvals[23][j*Ntheta+i]=1.;
      Pvals[24][j*Ntheta+i]=-(z[26]/sqrt(2.)+z[25])/(z[29]-exp(1./sqrt(2.)*2.*pi/omega));
/*
  cout <<endl<<endl<<"P0 for: "<<thetac[0][i]<<" "<<thetac[1][j]<<endl;
  for (k=0;k<25;k++){
      if (k%5==0){
	cout<<endl;
      }
      cout<<left<<setw(10)<<Pvals[k][j*Ntheta+i]<<" ";
  }
  cout <<endl<<endl<<"DtF for: "<<thetac[0][i]<<" "<<thetac[1][j]<<endl;
  for (k=0;k<25;k++){
      if (k%5==0){
	cout<<endl;
      }
      cout<<left<<setw(10)<<z[k+5]<<" ";
  }
*/

      //Now the inverse: P_0(theta,c)^{-1}
      for (k=0;k<12;k++){
	if (k==0 || k==6){
	  invPvals[k][j*Ntheta+i]=1.;
	}
	else{
	  invPvals[k][j*Ntheta+i]=0.;
	}
      }
      invPvals[12][j*Ntheta+i]=0.5;
      invPvals[13][j*Ntheta+i]=-1./sqrt(2.);
      invPvals[14][j*Ntheta+i]=0.;
      invPvals[15][j*Ntheta+i]=-Pvals[20][j*Ntheta+i];
      invPvals[16][j*Ntheta+i]=-Pvals[21][j*Ntheta+i];
      invPvals[17][j*Ntheta+i]=-0.5*Pvals[22][j*Ntheta+i];
      invPvals[17][j*Ntheta+i]+=-0.5*Pvals[24][j*Ntheta+i];
      invPvals[18][j*Ntheta+i]=1./sqrt(2.)*Pvals[22][j*Ntheta+i];
      invPvals[18][j*Ntheta+i]+=-1./sqrt(2.)*Pvals[24][j*Ntheta+i];
      invPvals[19][j*Ntheta+i]=1.;
      invPvals[20][j*Ntheta+i]=0.;
      invPvals[21][j*Ntheta+i]=0.;
      invPvals[22][j*Ntheta+i]=0.5;
      invPvals[23][j*Ntheta+i]=1./sqrt(2.);
      invPvals[24][j*Ntheta+i]=0.;

      //Now we multiply DF*P0 and store it in Lambdavals
      for (k=0;k<25;k++){
	aux1[k]=Pvals[k][j*Ntheta+i];
      }
      multMatrix_array(aux2,DF,5,5,aux1,5,5);
      for (k=0;k<25;k++){
	Lambdavals[k][j*Ntheta+i]=aux2[k];
      }
    }
    delete[] uv;
    delete[] uvw;
    delete[] duv;
    delete[] duvw;
    delete[] z;
    delete[] DPhi;
    delete[] DinvPhi;
    delete[] DF;
    delete[] aux1;
    delete[] aux2;
    delete[] Dthetacuv;
  }
  clean_data(Pvals,25);
  clean_data(invPvals,25);
  clean_data(DFKvals,25);
  ofstream fperiods;
  fperiods.open("unperperiods.tna");
  for (j=0;j<Nc;j++){
    fperiods<<thetac[1][j]<<" "<<alphavals[j]<<endl;
  }
  fperiods.close();
  string filename="DFKvals";
  write_data_in_file(thetac[0],thetac[1],DFKvals,25,filename);
  cout <<"  done"<<endl;

  cout<<"  Computing P_0(f(theta,c))^{-1}..."<<endl;
  //We now go for P_0(f(theta,c))^{-1}:
  #pragma omp parallel for private(h,t,t2,i,ndim,k,aux)
  for (j=0;j<Nc;j++){
    double *uv=new double[6];
    double *uvw=new double[12];
    double *duvw=new double[12];
    double *duv=new double[2];
    double *aux1=new double[25];
    double *aux2=new double[25];
    double *Pf=new double[25];
    double *invPf=new double[25];
    double *z=new double[30];
    double omegap_tmp;
    ndim=6;
    t=0.;
    h=hini;
    uv[0]=0;
    uv[1]=sqrt(2.*thetac[1][j]);//For eps=0, c does not depend on theta
    uv[2]=uv[5]=1.;
    uv[3]=uv[4]=0.;
    for (i=0;i<Ntheta;i++){
      if (i!=0 && modulo(fvals[0][j*Ntheta+i],1.)<modulo(fvals[0][j*Ntheta+i-1],1.) ){
	//Store everything and use it here.
	//We start integrating again from u=0,v=sqrt(2c):
	t=0.;
	uv[0]=0;
	uv[1]=sqrt(2.*fvals[1][j*Ntheta+i]);
	uv[2]=uv[5]=1.;
	uv[3]=uv[4]=0.;
      }
      ndim=6;
      ini_rk78(ndim);
      h=hini;
      while (t<modulo(fvals[0][j*Ntheta+i],1.)*alphavals[j]){
	rk78(&t,uv,&h,tol,hmin,hmax,ndim,vfieldUvar2);
      }
      h=-(t-modulo(fvals[0][j*Ntheta+i],1.)*alphavals[j]);
      rk78(&t,uv,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar2);
      end_rk78(ndim);
      //We now compute w^p(u,v) and D_{u,v,w}w^p: (needed in DK)
      uvw[0]=uv[0];
      uvw[1]=uv[1];
      //w^p(u,v):
      uvw[2]=omegap_num(uvw[0],uvw[1]);
      omegap_tmp=uvw[2];
      for (k=3;k<12;k++){
	uvw[k]=0.;
      }
      uvw[3]=1.;
      uvw[7]=1.;
      uvw[11]=1.;
      ndim=12;
      vfieldUvar(t,uvw,ndim,duvw);
      t2=0.;
      h=hini;
      ini_rk78(ndim);
      while(t2<alphavals[j]){
	rk78(&t2,uvw,&h,tol,hmin,hmax,ndim,vfieldUvar);
      }
      h=-(t2-alphavals[j]);
      rk78(&t2,uvw,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar);
      end_rk78(ndim);
      //With this we compute D_{uv}w^p using implicit function theorem to
      //the equation:
      //Pi_w(\phi_{uvw}(alpha;u,v,w))-w=0
      //partial w^p/partial tau:
      aux=uvw[9]*uv[2]+uvw[10]*uv[4];
      aux=-aux/(uvw[11]-1.);
      Pf[20]=aux;
      //partial w^p/partial c:
      aux=uvw[9]*uv[3]+uvw[10]*uv[5]+duvw[2]*dalphavals[j]/sqrt(2.*thetac[1][j]);
      aux=-aux/(uvw[11]-1.);
      Pf[21]=aux;
      //The rest of DK:
      for (k=0;k<8;k++){
	if (k==0 ||k==3){
	  //Pf[k/2*3+k]=1.;
	  Pf[k/2*5+k%2]=1.;
	}
	else{
	  //Pf[k/2*3+k]=0.;
	  Pf[k/2*5+k%2]=0.;
	}
      }
      //Now N:
      //For that we need to compute DF to compute P[22] and P[24]
      //Apanyu temporal:---------------------------------
      z[0]=z[1]=0.;
      z[2]=uv[0];
      z[3]=uv[1];
      z[4]=omegap_tmp;
      for (k=5;k<30;k++){
	if ((k-5)%6==0){
	  z[k]=1.;
	}
	else{
	  z[k]=0.;
	}
      }
      t2=0;
      h=hini;
      ndim=30;
      ini_rk78(ndim);
      while (t2<2.*pi/omega){
	rk78(&t2,z,&h,tol,hmin,hmax,ndim,vfieldunpertvar);
      }
      h=-(t2-2.*pi/omega);
      rk78(&t2,z,&h,tol,fabs(h),fabs(h),ndim,vfieldunpertvar);
      end_rk78(ndim);
      //Fins aqui l'apanyu temporal.-----------------------
      for (k=0;k<6;k++){
	//Pf[k/3*2+k+2]=0.;
	Pf[2+k/2*5+k%2]=0.;
      }
      Pf[12]=1.;
      Pf[13]=0.;
      Pf[14]=1.;
      Pf[17]=-1./sqrt(2.);
      Pf[18]=0.;
      Pf[19]=1./sqrt(2.);
      Pf[22]=(z[26]/sqrt(2.)-z[25])/(z[29]-exp(-1./sqrt(2.)*2.*pi/omega));
      Pf[23]=1.;
      Pf[24]=-(z[26]/sqrt(2.)+z[25])/(z[29]-exp(1./sqrt(2.)*2.*pi/omega));
      //Now the inverse: P_0(f(theta,c))^{-1}
      for (k=0;k<12;k++){
	if (k==0 || k==6){
	  invPf[k]=1.;
	}
	else{
	  invPf[k]=0.;
	}
      }
      invPf[12]=0.5;
      invPf[13]=-1./sqrt(2.);
      invPf[14]=0.;
      invPf[15]=-Pf[20];
      invPf[16]=-Pf[21];
      invPf[17]=-0.5*Pf[22];
      invPf[17]+=-0.5*Pf[24];
      invPf[18]=1./sqrt(2.)*Pf[22];
      invPf[18]+=-1./sqrt(2.)*Pf[24];
      invPf[19]=1.;
      invPf[20]=0.;
      invPf[21]=0.;
      invPf[22]=0.5;
      invPf[23]=1./sqrt(2.);
      invPf[24]=0.;

      //We finally obtain Lambda:
      for (k=0;k<25;k++){
	aux1[k]=Lambdavals[k][j*Ntheta+i];
      }
      multMatrix_array(aux2,invPf,5,5,aux1,5,5);
      //Manual correction: we know some elements are zero and due to
      //numerical errors they are not exactly.
      aux2[5]=0.;
      aux2[14]=0.;
      aux2[17]=0.;
      aux2[19]=0.;
      aux2[22]=0.;
      for (k=0;k<25;k++){
	Lambdavals[k][j*Ntheta+i]=aux2[k];
      }
    }
    delete[] uv;
    delete[] uvw;
    delete[] duvw;
    delete[] duv;
    delete[] Pf;
    delete[] invPf;
    delete[] aux1;
    delete[] aux2;
    delete[] z;
  }
  clean_data(Lambdavals,25);
  cout <<"  done."<<endl;
  ofstream fLambdavals;
  fLambdavals.open("Lambdavals.tna");
  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      fLambdavals<<thetac[0][i]<<" "<<thetac[1][j];
      for (k=0;k<25;k++){
	if (k%5==0){
	  fLambdavals<<endl;
	}
	fLambdavals<<Lambdavals[k][j*Ntheta+i]<<" ";
      }
      fLambdavals<<endl;
    }
  }
  fLambdavals.close();

  //Finally we store LambdaL, LambdaS and LambdaU:
  #pragma omp parallel for private(j,k)
  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      for (k=0;k<4;k++){
	LambdaLvals[k][j*Ntheta+i]=Lambdavals[k/2*5+k%2][j*Ntheta+i];
	LambdaSvals[k][j*Ntheta+i]=Lambdavals[12 + k/2*5+k%2][j*Ntheta+i];
      }
      LambdaUvals[j*Ntheta+i]=Lambdavals[24][j*Ntheta+i];
    }
  }
/*
  //For checking reasons:-------
  double **EredNvals=new double*[15];
  for (i=0;i<15;i++){
    EredNvals[i]=new double[Ntheta*Nc];
  }
  compute_EredN(thetac,fvals,invPvals,Pvals,DFKvals,LambdaSvals,LambdaUvals,EredNvals);
  filename="EredNvals";
  write_data_in_file(thetac[0],thetac[1],EredNvals,15,filename);
  for (i=0;i<14;i++){
    delete[] EredNvals[i];
  }
  delete[] EredNvals;
  //--------------
  */

  for (i=0;i<25;i++){
    delete[] Lambdavals[i];
    delete[] DFKvals[i];
  }
  delete[] Lambdavals;
  delete[] DFKvals;
}

void ini_functions(double **thetac,double **LambdaLvals, double **LambdaSvals, double *LambdaUvals, double **Kvals, double **Pvals, double **invPvals,double **fvals){

  int i,j,l,m;
  #pragma omp parallel for private(j,l,m) 
  for (i=0;i<Ntheta;i++){
  gsl_matrix *Lambda=gsl_matrix_alloc(5,5);
  gsl_matrix *P0=gsl_matrix_alloc(5,5);
  gsl_matrix *invP0=gsl_matrix_alloc(5,5);
  double *faux=new double[2];
    for (j=0;j<Nc;j++){
      Kvals[0][j*Ntheta+i]=thetac[0][i];
      Kvals[1][j*Ntheta+i]=thetac[1][j];
      Kvals[2][j*Ntheta+i]=0.;
      Kvals[3][j*Ntheta+i]=0.;
      //cout <<thetac[0][i]<<" "<<thetac[1][j]<<endl;
      Lambda0_P0(thetac[0][i],thetac[1][j],Lambda,P0,invP0,faux,&Kvals[4][j*Ntheta+i]);
      /*
      if (faux[0]<0){
	faux[0]=1.+faux[0];
      }
      fvals[0][j*Ntheta+i]=fmod(faux[0],1.);
      */
      fvals[0][j*Ntheta+i]=faux[0];
      fvals[1][j*Ntheta+i]=faux[1];
      for (l=0;l<5;l++){
	for (m=0;m<5;m++){
	  Pvals[l*5+m][j*Ntheta+i]=gsl_matrix_get(P0,l,m);
	  invPvals[l*5+m][j*Ntheta+i]=gsl_matrix_get(invP0,l,m);
	}
      }
      for (l=0;l<2;l++){
	for (m=0;m<2;m++){
	  LambdaLvals[l*2+m][j*Ntheta+i]=gsl_matrix_get(Lambda,l,m);
	  LambdaSvals[l*2+m][j*Ntheta+i]=gsl_matrix_get(Lambda,l+2,m+2);
	}
      }
      LambdaUvals[j*Ntheta+i]=gsl_matrix_get(Lambda,4,4);
    }
  delete[] faux;
  gsl_matrix_free(Lambda);
  gsl_matrix_free(P0);
  gsl_matrix_free(invP0);
  }

}

void Lambda0_P0(double theta,double c,gsl_matrix *Lambdaaux,gsl_matrix *P0out,gsl_matrix *invP0out,double *f,double *wp){
  //Given theta_ij and c_ij this returns de matrices Lambda, P, P^{-1} and K_w for eps=0
  int i,j;
  double *thetax=new double[5];
  double **DF=new double*[5];
  double **P0=new double*[5];
  double **invP0=new double*[5];
  double **Lambda=new double*[5];
  double **aux=new double*[5];
  for (i=0;i<5;i++){
    DF[i]=new double[5];
    P0[i]=new double[5];
    invP0[i]=new double[5];
    aux[i]=new double[5];
    for (j=0;j<5;j++){
      invP0[i][j]=0.;
    }
    invP0[i][i]=1.;
    Lambda[i]=new double[5];
  }
  double tau=theta;
  double *uv=new double[12];
  double *uvaux=new double[12];
  gsl_matrix * P0aux = gsl_matrix_alloc (5, 5);
  gsl_matrix * DF0aux = gsl_matrix_alloc (5, 5);
  gsl_matrix * invP0aux = gsl_matrix_alloc (5, 5);
  //gsl_matrix * Lambdaaux = gsl_matrix_alloc (5, 5);
  gsl_permutation * perm = gsl_permutation_alloc (5);
  int s;
  gsl_vector_complex *eval=gsl_vector_complex_alloc(5);
  gsl_matrix_complex *evec=gsl_matrix_complex_alloc(5,5);
  
  //We first compute P0(tau,c)
  P_0(tau,c,P0aux);
  clean_matrix(P0aux);
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      P0[i][j]=gsl_matrix_get(P0aux,i,j);
    }
  }
  //Final P0 is computed below, after computing eigenvectors of DF.
  /*
  cout <<endl<<"P0: for "<<tau<<" "<<c<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      //cout<<left<<setw(10)<<P0[i][j]<<" ";
      cout<<left<<setw(10)<<gsl_matrix_get(P0aux,i,j)<<" ";
    }
    cout<<endl;
  }
  */


  //We get K(tau,c):
  thetax[0]=tau;
  thetax[1]=c;
  thetax[2]=thetax[3]=0.;//x,y
  //We need uv to compute w^p(u,v)
  tauc2uv(uv,tau,c);
  thetax[4]=omegap_num(uv[0],uv[1]);//w
  *wp=thetax[4];
  
  //We now get F_0(thetax) and DF_0(thetax)
  F_0(thetax,DF);
  f[0]=thetax[0];
  f[1]=thetax[1];
/*
  cout <<endl;
  cout<<"DF_0 for "<<tau<<" "<<c<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      cout<<left<<setw(10)<<DF[i][j]<<" ";
      //cout<<left<<setw(10)<<gsl_matrix_get(DF0aux,i,j)<<" ";
    }
    cout<<endl;
  }
*/

  //This is to compute eigenvalues of DF_0
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
    gsl_matrix_set(DF0aux,i,j,DF[i][j]);
    }
  }
  clean_matrix(DF0aux);
  gsl_eigen_nonsymmv_workspace * w = gsl_eigen_nonsymmv_alloc (5);
  gsl_eigen_nonsymmv (DF0aux, eval, evec, w);
  gsl_eigen_nonsymmv_free (w);
  gsl_eigen_nonsymmv_sort (eval, evec,GSL_EIGEN_SORT_ABS_ASC);
  /*
  cout <<"Vaps:"<<endl;
  for (i = 0; i < 5; i++){
    cout << GSL_REAL(gsl_vector_complex_get(eval,i))<<" ";
  }
  cout <<endl<<"Veps: "<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      cout<<left<<setw(14)<<GSL_REAL(gsl_matrix_complex_get(evec,i,j))<<" ";
    }
    cout<<endl;
  }
  */
  for(i=2;i<5;i++){
    gsl_matrix_set(P0aux,i,2,GSL_REAL(gsl_matrix_complex_get(evec,i,0)));
    gsl_matrix_set(P0aux,i,3,GSL_REAL(gsl_matrix_complex_get(evec,i,1)));
    gsl_matrix_set(P0aux,i,4,GSL_REAL(gsl_matrix_complex_get(evec,i,4)));
  }
  //cout<<endl<<"P0 for "<<tau<<" "<<c<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      P0[i][j]=gsl_matrix_get(P0aux,i,j);
      gsl_matrix_set(P0out,i,j,gsl_matrix_get(P0aux,i,j));
      //cout<<left<<setw(14)<<P0[i][j]<<" ";
    }
    //cout<<endl;
  }
  //We invert P0:
  gsl_linalg_LU_decomp (P0aux, perm, &s);
  gsl_linalg_LU_invert (P0aux, perm, invP0aux);
  clean_matrix(invP0aux);
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      gsl_matrix_set(invP0out,i,j,gsl_matrix_get(invP0aux,i,j));
    }
  }
/*
  cout <<endl<<"P0:"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      //cout<<left<<setw(10)<<P0[i][j]<<" ";
      cout<<left<<setw(10)<<gsl_matrix_get(P0aux,i,j)<<" ";
    }
    cout<<endl;
  }
*/
  //We now compute (P0(f(theta))^-1
  //(thetax[0],thetax[1])=f0(tau,c)
  P_0(thetax[0],thetax[1],P0aux);
  clean_matrix(P0aux);
  for(i=2;i<5;i++){
    gsl_matrix_set(P0aux,i,2,GSL_REAL(gsl_matrix_complex_get(evec,i,0)));
    gsl_matrix_set(P0aux,i,3,GSL_REAL(gsl_matrix_complex_get(evec,i,1)));
    gsl_matrix_set(P0aux,i,4,GSL_REAL(gsl_matrix_complex_get(evec,i,4)));
  }
  clean_matrix(P0aux);
  //We compute the inverse of P0aux:
  gsl_linalg_LU_decomp (P0aux, perm, &s);
  gsl_linalg_LU_invert (P0aux, perm, invP0aux);
  clean_matrix(invP0aux);
  //cout <<endl<<"P0^{-1}:"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
     // cout<<left<<setw(10)<<invP0[i][j]<<" ";
      invP0[i][j]=gsl_matrix_get(invP0aux,i,j);
      //cout<<left<<setw(10)<<gsl_matrix_get(invP0aux,i,j)<<" ";
    }
    //cout<<endl;
  }
  //cout<<endl;

  //We multiply the matrices:
  multMatrix(aux,DF,5,5,P0,5,5);
  /*
  cout<<endl<<"DF*P0:"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      cout<<left<<setw(10)<<aux[i][j]<<" ";
    }
    cout<<endl;
  }
  */
  multMatrix(Lambda,invP0,5,5,aux,5,5);
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      gsl_matrix_set(Lambdaaux,i,j,Lambda[i][j]);
    }
  }
  clean_matrix(Lambdaaux);
  /*
  cout<<endl<<"Lambda:"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      //cout<<left<<setw(10)<<Lambda[i][j]<<" ";
      cout<<left<<setw(10)<<gsl_matrix_get(Lambdaaux,i,j)<<" ";
    }
    cout<<endl;
  }
  */
  
  for (i=0;i<5;i++){
   delete[] DF[i];
   delete[] P0[i];
   delete[] invP0[i];
   delete[] Lambda[i];
   delete[] aux[i];
  }
  delete[] DF;
  delete[] invP0;
  delete[] P0;
  delete[] Lambda;
  delete[] aux;
  delete[] uv;
  delete[] uvaux;
  delete[] thetax;
  gsl_matrix_free(P0aux);
  gsl_matrix_free(invP0aux);
  gsl_matrix_free(DF0aux);
  //gsl_matrix_free(Lambdaaux);
  gsl_matrix_complex_free(evec);
  gsl_vector_complex_free(eval);
  gsl_permutation_free(perm);

}

void multMatrix(double **C,double **A, int fA, int cA, double **B,int fB,int cB){
  int i,j,k;
  if (cA!=fB){
    cout<<"Failed multiplying matrices"<<endl;
    cout<<"Columns A!=Rows B"<<endl;
    exit(1);
  }

  for (i=0;i<fA;i++){
    for (j=0;j<cB;j++){
      C[i][j]=0.;
      for (k=0;k<cA;k++){
	C[i][j]=C[i][j]+A[i][k]*B[k][j];
      }
    }
  }
}

void P_0(double tau, double c, gsl_matrix *P0){

  double *uv=new double[6];
  double *uvw=new double[12];
  double *duvw=new double[12];
  int i,j;
  int ndim=2;
  double alpha,dalpha,aux;
  double t,h;
  alpha=periodporbit(sqrt(2.*c));
  dalpha=dperiodporbit(sqrt(2.*c));

  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      if (i==j){
	gsl_matrix_set(P0,i,j,1.);
      }
      else{
	gsl_matrix_set(P0,i,j,0.);
      }
    }
  }
  //P=(L N)
  //L->5x2
  //N->5x3:
  //Tangent vectors to the stable and unstable manifolds:
  gsl_matrix_set(P0,3,2,1./sqrt(2.));
  gsl_matrix_set(P0,3,3,-1./sqrt(2.));
  gsl_matrix_set(P0,2,3,1.);
  //gsl_matrix_set(P0,2,2,1.);
  //gsl_matrix_set(P0,3,3,1.);
  //gsl_matrix_set(P0,4,4,1.);
  
  //We compute K(tau,c) and DK(tau,c):
  //uv(tau,c) and D_{tau,c}uv:
  tauc2uv(uv,tau,c);
  //We now compute w^p(u,v) and D_{u,v,w}w^p
  ndim=12;
  uvw[0]=uv[0];
  uvw[1]=uv[1];
  //w^p(u,v):
  uvw[2]=omegap_num(uvw[0],uvw[1]);
  for (i=3;i<12;i++){
    uvw[i]=0.;
  }
  uvw[3]=1.;
  uvw[7]=1.;
  uvw[11]=1.;
  vfieldUvar(t,uvw,ndim,duvw);
  //We integrate the system evaluated at the fixed point (u,v,w^p),
  //(periodic orbit of period alpha)
  h=hini;
  t=0.;
  //while (t<2.*pi/omega){
  ini_rk78(ndim);
  while (t<alpha){
    rk78(&t,uvw,&h,tol,hmin,hmax,ndim,vfieldUvar);
  }
  //h=-(t-2*pi/omega);
  h=-(t-alpha);
  rk78(&t,uvw,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar);
  end_rk78(ndim);
  //gsl_matrix_set(P0,4,0,uvw[9]*uv[2]+uvw[10]*uv[4]);
  //gsl_matrix_set(P0,4,1,uvw[9]*uv[3]+uvw[10]*uv[5]);
  //With this we compute D_{uv}w^p using implicit function theorem to
  //the equation:
  //Pi_w(\phi_{uvw}(alpha;u,v,w))-w=0
  //partial w^p/partial tau:
  /*
  cout <<endl<<endl;
  cout<<"uv for "<<tau<<" "<<c<<endl;
  cout<<uv[0]<<" "<<uv[1]<<endl;
  cout<<"uvw:";
  for (i=0;i<12;i++){
    if (i%3==0){
      cout <<endl;
    }
    cout<<left<<setw(10)<<uvw[i]<<" ";
  }
  */
  aux=uvw[9]*uv[2]+uvw[10]*uv[4];
  aux=-aux/(uvw[11]-1.);
  gsl_matrix_set(P0,4,0,aux);
  //partial w^p/partial c:
  aux=uvw[9]*uv[3]+uvw[10]*uv[5]+duvw[2]*dalpha/sqrt(2.*c);
  aux=-aux/(uvw[11]-1.);
  gsl_matrix_set(P0,4,1,aux);

  delete[] uv;
  delete[] uvw;
  delete[] duvw;

}

void D_uv_thetac(double u,double v, double alpha,double dalpha, double c,double *Dthetacuv){
  //Given u and v, this computes the differential DPhi^{-1}, where 
  //Phi^{-1}: u,v  ---> theta,c
  double h,t,theta;
  double Newtol=1e-12;
  double uprev;
  double *duv=new double[6];
  double *duv2=new double[6];
  double *uvaux=new double[6];
  int ndim=6,i,Maxiter=200;

  ini_rk78(ndim);
  t=0.;
  h=-hini;
  uprev=u;
  uvaux[0]=u;
  uvaux[1]=v;
  uvaux[2]=uvaux[5]=1.;
  uvaux[3]=uvaux[4]=0.;
  rk78(&t,uvaux,&h,tol,hmin,hmax,ndim,vfieldUvar2);
  while (!(uvaux[0]*uprev<0 && uvaux[1]>0)){
    uprev=uvaux[0];
    rk78(&t,uvaux,&h,tol,hmin,hmax,ndim,vfieldUvar2);
  }
  i=0;
  while (fabs(uvaux[0])>Newtol && i<Maxiter){
    vfieldUvar2(t,uvaux,ndim,duv);
    h=-uvaux[0]/duv[0];
    rk78(&t,uvaux,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar2);
    i++;
  }
  end_rk78(ndim);
  if (i>=Maxiter){
    cout <<"Newton failed when computing theta in D_uv_thetac"<<endl;
    exit(1);
  }
  theta=fabs(t)/alpha;
  duv[0]=v;
  duv[1]=0.5*beta*u*(1.-u*u);
  Dthetacuv[2]=-duv[1];//\partial c/\partial u
  Dthetacuv[3]=duv[0];//\partial c/\partial v
  uvaux[0]=0;
  uvaux[1]=sqrt(2.*c);
  vfieldUvar2(t,uvaux,ndim,duv2);//field at (0,sqrt(2*c))
  Dthetacuv[0]=-(uvaux[2]+duv2[0]*(-theta*dalpha/sqrt(2.*c))*(-duv[1]));
  Dthetacuv[0]=Dthetacuv[0]/(-duv2[0]*alpha);//\partial \tau/\partial u
  Dthetacuv[1]=-(uvaux[3]+duv2[0]*(-theta*dalpha/sqrt(2.*c))*duv[0]);
  Dthetacuv[1]=Dthetacuv[1]/(-duv2[0]*alpha);//\partial \tau/\partial v

  delete[] duv;
  delete[] duv2;
  delete[] uvaux;
}

void invF_eps(double theta, double c,double x, double y, double w, double *Fthetax){
  //This returns the value of F^{-1} at thetax
  
  double alpha=periodporbit(sqrt(2.*c));
  double *uv=new double[6];
  double *z=new double[5];
  double h,t;
  int ndim=5;

  tauc2uv(uv,theta,c);
  
  h=-hini;
  t=0.;
  z[0]=x;
  z[1]=y;
  z[2]=uv[0];
  z[3]=uv[1];
  z[4]=w;
  //cout <<w;
  ini_rk78(ndim);
  while (t>-2.*pi/omega){
    rk78(&t,z,&h,tol,hmin,hmax,ndim,vfieldpert);
  }
  h=-(t+2.*pi/omega);
  rk78(&t,z,&h,tol,fabs(h),fabs(h),ndim,vfieldpert);
  end_rk78(ndim);

  uv[0]=z[2];
  uv[1]=z[3];
  uv2tauc(uv,&Fthetax[0],&Fthetax[1]);
  //Careful, I think Fthetax[0] may be discontinuous if computed that
  //way....
  Fthetax[2]=z[0];
  Fthetax[3]=z[1];
  Fthetax[4]=z[4];
  //cout<<" "<<z[4]<<endl;

  delete[] uv;
  delete[] z;
}

void F_eps(double theta, double c,double x, double y, double w, double *Fthetax){
  //This returns the value of F at thetax
  
  double alpha=periodporbit(sqrt(2.*c));
  double *uv=new double[6];
  double *z=new double[5];
  double h,t;
  int ndim=5;

  tauc2uv(uv,theta,c);
  
  h=hini;
  t=0.;
  z[0]=x;
  z[1]=y;
  z[2]=uv[0];
  z[3]=uv[1];
  z[4]=w;
  //cout <<w;
  ini_rk78(ndim);
  while (t<2.*pi/omega){
    rk78(&t,z,&h,tol,hmin,hmax,ndim,vfieldpert);
  }
  h=-(t-2.*pi/omega);
  rk78(&t,z,&h,tol,fabs(h),fabs(h),ndim,vfieldpert);
  end_rk78(ndim);

  uv[0]=z[2];
  uv[1]=z[3];
  uv2tauc(uv,&Fthetax[0],&Fthetax[1]);
  //Careful, I think Fthetax[0] may be discontinuous if computed that
  //way....
  Fthetax[2]=z[0];
  Fthetax[3]=z[1];
  Fthetax[4]=z[4];
  //cout<<" "<<z[4]<<endl;

  delete[] uv;
  delete[] z;
}

void F_DF_eps(double theta, double c,double x, double y, double w, double *Fthetax,double *DF){
  //This returns the value of F and DF at thetax
  //This is aparently giving some troubles when computing DF[12-13] and DF[22-23]....
  //In addition, this function could be improved following the style
  //of ini_functions2
  
  double alpha=periodporbit(sqrt(2.*c));
  double *uv=new double[6];
  double *z=new double[30];
  double h,t;
  int i,j;
  double **aux1=new double*[5];
  double **aux2=new double*[5];
  double **aux3=new double*[5];
  for (i=0;i<5;i++){
    aux1[i]=new double[5];
    aux2[i]=new double[5];
    aux3[i]=new double[5];
  }
  int ndim=30;

  tauc2uv(uv,theta,c);
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      aux1[i][j]=0.;
    }
  }
  aux1[0][2]=1.;
  aux1[1][3]=1.;
  aux1[2][0]=uv[2];
  aux1[2][1]=uv[3];
  aux1[3][0]=uv[4];
  aux1[3][1]=uv[5];
  aux1[4][4]=1.;
  
  h=hini;
  t=0.;
  z[0]=x;
  z[1]=y;
  z[2]=uv[0];
  z[3]=uv[1];
  z[4]=w;
  for (i=5;i<30;i++){
    z[i]=0.;
  }
  z[5]=1.;
  z[11]=1.;
  z[17]=1.;
  z[23]=1.;
  z[29]=1.;

  ini_rk78(ndim);
  h=hmin;
  while (t<2.*pi/omega){
    rk78(&t,z,&h,tol,hmin,hmax,ndim,vfieldpertvar);
  }
  h=-(t-2.*pi/omega);
  rk78(&t,z,&h,tol,fabs(h),fabs(h),ndim,vfieldpertvar);
  end_rk78(ndim);

  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      aux3[i][j]=z[5+i*5+j];
    }
  }
  multMatrix(aux2,aux3,5,5,aux1,5,5);

  uv[0]=z[2];
  uv[1]=z[3];
  uv2tauc(uv,&Fthetax[0],&Fthetax[1]);
  //Careful, I think Fthetax[0] may be discontinuous if computed that
  //way....
  Fthetax[2]=z[0];
  Fthetax[3]=z[1];
  Fthetax[4]=z[4];

  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      aux1[i][j]=0;
    }
  }
  aux1[0][2]=uv[2];
  aux1[0][3]=uv[3];
  aux1[1][2]=uv[4];
  aux1[1][3]=uv[5];
  aux1[2][0]=1.;
  aux1[3][1]=1.;
  aux1[4][4]=1.;
  multMatrix(aux3,aux1,5,5,aux2,5,5);
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      DF[5*i+j]=aux3[i][j];
    }
  }

  delete[] uv;
  delete[] z;
  for (i=0;i<5;i++){
    delete[] aux1[i];
    delete[] aux2[i];
    delete[] aux3[i];
  }
  delete[] aux1;
  delete[] aux2;
  delete[] aux3;
}


void F_0(double *thetax,double **DF){
  //Coordinates in thetax:
  //thetax= (tau,c,x,y,w)

  int i,j,Maxiter=200;
  double *z=new double[30];
  double *uv=new double[6];
  double **Duv=new double*[2];
  double **aux1=new double*[5];
  double **aux2=new double*[5];
  for (i=0;i<5;i++){
    aux1[i]=new double [5];
    aux2[i]=new double [5];
  }
  Duv[0]=new double[2];
  Duv[1]=new double[2];
  int ndim=2;
  double alpha,t=0;
  double h=hini;
  double Newtol=1e-12;
  double *duv=new double[6];
  double det;
  double uprev;
  double theta,c;

  //Chage of variables: tau,c->u,v and D_{tau,c}uv
  uv[0]=0;
  uv[1]=sqrt(2.*thetax[1]);
  uv[2]=1.;
  uv[3]=0;
  uv[4]=0;
  uv[5]=1.;
  ndim=2;
  alpha=periodporbit(uv[1]);
  tauc2uv(uv,thetax[0],thetax[1]);
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      aux1[i][j]=0.;
    }
  }
  aux1[0][2]=1.;
  aux1[1][3]=1.;
  aux1[2][0]=uv[2];
  aux1[2][1]=uv[3];
  aux1[3][0]=uv[4];
  aux1[3][1]=uv[5];
  aux1[4][4]=1.;
  /*
  cout<<endl<<"Dphi(tau,c,x,y,w):"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      cout<<left<<setw(12)<<aux1[i][j]<<" ";
    }
    cout<<endl;
  }
  */
  //We now compute the image of the stroboscopic map using the
  //original coordinates
  z[0]=thetax[2];
  z[1]=thetax[3];
  z[2]=uv[0];
  z[3]=uv[1];
  z[4]=thetax[4];
  for (i=5;i<30;i++){
    z[i]=0;
  }
  z[5]=1.;
  z[11]=1.;
  z[17]=1.;
  z[23]=1.;
  z[29]=1.;
  ndim=30;
  t=0;
  h=hini;
  ini_rk78(ndim);
  while (t<2.*pi/omega){
    rk78(&t,z,&h,tol,hmin,hmax,ndim,vfieldunpertvar);
  }
  h=-(t-2.*pi/omega);
  rk78(&t,z,&h,tol,fabs(h),fabs(h),ndim,vfieldunpertvar);
  end_rk78(ndim);

  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      DF[i][j]=z[5+i*5+j];
    }
  }
  /*
  cout<<endl<<"Ds_0(z):"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      cout<<left<<setw(10)<<DF[i][j]<<" ";
    }
    cout<<endl;
  }
  */

  multMatrix(aux2,DF,5,5,aux1,5,5);
  /*
  cout<<endl<<"Ds_0*Dphi:"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      cout<<left<<setw(10)<<aux2[i][j]<<" ";
    }
    cout<<endl;
  }
  */

  //We now change variables x,y,u,v,w -> tau,c,x,y,w
  uv[0]=z[2];
  uv[1]=z[3];
  uv[2]=1.;
  uv[3]=uv[4]=0;
  uv[5]=1.;
  theta=thetax[0];
  c=thetax[1];
  uv2tauc(uv,&thetax[0],&thetax[1]);
  //To make thetax[0] continuous (for future interpolation) we better
  //use what we know about f for eps=0: (we still need the previous
  //line for the differential)
  thetax[0]=theta+2.*pi/(omega*alpha);
  thetax[1]=c;
  //We have now the value of F_0:
  thetax[2]=z[0];
  thetax[3]=z[1];
  thetax[4]=z[4];
  /*
  cout <<endl<<"Value of F_0:"<<endl;
  for (i=0;i<5;i++){
    cout <<thetax[i]<<" ";
  }
  cout <<endl;
  */

  //From uv2tauc we also have the differential of tau-c wr2 u-v

  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      aux1[i][j]=0;
    }
  }
  aux1[0][2]=uv[2];
  aux1[0][3]=uv[3];
  aux1[1][2]=uv[4];
  aux1[1][3]=uv[5];
  aux1[2][0]=1.;
  aux1[3][1]=1.;
  aux1[4][4]=1.;
  /*
  cout<<endl<<"Dphi^{-1}(x,y,u,v,w):"<<endl;
  for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      cout<<left<<setw(10)<<aux1[i][j]<<" ";
    }
    cout<<endl;
  }
  */
  multMatrix(DF,aux1,5,5,aux2,5,5);

  for (i=0;i<5;i++){
   delete[] aux1[i];
   delete[] aux2[i];
  }
  delete[] aux1;
  delete[] aux2;
  delete [] Duv[0];
  delete[] Duv[1];
  delete[] Duv;
  delete [] z;
  delete[] duv;
  delete [] uv;
}

void torus_correction_unstable(double **thetac,double *LambdaUvals,double *zetaUvals, double **fvals, double **Kvals,double **invPvals,double **Evals){
  //This returns the correction of the torus in the normal unstable direction
  //Here we will ned to interpolate the last row of invP, zetaU and K
  int i,j,l,s,maxiter=2000;
  double maxdiff,itertol=1e-8,tmp;
  double *newzetaUvals=new double[Ntheta*Nc];
  double *etaUvals=new double[Ntheta*Nc];
  gsl_interp2d *zetaU=gsl_interp2d_alloc(T,Ntheta,Nc);

  //We set the initial values of zetau at the grid to zero:
  for (i=0;i<Ntheta*Nc;i++){
    newzetaUvals[i]=0.;
  }
  //etaU does not change at each iteration, so we can compute all
  //values outside the loop:
  gsl_interp2d **invP=new gsl_interp2d*[5];
  for (l=0;l<5;l++){
    invP[l]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(invP[l],thetac[0],thetac[1],invPvals[20+l],Ntheta,Nc);
  }
  gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
  gsl_interp_accel *caac=gsl_interp_accel_alloc();
  #pragma omp parallel for private(j,l,tmp)
  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      tmp=0;
      for (l=0;l<5;l++){
	//We interpolate invPvals at f(theta,c)
	tmp=tmp-Evals[l][j*Ntheta+i]*gsl_interp2d_eval_extrap(invP[l],thetac[0],thetac[1],invPvals[20+l],modulo(fvals[0][j*Ntheta+i],1.),fvals[1][j*Ntheta+i],thetaaac,caac);
      }
      etaUvals[j*Ntheta+i]=tmp;
    }
  }
  gsl_interp_accel_free(thetaaac);
  gsl_interp_accel_free(caac);
  for (l=0;l<5;l++){
    gsl_interp2d_free(invP[l]);
  }
  delete[] invP;

  //We start the loop:
  maxdiff=10*itertol;
  s=1;
  while (maxdiff>itertol && s<maxiter){
    #pragma omp parallell for
    for (i=0;i<Ntheta*Nc;i++){
      zetaUvals[i]=newzetaUvals[i];
    }
    gsl_interp2d_init(zetaU,thetac[0],thetac[1],zetaUvals,Ntheta,Nc);
    #pragma omp parallel for private(j)
    for (i=0;i<Ntheta;i++){
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      for (j=0;j<Nc;j++){
	newzetaUvals[j*Ntheta+i]=1./LambdaUvals[j*Ntheta+i]*(gsl_interp2d_eval_extrap(zetaU,thetac[0],thetac[1],zetaUvals,modulo(fvals[0][j*Ntheta+i],1.),fvals[1][j*Ntheta+i],thetaaac,caac)+etaUvals[j*Ntheta+i]);
      }

      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
    }
    maxdiff=0.;
    for (i=0;i<Ntheta*Nc;i++){
      if (fabs(newzetaUvals[0]-zetaUvals[0])>maxdiff){
	maxdiff=fabs(newzetaUvals[0]-zetaUvals[0]);
      }
    }
    //cout <<"Err in zetaU= "<<maxdiff<<endl;
    s++;
  }
  if (s>=maxiter){
    cout<<"  Newton computing zetaU diverges"<<endl;
    cout <<"  Number of iterations done: "<<s<<endl;
    cout <<"  Err in zetaU: "<<maxdiff<<endl;
    exit(1);
  }
  cout <<"  Err in zetaU: "<<maxdiff<<endl;
  cout <<"  Number of iterations needed: "<<s<<endl;

  #pragma omp parallel for
  for (i=0;i<Ntheta*Nc;i++){
    zetaUvals[i]=newzetaUvals[i];
  }
  gsl_interp2d_free(zetaU);
  delete[] newzetaUvals;
  delete[] etaUvals;

}

void torus_correction_stable(double **thetac,double **LambdaSvals,double **zetaSvals, double **invfvals, double **Kvals, double **invPvals,double **Evals, gsl_interp2d **K){
  int i,j,k,l,s,maxiter=10000;;
  double maxdiff,itertol=1e-8;
  double **newzetaSvals=new double*[2];
  gsl_interp2d **zetaS=new gsl_interp2d*[2];
  for (l=0;l<2;l++){
    newzetaSvals[l]=new double[Ntheta*Nc];
    #pragma omp parallel for
    for (k=0;k<Ntheta*Nc;k++){
      newzetaSvals[l][k]=zetaSvals[l][k]=0.;
    }
    zetaS[l]=gsl_interp2d_alloc(T,Ntheta,Nc);
  }
  
  //The values of etaS do not change at each iteration, so we compute
  //them all before the loop. LambdaSinvfvals have been computed
  //outside, because they are also needed when computing QSU
  double **etaSvals=new double*[2];
  etaSvals[0]=new double[Ntheta*Nc];
  etaSvals[1]=new double[Ntheta*Nc];
  //I think it would be better to interpolate Evals at invfvals
  /*
  #pragma omp parallel for private(j,l,k)
  for (i=0;i<Ntheta;i++){
    double E,tmp;
    double *Fthetax=new double[5];
    double *thetax=new double[5];
    double *Ktmp=new double[5];
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (j=0;j<Nc;j++){
      for (l=0;l<5;l++){
	Ktmp[l]=gsl_interp2d_eval_extrap(K[l],thetac[0],thetac[1],Kvals[l],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
      }
      F_eps(Ktmp[0],Ktmp[1],Ktmp[2],Ktmp[3],Ktmp[4],Fthetax);
      for (k=0;k<2;k++){
	etaSvals[k][j*Ntheta+i]=0;
	for (l=0;l<5;l++){
	  E=Fthetax[l]-Kvals[l][j*Ntheta+i];
	  etaSvals[k][j*Ntheta+i]=etaSvals[k][j*Ntheta+i]-E*invPvals[10+k*5+l][j*Ntheta+i];
	}
      }
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
    delete[] Ktmp;
    delete[] Fthetax;
    delete[] thetax;
  }
  */
  //We first interpolate E and LambdaS at invf:
  gsl_interp2d **E=new gsl_interp2d*[5];
  double **Einvfvals=new double*[5];
  #pragma omp parallel for
  for (i=0;i<5;i++){
    Einvfvals[i]=new double[Ntheta*Nc];
    E[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(E[i],thetac[0],thetac[1],Evals[i],Ntheta,Nc);
  }
  gsl_interp2d **LambdaS=new gsl_interp2d*[4];
  double **LambdaSinvfvals=new double*[4];
  #pragma omp parallel for
  for (i=0;i<4;i++){
    LambdaSinvfvals[i]=new double[Ntheta*Nc];
    LambdaS[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(LambdaS[i],thetac[0],thetac[1],LambdaSvals[i],Ntheta,Nc);
  }
  #pragma omp parallel for private(j,k)
  for (i=0;i<Ntheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (j=0;j<Nc;j++){
      for (k=0;k<5;k++){
	Einvfvals[k][j*Ntheta+i]=gsl_interp2d_eval_extrap(E[k],thetac[0],thetac[1],Evals[k],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
      }
      for (k=0;k<4;k++){
	LambdaSinvfvals[k][j*Ntheta+i]=gsl_interp2d_eval_extrap(LambdaS[k],thetac[0],thetac[1],LambdaSvals[k],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
      }
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  clean_data(Einvfvals,5);
  clean_data(LambdaSinvfvals,4);
  //tmp:-----
  /*
  maxiter=25;
  string filename;
  filename="LambdaSinvfvals";
  write_data_in_file(thetac[0],thetac[1],LambdaSinvfvals,4,filename);
  */
  //-----

  /*
  string filename;
  filename="LambdaSinvf";
  write_data_in_file(fvals[0],fvals[1],LambdaSinvfvals,4,filename);
  */
  //We now compute etaS (at invf)
  #pragma omp parallel for private(j,k,l)
  for (i=0;i<Ntheta;i++){
    double tmp;
    for (j=0;j<Nc;j++){
      for (k=0;k<2;k++){
	etaSvals[k][j*Ntheta+i]=0.;
	for (l=0;l<5;l++){
	  etaSvals[k][j*Ntheta+i]+=-invPvals[10+k*5+l][j*Ntheta+i]*Einvfvals[l][j*Ntheta+i];
	}
      }
    }
  }
  //tmp:------
  /*
  filename="etaSvals";
  write_data_in_file(thetac[0],thetac[1],etaSvals,2,filename);
  */
  //-----
  for(i=0;i<5;i++){
    gsl_interp2d_free(E[i]);
    delete[] Einvfvals[i];
  }
  delete[] E;
  delete[] Einvfvals;
  for(i=0;i<4;i++){
    gsl_interp2d_free(LambdaS[i]);
  }
  delete[] LambdaS;

  maxdiff=10*itertol;
  s=0;
  while (maxdiff>itertol && s<maxiter){
    for (i=0;i<2;i++){
      #pragma omp parallel for
      for (j=0;j<Ntheta*Nc;j++){
	zetaSvals[i][j]=newzetaSvals[i][j];
      }
    }
    for (l=0;l<2;l++){
      gsl_interp2d_init(zetaS[l],thetac[0],thetac[1],zetaSvals[0],Ntheta,Nc);
    }
    #pragma omp parallel for private(j,l,k)
    for (i=0;i<Ntheta;i++){
      double tmp;
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      for (j=0;j<Nc;j++){
	for (k=0;k<2;k++){
	  tmp=gsl_interp2d_eval_extrap(zetaS[0],thetac[0],thetac[1],zetaSvals[0],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
	  newzetaSvals[k][j*Ntheta+i]=LambdaSinvfvals[2*k][j*Ntheta+i]*tmp;
	  tmp=gsl_interp2d_eval_extrap(zetaS[1],thetac[0],thetac[1],zetaSvals[1],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
	  newzetaSvals[k][j*Ntheta+i]=newzetaSvals[k][j*Ntheta+i]+LambdaSinvfvals[2*k+1][j*Ntheta+i]*tmp-etaSvals[k][j*Ntheta+i];
	  //cout <<"DetLambdaSinvf at "<<thetac[0][i]<<" "<<thetac[1][j]<<" = "<<LambdaSinvfvals[0][j*Ntheta+i]*LambdaSinvfvals[3][j*Ntheta+i]-LambdaSinvfvals[1][j*Ntheta+i]*LambdaSinvfvals[2][j*Ntheta+i]<<endl;
	}
      }
      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
    }
    //tmp:----
    /*
    filename="newzetaSvals";
    std:ostringstream stmp;
    stmp<<filename.c_str()<<"_"<<s;
    filename=stmp.str();
    write_data_in_file(thetac[0],thetac[1],newzetaSvals,2,filename);
    */
    //----
    maxdiff=0.;
    for (i=0;i<2;i++){
      for (j=0;j<Ntheta*Nc;j++){
	if (fabs(newzetaSvals[i][j]-zetaSvals[i][j])>maxdiff){
	  maxdiff=fabs(newzetaSvals[i][j]-zetaSvals[i][j]);
	}
      }
    }
    //cout <<"Err in zetaS: "<<maxdiff<<endl;
    s++;
  }
  if (s>=maxiter){
    cout<<"  Newton computing zetaS diverges"<<endl;
    cout <<"  Err in zetaS: "<<maxdiff<<endl;
    cout <<"  Number of iterations done: "<<s<<endl;
    exit(1);
  }
  cout <<"  Err in zetaS: "<<maxdiff<<endl;
  cout <<"  Number of iterations needed: "<<s<<endl;
  for (i=0;i<2;i++){
    #pragma omp parallel for private(j)
    for (j=0;j<Ntheta*Nc;j++){
      zetaSvals[i][j]=newzetaSvals[i][j];
    }
  }
  for (i=0;i<4;i++){
    delete[] LambdaSinvfvals[i];
  }
  delete[] LambdaSinvfvals;
  gsl_interp2d_free(zetaS[0]);
  gsl_interp2d_free(zetaS[1]);
  delete[] zetaS;
  delete[] newzetaSvals[0];
  delete[] newzetaSvals[1];
  delete[] newzetaSvals;
  delete[] etaSvals[0];
  delete[] etaSvals[1];
  delete[] etaSvals;
}

void innerdynamics_correction(double **thetac,double **Deltafvals,double **fvals,double **invPvals, double **Evals){
  int i,j,k,l;
  double tmp,tmp2;

  gsl_interp2d **invP=new gsl_interp2d*[10];
  for (k=0;k<10;k++){
    invP[k]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(invP[k],thetac[0],thetac[1],invPvals[k],Ntheta,Nc);
  }

  #pragma omp parallel for private (j,k,l,tmp,tmp2)
  for (i=0;i<Ntheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (j=0;j<Nc;j++){
      for (k=0;k<2;k++){
	tmp2=0.;
	for (l=0;l<5;l++){
	  tmp=gsl_interp2d_eval_extrap(invP[5*k+l],thetac[0],thetac[1],invPvals[5*k+l],modulo(fvals[0][j*Ntheta+i],1.),fvals[1][j*Ntheta+i],thetaaac,caac);
	  tmp2=tmp2-tmp*Evals[l][j*Ntheta+i];
	}
	Deltafvals[k][j*Ntheta+i]=-tmp2;
      }
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  for (k=0;k<10;k++){
    gsl_interp2d_free(invP[k]);
  }
  delete [] invP;
}

void compute_invf(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals){
  //Here we compute the inverse of f using Newton method
  double Newtol=1e-12;
  double det;
  int i,j,l,k;
  int maxiter=100;
  int xextragrid=5;
  int yextragrid=5;
  double maxErr;
  double F2,F1;

  #pragma omp parallel for private(j,l,det,maxErr,F2,F1,k)
  for (i=0;i<Ntheta;i++){
    double *df=new double[4];
    double *invdf=new double[4];
    double *nextinvf=new double[2];
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    //Extra grid in case we need to compute derivatives outside the
    //original grid
    double **extrathetac=new double*[2];
    extrathetac[0]=new double[xextragrid];
    extrathetac[1]=new double[yextragrid];
    double **extrafvals=new double*[2];
    extrafvals[0]=new double[xextragrid*yextragrid];
    extrafvals[1]=new double[xextragrid*yextragrid];
    for (j=0;j<Nc;j++){
      //invfvals should contain proper initial sead for the Newton.
      maxErr=10*Newtol;
      nextinvf[0]=modulo(invfvals[0][j*Ntheta+i],1.);
      nextinvf[1]=invfvals[1][j*Ntheta+i];
      l=0;
      while (fabs(maxErr)>Newtol && l<maxiter){
	invfvals[0][j*Ntheta+i]=modulo(nextinvf[0],1.);
	invfvals[1][j*Ntheta+i]=nextinvf[1];
	if (invfvals[1][j*Ntheta+i]< thetac[1][0] || invfvals[1][j*Ntheta+i]> thetac[1][Nc-1]){
	  gsl_interp_accel *thetaaac2= gsl_interp_accel_alloc();
	  gsl_interp_accel *caac2=gsl_interp_accel_alloc();
	  gsl_interp2d **extraf=new gsl_interp2d*[2];
	  extraf[0]=gsl_interp2d_alloc(T,xextragrid,yextragrid);
	  extraf[1]=gsl_interp2d_alloc(T,xextragrid,yextragrid);
	  //We need to extend the grid:
	  //cout <<"Out of grid:"<<endl;
	  extendgrid(thetac,fvals,f,invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],extrathetac,extrafvals,extraf,xextragrid,yextragrid);
	  gsl_interp2d_init(extraf[0],extrathetac[0],extrathetac[1],extrafvals[0],xextragrid,yextragrid);
	  gsl_interp2d_init(extraf[1],extrathetac[0],extrathetac[1],extrafvals[1],xextragrid,yextragrid);
	  df[0]=gsl_interp2d_eval_deriv_x(extraf[0],extrathetac[0],extrathetac[1],extrafvals[0],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac2,caac2);
	  df[1]=gsl_interp2d_eval_deriv_y(extraf[0],extrathetac[0],extrathetac[1],extrafvals[0],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac2,caac2);
	  df[2]=gsl_interp2d_eval_deriv_x(extraf[1],extrathetac[0],extrathetac[1],extrafvals[1],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac2,caac2);
	  df[3]=gsl_interp2d_eval_deriv_y(extraf[1],extrathetac[0],extrathetac[1],extrafvals[1],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac2,caac2);
	  gsl_interp2d_free(extraf[0]);
	  gsl_interp2d_free(extraf[1]);
	  delete[] extraf;
	  gsl_interp_accel_free(thetaaac2);
	  gsl_interp_accel_free(caac2);
	 }
	 else{
	  df[0]=gsl_interp2d_eval_deriv_x(f[0],thetac[0],thetac[1],fvals[0],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac,caac);
	  df[1]=gsl_interp2d_eval_deriv_y(f[0],thetac[0],thetac[1],fvals[0],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac,caac);
	  df[2]=gsl_interp2d_eval_deriv_x(f[1],thetac[0],thetac[1],fvals[1],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac,caac);
	  df[3]=gsl_interp2d_eval_deriv_y(f[1],thetac[0],thetac[1],fvals[1],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac,caac);
	 }
	det=df[0]*df[3]-df[1]*df[2];
	invdf[0]=df[3]/det;
	invdf[3]=df[0]/det;
	invdf[1]=-df[1]/det;
	invdf[2]=-df[2]/det;
	//F1=gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],modulo(nextinvf[0],1.),nextinvf[1],thetaaac,caac);
	F1=gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac,caac);
	F1+=-thetac[0][i];
	//F2=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],modulo(nextinvf[0],1.),nextinvf[1],thetaaac,caac);
	F2=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],invfvals[0][j*Ntheta+i],invfvals[1][j*Ntheta+i],thetaaac,caac);
	F2+=-thetac[1][j];
	if (F1>0.5){
	  F1=F1-1.;
	}
	if (F1<-0.5){
	  F1=F1+1.;
	}
	//}
	nextinvf[0]=invfvals[0][j*Ntheta+i]-(invdf[0]*F1+invdf[1]*F2);
	nextinvf[0]=modulo(nextinvf[0],1.);
	nextinvf[1]=invfvals[1][j*Ntheta+i]-(invdf[2]*F1+invdf[3]*F2);
	maxErr=fabs(F2);
	if (fabs(F1)>maxErr){
	  maxErr=fabs(F1);
	}
	l++;
	//cout <<"Error in current step computing inverse: "<<maxErr<<endl;
      }
      //cout <<"Err= "<<maxErr<<endl;
      //cout<<"Numit= "<<l<<endl<<endl;
      if (l>=maxiter && fabs(maxErr)>Newtol){
	cout<<"  Newton computing inverse diverges"<<endl;
	cout <<"  Error in inverse: "<<maxErr<<endl;
	exit(1);
      }
    }
    delete[] invdf;
    delete[] df;
    delete[] nextinvf;
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
    for (k=0;k<2;k++){
      delete[] extrathetac[k];
      delete[] extrafvals[k];
    }
    delete[] extrathetac;
    delete[] extrafvals;
  }
}

double modulo(double a, double b){
  double tmp,tmp2;
  tmp=a;
  while (tmp<=0){
    tmp+=b;
  }
  tmp2=fmod(tmp,b);
  if (fabs(tmp2-b)<1e-12){
    tmp2=0.;
    //cout <<"oju"<<endl;
  }

  return tmp2;
}
/*
double modulo(double a, double b){
  double tmp;
  double Tol=1e-12;
  tmp=fmod(a,b);
  if (a<0){
    tmp=b+tmp;
  }
  if (fabs(tmp-b)<Tol || fabs(tmp)<Tol){
    tmp=0.;
  }

  return tmp;
}
*/

void lift_data(double *data){
  //We assume that the lift should be increasing.
  int i,j,count,intini;
  double tol=0.5;
  for (j=0;j<Nc;j++){
    count=0;
    //count=1;
    data[j*Ntheta]+=double(count);
    //intini=
    for (i=1;i<Ntheta;i++){
      if (fabs(data[j*Ntheta+i]+double(count)-data[j*Ntheta+i-1])>tol){
	count++;
      }
      data[j*Ntheta+i]=data[j*Ntheta+i]+double(count);
    }
  }
}

void compute_E(double **thetac,double **Evals,double **fvals,double **Kvals, gsl_interp2d **K, double **FKvals){
  int i,j,k;
  double tmp;
  double **Kfvals=new double*[5];
  for (i=0;i<5;i++){
    Kfvals[i]=new double[Ntheta*Nc];
  }
  ofstream fKfvals;
  //fKfvals.open("Kfvals.tna");
  #pragma omp parallel for private(j,k,tmp)
  for (i=0;i<Ntheta;i++){
    double *Fthetax=new double[5];
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (j=0;j<Nc;j++){
      fKfvals<<thetac[0][i]<<" "<<thetac[1][j]<<" ";
      for (k=0;k<5;k++){
	Evals[k][j*Ntheta+i]=FKvals[k][j*Ntheta+i];
	Kfvals[k][j*Ntheta+i]=gsl_interp2d_eval_extrap(K[k],thetac[0],thetac[1],Kvals[k],modulo(fvals[0][j*Ntheta+i],1.),fvals[1][j*Ntheta+i],thetaaac,caac);
	//tmp=gsl_interp2d_eval_extrap(K[k],thetac[0],thetac[1],Kvals[k],fvals[0][j*Ntheta+i],fvals[1][j*Ntheta+i],thetaaac,caac);
	//Evals[k][j*Ntheta+i]+=-tmp;
	//fKfvals<<Kfvals[k][j*Ntheta+i]<<" ";
      }
      //Evals[0][j*Ntheta+i]=modulo(Evals[0][j*Ntheta+i],1.);
      //fKfvals<<endl;
    }
    delete[] Fthetax;
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  lift_data(Kfvals[0]);
  #pragma omp parallel for private(k)
  for (i=0;i<Ntheta*Nc;i++){
    for (k=0;k<5;k++){
      Evals[k][i]+=-Kfvals[k][i];
    }
  }
  string filename="Kfvals";
  write_data_in_file(thetac[0],thetac[1],Kfvals,5,filename);
  for (i=0;i<5;i++){
    delete[] Kfvals[i];
  }
  delete[] Kfvals;
  //fKfvals.close();
}

void QLS_correction(double **thetac,double **LambdaLvals,double **LambdaSvals,double **fvals,double **EredNvals,double **QLSvals){
  int i,j,k,maxiter=5000,s,l;
  double **newQLSvals=new double*[4];
  double maxdiff,itertol=1e-8;
  for (i=0;i<4;i++){
    newQLSvals[i]=new double[Ntheta*Nc];
  }
  gsl_interp2d **QLS=new gsl_interp2d*[4];
  for (i=0;i<4;i++){
    QLS[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    #pragma omp parallel for
    for (j=0;j<Ntheta*Nc;j++){
      newQLSvals[i][j]=0.;
    }
  }
  
  maxdiff=10*itertol;
  s=0;
  while (maxdiff>itertol && s<maxiter){
    #pragma omp parallel for private(j)
    for (i=0;i<Ntheta*Nc;i++){
      for (j=0;j<4;j++){
	QLSvals[j][i]=newQLSvals[j][i];
      }
    }
    #pragma omp parallel for 
    for (i=0;i<4;i++){
      gsl_interp2d_init(QLS[i],thetac[0],thetac[1],QLSvals[i],Ntheta,Nc);
    }

    #pragma omp parallel for private(j,k,l)
    for (i=0;i<Ntheta;i++){
      double *QLSf=new double[4];
      double *tmp=new double[4];
      double det;
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();

      for (j=0;j<Nc;j++){
	for (k=0;k<4;k++){
	  QLSf[k]=gsl_interp2d_eval_extrap(QLS[k],thetac[0],thetac[1],QLSvals[k],modulo(fvals[0][j*Ntheta+i],1.),fvals[1][j*Ntheta+i],thetaaac,caac);
	}
	for (k=0;k<4;k++){
	  tmp[k]=0;
	  for (l=0;l<2;l++){
	    tmp[k]+=QLSf[(k/2)*2+l]*LambdaSvals[(k%2)+2*l][j*Ntheta+i];
	  }
	  //tmp[k]+=-EredNvals[k][j*Ntheta+i];
	  tmp[k]+=-EredNvals[(k/2)*3 +k%2][j*Ntheta+i];
	}
	det=LambdaLvals[0][j*Ntheta+i]*LambdaLvals[3][j*Ntheta+i]-LambdaLvals[1][j*Ntheta+i]*LambdaLvals[2][j*Ntheta+i];
	newQLSvals[0][j*Ntheta+i]=LambdaLvals[3][j*Ntheta+i]*tmp[0];
	newQLSvals[0][j*Ntheta+i]+=-LambdaLvals[1][j*Ntheta+i]*tmp[2];
	newQLSvals[1][j*Ntheta+i]=LambdaLvals[3][j*Ntheta+i]*tmp[1];
	newQLSvals[1][j*Ntheta+i]+=-LambdaLvals[1][j*Ntheta+i]*tmp[3];
	newQLSvals[2][j*Ntheta+i]=-LambdaLvals[2][j*Ntheta+i]*tmp[0];
	newQLSvals[2][j*Ntheta+i]+=LambdaLvals[0][j*Ntheta+i]*tmp[2];
	newQLSvals[3][j*Ntheta+i]=-LambdaLvals[2][j*Ntheta+i]*tmp[1];
	newQLSvals[3][j*Ntheta+i]+=LambdaLvals[0][j*Ntheta+i]*tmp[3];

	for (k=0;k<4;k++){
	  newQLSvals[k][j*Ntheta+i]=newQLSvals[k][j*Ntheta+i]/det;
	}
      }
      delete[] QLSf;
      delete[] tmp;
      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
    }
    maxdiff=0.;
    for (i=0;i<4;i++){
      for (j=0;j<Ntheta*Nc;j++){
	if (fabs(newQLSvals[i][j]-QLSvals[i][j])>maxdiff){
	  maxdiff=fabs(newQLSvals[i][j]-QLSvals[i][j]);
	}
      }
    }
    s++;
    //cout <<maxdiff<<endl;
  }
  if (s>=maxiter){
    cout<<"  Newton computing QLS diverges:"<<endl;
  cout<<"  Error in QLS: "<<maxdiff<<endl;
  cout<<"  Number of iterations done: "<<s<<endl;
    exit(1);
  }
  cout<<"  Error in QLS: "<<maxdiff<<endl;
  cout<<"  Number of iterations needed: "<<s<<endl;

  for (i=0;i<4;i++){
    delete[] newQLSvals[i];
    gsl_interp2d_free(QLS[i]);
  }
  delete[] newQLSvals;
  delete[] QLS;
}

void QLU_correction(double **thetac,double **LambdaLinvfvals, double *LambdaUinvfvals, double **invfvals, double **EredNvals, double **QLUvals){
  int i,j,k,s,l;
  int maxiter=5000;
  double maxdiff,tmp;
  double itertol=1e-8;
  double **newQLUvals=new double*[2];
  newQLUvals[0]= new double[Ntheta*Nc];
  newQLUvals[1]= new double[Ntheta*Nc];

  //EredLU at invfvals (LambdaU and LambdaL have
  //been interpolated outside):
  gsl_interp2d **EredLU=new gsl_interp2d*[2];
  double **EredLUinvfvals=new double*[2];
  #pragma omp paralell for
  for (i=0;i<2;i++){
    EredLU[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    EredLUinvfvals[i]=new double[Ntheta*Nc];
    gsl_interp2d_init(EredLU[i],thetac[0],thetac[1],EredNvals[i*3+2],Ntheta,Nc);
  }

  #pragma omp paralell for private(j,k)
  for (i=0;i<Ntheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (j=0;j<Nc;j++){
      for (k=0;k<2;k++){
	EredLUinvfvals[k][j*Ntheta+i]=gsl_interp2d_eval_extrap(EredLU[k],thetac[0],thetac[1],EredNvals[k*3+2],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
      }
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  gsl_interp2d_free(EredLU[0]);
  gsl_interp2d_free(EredLU[1]);
  delete[] EredLU;

  gsl_interp2d **QLU=new gsl_interp2d*[2];
  for (i=0;i<2;i++){
    QLU[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    #pragma omp parallel for
    for (j=0;j<Ntheta*Nc;j++){
      newQLUvals[i][j]=0.;
    }
  }

  maxdiff=10*itertol;
  s=0;
  while (maxdiff>itertol && s<maxiter){
    #pragma omp parallel for
    for (i=0;i<Ntheta*Nc;i++){
      QLUvals[0][i]=newQLUvals[0][i];
      QLUvals[1][i]=newQLUvals[1][i];
    }
    #pragma omp parallel for
    for (i=0;i<2;i++){
      gsl_interp2d_init(QLU[i],thetac[0],thetac[1],QLUvals[i],Ntheta,Nc);
    }
    #pragma omp parallel for private(j,k,l)
    for (i=0;i<Ntheta;i++){
      double *QLUinvf=new double[2];
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      for (j=0;j<Nc;j++){
	for (k=0;k<2;k++){
	  QLUinvf[k]=gsl_interp2d_eval_extrap(QLU[k],thetac[0],thetac[1],QLUvals[k],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
	}
	for (k=0;k<2;k++){
	  newQLUvals[k][j*Ntheta+i]=0;
	  for (l=0;l<2;l++){
	    newQLUvals[k][j*Ntheta+i]+=LambdaLinvfvals[k*2+l][j*Ntheta+i]*QLUinvf[l];
	  }
	  newQLUvals[k][j*Ntheta+i]+=EredLUinvfvals[k][j*Ntheta+i];
	  newQLUvals[k][j*Ntheta+i]=newQLUvals[k][j*Ntheta+i]/LambdaUinvfvals[j*Ntheta+i];
	}
      }
      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
      delete[] QLUinvf;
    }
    maxdiff=fabs(newQLUvals[0][0]-QLUvals[0][0]);
    for (i=0;i<2;i++){
      for (j=0;j<Ntheta*Nc;j++){
	tmp=fabs(newQLUvals[i][j]-QLUvals[i][j]);
	if (tmp>maxdiff){
	  maxdiff=tmp;
	}
      }
    }
    s++;
    //cout <<maxdiff<<endl; 
  }
  if (s>=maxiter){
    cout<<"  Newton computing QLU diverges"<<endl;
    cout<<"  Error in QLU: "<<maxdiff<<endl;
  cout<<"  Number of iterations done: "<<s<<endl;
    exit(1);
  }
  cout<<"  Error in QLU: "<<maxdiff<<endl;
  cout<<"  Number of iterations needed: "<<s<<endl;

  for (i=0;i<2;i++){
    gsl_interp2d_free(QLU[i]);
    delete[] newQLUvals[i];
    delete[] EredLUinvfvals[i];
  }
  delete[] QLU;
  delete[] newQLUvals;
  delete[] EredLUinvfvals;
}

void QUS_correction(double **thetac, double **LambdaSvals, double *LambdaUvals,double **fvals,double **EredNvals, double **QUSvals){
  int i,j,k,maxiter=5000,s,l;
  double maxdiff,itertol=1e-8,tmp;
  double **newQUSvals=new double*[2];
  gsl_interp2d **QUS=new gsl_interp2d*[2];
  for (i=0;i<2;i++){
    QUS[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    newQUSvals[i]=new double[Ntheta*Nc];
    #pragma omp parallel for
    for (j=0;j<Ntheta*Nc;j++){
      newQUSvals[i][j]=0.;
    }
  }
  maxdiff=10*itertol;
  s=0;
  while (maxdiff>itertol && s<maxiter){
    #pragma omp parallel for
    for (i=0;i<Ntheta*Nc;i++){
      QUSvals[0][i]=newQUSvals[0][i];
      QUSvals[1][i]=newQUSvals[1][i];
    }
    #pragma omp parallel for
    for (i=0;i<2;i++){
      gsl_interp2d_init(QUS[i],thetac[0],thetac[1],QUSvals[i],Ntheta,Nc);
    }
    #pragma om parallel for private (j,k,l)
    for (i=0;i<Ntheta;i++){
      double *QUSf=new double[2];
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      for (j=0;j<Nc;j++){
	for (k=0;k<2;k++){
	  QUSf[k]=gsl_interp2d_eval_extrap(QUS[k],thetac[0],thetac[1],QUSvals[k],modulo(fvals[0][j*Ntheta+i],1.),fvals[1][j*Ntheta+i],thetaaac,caac);
	}
	for (k=0;k<2;k++){
	  newQUSvals[k][j*Ntheta+i]=0.;
	  for (l=0;l<2;l++){
	    newQUSvals[k][j*Ntheta+i]+=QUSf[l]*LambdaSvals[k+l*2][j*Ntheta+i];
	  }
	  newQUSvals[k][j*Ntheta+i]+=-EredNvals[12+k][j*Ntheta+i];
	  newQUSvals[k][j*Ntheta+i]=newQUSvals[k][j*Ntheta+i]/LambdaUvals[j*Ntheta+i];
	}
      }
      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
      delete[] QUSf;
    }
    maxdiff=0;
    for (i=0;i<2;i++){
      for (j=0;j<Ntheta*Nc;j++){
	tmp=fabs(newQUSvals[i][j]-QUSvals[i][j]);
	if (tmp>maxdiff){
	  maxdiff=tmp;
	}
      }
    }
    //cout <<maxdiff<<endl;
    s++;
  }
  if (s>=maxiter){
    cout<<"  Newton computing QUS diverges"<<endl;
    cout<<"  Error in QUS: "<<maxdiff<<endl;
  cout<<"  Number of iterations done: "<<s<<endl;
    exit(1);
  }
  cout<<"  Error in QUS: "<<maxdiff<<endl;
  cout<<"  Number of iterations needed: "<<s<<endl;
 
  for (i=0;i<2;i++){
    delete[] newQUSvals[i];
    gsl_interp2d_free(QUS[i]);
  }
  delete[] newQUSvals;
  delete[] QUS;

}

void QSU_correction(double **thetac, double **LambdaSinvfvals,double *LambdaUinvfvals,double **invfvals,double **EredNvals,double **QSUvals){
  int i,j,k,maxiter=5000,s,l;
  double maxdiff,itertol=1e-8,tmp;
  double **newQSUvals=new double*[2];
  gsl_interp2d **QSU=new gsl_interp2d*[2];
  for (i=0;i<2;i++){
    QSU[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    newQSUvals[i]=new double[Ntheta*Nc];
    #pragma omp parallel for
    for (j=0;j<Ntheta*Nc;j++){
      newQSUvals[i][j]=0.;
    }
  }

  maxdiff=10*itertol;
  s=0;
  while (maxdiff>itertol && s <maxiter){
    #pragma omp parallel for
    for (i=0;i<Ntheta*Nc;i++){
      QSUvals[0][i]=newQSUvals[0][i];
      QSUvals[1][i]=newQSUvals[1][i];
    }
    #pragma omp parallel for
    for (i=0;i<2;i++){
      gsl_interp2d_init(QSU[i],thetac[0],thetac[1],QSUvals[i],Ntheta,Nc);
    }
    #pragma omp parallel for private(j,k,l)
    for (i=0;i<Ntheta;i++){
      double *QSUinvf=new double[2];
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      for (j=0;j<Nc;j++){
	for (k=0;k<2;k++){
	  QSUinvf[k]=gsl_interp2d_eval_extrap(QSU[k],thetac[0],thetac[1],QSUvals[k],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
	}
	for (k=0;k<2;k++){
	  newQSUvals[k][j*Ntheta+i]=0.;
	  for (l=0;l<2;l++){
	    newQSUvals[k][j*Ntheta+i]+=LambdaSinvfvals[2*k+l][j*Ntheta+i]*QSUinvf[l];
	  }
	  newQSUvals[k][j*Ntheta+i]+=EredNvals[8+3*k][j*Ntheta+i];
	  newQSUvals[k][j*Ntheta+i]=newQSUvals[k][j*Ntheta+i]/LambdaUinvfvals[j*Ntheta+i];
	}
      }

      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
      delete[] QSUinvf;
    }
    maxdiff=0;
    for (i=0;i<2;i++){
      for (j=0;j<Ntheta*Nc;j++){
	tmp=fabs(newQSUvals[i][j]-QSUvals[i][j]);
	if (tmp>maxdiff){
	  maxdiff=tmp;
	}
      }
    }
    //cout <<maxdiff<<endl;
    s++;
  }
  if (s>=maxiter){
    cout<<"  Newton computing QSU diverges"<<endl;
    cout<<"  Error in QSU: "<<maxdiff<<endl;
  cout<<"  Number of iterations done: "<<s<<endl;
    exit(1);
  }
  cout<<"  Error in QSU: "<<maxdiff<<endl;
  cout<<"  Number of iterations needed: "<<s<<endl;


  for (i=0;i<2;i++){
    gsl_interp2d_free(QSU[i]);
    delete[] newQSUvals[i];
  }
  delete[] QSU;
  delete[] newQSUvals;

}

void compute_EredN(double **thetac,double **fvals,double **invPvals,double **Pvals,double **DFKvals,double **LambdaSvals,double *LambdaUvals,double **EredNvals){
  int i,j,k;
  
  //Pvals have been partially corrected (only L), so we interpolate again
  //(ToDo: maybe this is not necessary to compute all its components again)
  //We take DF at Kvals from compute_FK_DFK.
  //We assume that P(theta)^{-1} has been also updated and computed outside
  gsl_interp2d **invP=new gsl_interp2d*[25];
  #pragma omp parallel for
  for (i=0;i<25;i++){
    invP[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(invP[i],thetac[0],thetac[1],invPvals[i],Ntheta,Nc);
  }

  //This is temporary:
  double **invPftmp=new double*[25];
  for (i=0;i<25;i++){
    invPftmp[i]=new double[Ntheta*Nc];
  }
  
  #pragma omp parallel for private(j,k)
  for (i=0;i<Ntheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    double *tmp2=new double[25];
    double *tmp=new double[15];
    double *N=new double[15];
    double *Eredtmp=new double[15];
    double *invPtmp=new double[25];
    for (j=0;j<Nc;j++){
      for (k=0;k<15;k++){
	N[k]=Pvals[2+(k/3)*5+k%3][j*Ntheta+i];
      }
      for (k=0;k<25;k++){
	tmp2[k]=DFKvals[k][j*Ntheta+i];
      }
      multMatrix_array(tmp,tmp2,5,5,N,5,3);
      for (k=0;k<25;k++){
	tmp2[k]=gsl_interp2d_eval_extrap(invP[k],thetac[0],thetac[1],invPvals[k],modulo(fvals[0][j*Ntheta+i],1.),fvals[1][j*Ntheta+i],thetaaac,caac);
      }
      multMatrix_array(Eredtmp,tmp2,5,5,tmp,5,3);
      for (k=0;k<15;k++){
	EredNvals[k][j*Ntheta+i]=Eredtmp[k];
	//For writting puprposes:
	invPftmp[k][j*Ntheta+i]=tmp[k];
      }
      EredNvals[6][j*Ntheta+i]+=-LambdaSvals[0][j*Ntheta+i];
      EredNvals[7][j*Ntheta+i]+=-LambdaSvals[1][j*Ntheta+i];
      EredNvals[9][j*Ntheta+i]+=-LambdaSvals[2][j*Ntheta+i];
      EredNvals[10][j*Ntheta+i]+=-LambdaSvals[3][j*Ntheta+i];
      EredNvals[14][j*Ntheta+i]+=-LambdaUvals[j*Ntheta+i];
    }
    delete[] tmp2;
    delete[] tmp;
    delete[] N;
    delete[] invPtmp;
    delete[] Eredtmp;
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  string filename="invPftmp";
  //write_data_in_file3(fvals,invPftmp,25,filename);
  write_data_in_file(thetac[0],thetac[1],invPftmp,15,filename);
  for (i=0;i<25;i++){
    gsl_interp2d_free(invP[i]);
    delete[] invPftmp[i];
  }
  delete[] invP;
  delete[] invPftmp;
}

void compute_FK_DFK(double **Kvals,double **FKvals, double **DFKvals){
  //This computes the values of F and DF at K evaluated at the grid
  //values.
  //This function should be rewritten following ini_functions2.
  int i,j,k;

  #pragma omp parallel for private(j,k)
  for (i=0;i<Ntheta;i++){
    double *Fthetax=new double[5];
    double *DFthetax=new double[25];
    for (j=0;j<Nc;j++){
      F_DF_eps(modulo(Kvals[0][j*Ntheta+i],1.),Kvals[1][j*Ntheta+i],Kvals[2][j*Ntheta+i],Kvals[3][j*Ntheta+i],Kvals[4][j*Ntheta+i],Fthetax,DFthetax);
      //F_DF_eps(Kvals[0][j*Ntheta+i],Kvals[1][j*Ntheta+i],Kvals[2][j*Ntheta+i],Kvals[3][j*Ntheta+i],Kvals[4][j*Ntheta+i],Fthetax,DFthetax);
      for (k=0;k<5;k++){
	FKvals[k][j*Ntheta+i]=Fthetax[k];
      }
      for (k=0;k<25;k++){
	DFKvals[k][j*Ntheta+i]=DFthetax[k];
      }
    }
    delete[] Fthetax;
    delete[] DFthetax;
  }
  lift_data(FKvals[0]);

}

void compute_FK_eps(double **Kvals,double **FKvals){
  //Here we compute the values of F at K evaluated at the grid values.
  //FKvals i used to compute E. This is indeed used only at the first
  //Newton iteration, as the next ones it is computed together with
  //the differential, DFK, in compute_FK_DFK.
  int i ,j,k;

  #pragma omp parallel for private(j,k)
  for (i=0;i<Ntheta;i++){
    double *Fthetax=new double[5];
    for (j=0;j<Nc;j++){
      F_eps(Kvals[0][j*Ntheta+i],Kvals[1][j*Ntheta+i],Kvals[2][j*Ntheta+i],Kvals[3][j*Ntheta+i],Kvals[4][j*Ntheta+i],Fthetax);
      for (k=0;k<5;k++){
	FKvals[k][j*Ntheta+i]=Fthetax[k];
      }
    }

    delete[] Fthetax;
  }
  lift_data(FKvals[0]);
}

void multMatrix_array(double *C,double *A, int fA, int cA, double *B,int fB,int cB){
  //This is to multiply two matrices in array form
  int i,j,k;

  if (cA!=fB){
    cout<<"Failed multiplying matrices"<<endl;
    cout<<"Columns A!=Rows B"<<endl;
    exit(1);
  }

  for (i=0;i<fA;i++){
    for (j=0;j<cB;j++){
      C[i*cB+j]=0.;
      for (k=0;k<cA;k++){
	C[i*cB+j]+=A[i*cA+k]*B[k*cB+j];
      }
    }
  }

}

void invertPvals2(double **invPvals, double *DeltaP,int k){
  double *tmp=new double[25];
  double *tmp2=new double[25];
  double *aux=new double[25];
  int i;
  //Here we compoute the inverse of P in first order expanding in terms of the correction of P, DeltaP.
  for (i=0;i<25;i++){
    aux[i]=invPvals[i][k];
  }

  multMatrix_array(tmp,DeltaP,5,5,aux,5,5);
  multMatrix_array(tmp2,aux,5,5,tmp,5,5);
  for (i=0;i<25;i++){
    invPvals[i][k]+=-tmp2[i];
  }
  delete[] tmp;
  delete[] tmp2;
  delete[] aux;
}

void invertPvals(double **Pvals,double **invPvals){
  //Here we get all matrix P and invert them by brut force.
  //Another option could to assume that P has been corrected in size
  //eps and compute the corresponding correction for P^{-1} in first
  //order:
  //Delta invP=-invP*DeltaP*invP
  int i,j,k,s;

  #pragma omp parallel for private(j,k,s) 
  for (i=0;i<Ntheta;i++){
    gsl_matrix *Ptmp=gsl_matrix_alloc(5,5);
    gsl_matrix *invPtmp=gsl_matrix_alloc(5,5);
    gsl_permutation * perm = gsl_permutation_alloc (5);
    for (j=0;j<Nc;j++){
      for (k=0;k<25;k++){
	gsl_matrix_set(Ptmp,k/5,k%5,Pvals[k][j*Ntheta+i]);
      }
      gsl_linalg_LU_decomp (Ptmp, perm, &s);
      gsl_linalg_LU_invert (Ptmp, perm, invPtmp);
      clean_matrix(invPtmp);
      for (k=0;k<25;k++){
	invPvals[k][j*Ntheta+i]=gsl_matrix_get(invPtmp,k/5,k%5);
      }
    }
    gsl_matrix_free(Ptmp);
    gsl_matrix_free(invPtmp);
    gsl_permutation_free(perm);
  }
}

void extendgrid(double **thetac,double **fvals,gsl_interp2d **f,double xpoint,double ypoint,double **extrathetac,double **extrafvals,gsl_interp2d **extraf,int xgrid,int ygrid){
  //This provides and new small grid and interpolation function by
  //extrapolating f. We need this in order to compute derivatives
  //outside the original grid.
  int i,j,k;
  size_t index;
  double hx=fabs(thetac[0][1]-thetac[0][0]);
  double hy=fabs(thetac[1][1]-thetac[1][0]);
  for (i=0;i<xgrid;i++){
    extrathetac[0][i]=xpoint+hx*double(i-(xgrid-1)/2);
  }
  for (i=0;i<ygrid;i++){
    extrathetac[1][i]=ypoint+hy*double(i-(xgrid-1)/2);
  }
  gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
  gsl_interp_accel *caac=gsl_interp_accel_alloc();
  for (i=0;i<xgrid;i++){
    for (j=0;j<ygrid;j++){
      extrafvals[0][j*xgrid+i]=gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],extrathetac[0][i],extrathetac[1][j],thetaaac,caac);
      extrafvals[1][j*xgrid+i]=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],extrathetac[0][i],extrathetac[1][j],thetaaac,caac);
    }
  }
  gsl_interp_accel_free(thetaaac);
  gsl_interp_accel_free(caac);
}

void correct_normal_bundle(double **thetac,double **fvals,double **invfvals,double **Kvals, double **DFKvals,double **Pvals, double **invPvals,double **LambdaLvals,double **LambdaSvals, double *LambdaUvals,double **EredNvals){
  int i,j,k,l;
  string filename;

  //We first compute EredN:
  //Looks like there is some problem already in the first computation
  //of this, in the second row of EredSS.
  //I don't think so anymore...
  compute_EredN(thetac,fvals,invPvals,Pvals,DFKvals,LambdaSvals,LambdaUvals,EredNvals);
  clean_data(EredNvals,15);
  filename="EredNvals";
  write_data_in_file(thetac[0],thetac[1],EredNvals,15,filename);
  
  //------------------------------------------------------
  // Computation of the Normal bundle corrections:
  //------------------------------------------------------
  double **QLSvals=new double*[4];
  for (i=0;i<4;i++){
    QLSvals[i]=new double[Ntheta*Nc];
  }
  double **QLUvals=new double*[2];
  double **QSUvals=new double*[2];
  double **QUSvals=new double*[2];
  for (i=0;i<2;i++){
    QLUvals[i]=new double[Ntheta*Nc];
    QSUvals[i]=new double[Ntheta*Nc];
    QUSvals[i]=new double[Ntheta*Nc];
  }

  //We will need to interpolate LambdaU at invf both in QLU_correction and
  //QSU_correction, so we do it only once here outside:
  //I think it is more efficient to compute all this interpolations in
  //a single parallelized loop here. I would also include later on the
  //interpolation of EredLU and EredSU.
  double *LambdaUinvfvals=new double[Ntheta*Nc];
  gsl_interp2d *LambdaU;
  LambdaU=gsl_interp2d_alloc(T,Ntheta,Nc);
  gsl_interp2d_init(LambdaU,thetac[0],thetac[1],LambdaUvals,Ntheta,Nc);
  double **LambdaSinvfvals=new double*[4];
  gsl_interp2d **LambdaS=new gsl_interp2d*[4];
  double **LambdaLinvfvals=new double*[4];
  gsl_interp2d **LambdaL=new gsl_interp2d*[4];
  #pragma omp parallel for
  for (i=0;i<4;i++){
    LambdaS[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(LambdaS[i],thetac[0],thetac[1],LambdaSvals[i],Ntheta,Nc);
    LambdaSinvfvals[i]=new double[Ntheta*Nc];
    LambdaL[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(LambdaL[i],thetac[0],thetac[1],LambdaLvals[i],Ntheta,Nc);
    LambdaLinvfvals[i]=new double[Ntheta*Nc];
  }

  #pragma omp parallel for private(j,k)
  for (i=0;i<Ntheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (j=0;j<Nc;j++){
      LambdaUinvfvals[j*Ntheta+i]=gsl_interp2d_eval_extrap(LambdaU,thetac[0],thetac[1],LambdaUvals,modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
      for (k=0;k<4;k++){
	LambdaSinvfvals[k][j*Ntheta+i]=gsl_interp2d_eval_extrap(LambdaS[k],thetac[0],thetac[1],LambdaSvals[k],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
	LambdaLinvfvals[k][j*Ntheta+i]=gsl_interp2d_eval_extrap(LambdaL[k],thetac[0],thetac[1],LambdaLvals[k],modulo(invfvals[0][j*Ntheta+i],1.),invfvals[1][j*Ntheta+i],thetaaac,caac);
      }
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }

  clean_data2(LambdaUinvfvals);
  clean_data(LambdaSinvfvals,4);
  clean_data(LambdaLinvfvals,4);
  cout<<"  Computing QLS..."<<endl;
  QLS_correction(thetac,LambdaLvals,LambdaSvals,fvals,EredNvals,QLSvals);
  clean_data(QLSvals,4);
  filename="QLSvals";
  write_data_in_file(thetac[0],thetac[1],QLSvals,4,filename);
  cout <<"  Done."<<endl;
  cout <<"  Computing QLU..."<<endl;
  QLU_correction(thetac,LambdaLinvfvals,LambdaUinvfvals,invfvals,EredNvals,QLUvals);
  clean_data(QLUvals,2);
  filename="QLUvals";
  write_data_in_file(thetac[0],thetac[1],QLUvals,2,filename);
  cout <<"  Done."<<endl;
  cout<<"  Computing QUS..."<<endl;
  QUS_correction(thetac,LambdaSvals,LambdaUvals,fvals,EredNvals,QUSvals);
  clean_data(QUSvals,2);
  filename="QUSvals";
  write_data_in_file(thetac[0],thetac[1],QUSvals,2,filename);
  cout <<"  Done."<<endl;
  cout<<"  Computing QSU..."<<endl;
  QSU_correction(thetac,LambdaSinvfvals,LambdaUinvfvals,invfvals,EredNvals,QSUvals);
  clean_data(QSUvals,2);
  filename="QSUvals";
  write_data_in_file(thetac[0],thetac[1],QSUvals,2,filename);
  cout <<"  Done."<<endl;

  //------------------------------------------------------------
  //----------Correction of the normal bundle normal:--------
  //------------------------------------------------------------
  /*
  #pragma omp parallel for private(j,l)
  for (i=0;i<Ntheta;i++){
    double *DeltaN=new double[15];
    double *Q=new double [15];
    double *Ptmp=new double[25];
    for (j=0;j<Nc;j++){
      for (l=0;l<4;l++){
	Q[l/2*3+l%2]=QLSvals[l][j*Ntheta+i];
	Q[6+l/2*3+l%2]=0.;
      }
      for (l=0;l<2;l++){
	Q[2+l*3]=QLUvals[l][j*Ntheta+i];
	Q[8+l*3]=QSUvals[l][j*Ntheta+i];
	Q[12+l]=QUSvals[l][j*Ntheta+i];
      }
      Q[14]=0.;
      for (l=0;l<25;l++){
	Ptmp[l]=Pvals[l][j*Ntheta+i];
      }
      multMatrix_array(DeltaN,Ptmp,5,5,Q,5,3);
      for (l=0;l<15;l++){
	Pvals[2+l/3*5+l%3][j*Ntheta+i]+=DeltaN[l];
      }
      for (l=0;l<4;l++){
	LambdaSvals[l][j*Ntheta+i]+=EredNvals[6+l/2*3+l%2][j*Ntheta+i];
      }
      LambdaUvals[j*Ntheta+i]+=EredNvals[14][j*Ntheta+i];
    }
    delete[] DeltaN;
    delete[] Q;
    delete[] Ptmp;
  }
  invertPvals(Pvals,invPvals);
  */
  //Second version:
  #pragma omp parallel for private(j,l)
  for (i=0;i<Ntheta;i++){
    double *DeltaP=new double[25];
    double *Q=new double [25];
    double *Ptmp=new double[25];
    for (j=0;j<Nc;j++){
      for (l=0;l<25;l++){
	Q[l]=0.;
      }
      for (l=0;l<4;l++){
	Q[2+l/2*5+l%2]=QLSvals[l][j*Ntheta+i];
	Q[8+l/2*5+l%2]=0.;
      }
      for (l=0;l<2;l++){
	Q[4+l*5]=QLUvals[l][j*Ntheta+i];
	Q[14+l*5]=QSUvals[l][j*Ntheta+i];
	Q[22+l]=QUSvals[l][j*Ntheta+i];
      }
      Q[24]=0.;
      for (l=0;l<25;l++){
	Ptmp[l]=Pvals[l][j*Ntheta+i];
      }
      multMatrix_array(DeltaP,Ptmp,5,5,Q,5,5);
      for (l=0;l<15;l++){
	Pvals[2+l/3*5+l%3][j*Ntheta+i]+=DeltaP[2+l/3*5+l%3];
      }
      invertPvals2(invPvals,DeltaP,j*Ntheta+i);
      for (l=0;l<4;l++){
	LambdaSvals[l][j*Ntheta+i]+=EredNvals[6+l/2*3+l%2][j*Ntheta+i];
      }
      LambdaUvals[j*Ntheta+i]+=EredNvals[14][j*Ntheta+i];
    }
    delete[] DeltaP;
    delete[] Q;
    delete[] Ptmp;
  }
  clean_data(Pvals,25);
  clean_data(invPvals,25);
  clean_data(LambdaSvals,4);
  clean_data2(LambdaUvals);


  gsl_interp2d_free(LambdaU);
  delete[] LambdaUinvfvals;
  for (i=0;i<4;i++){
    delete[] QLSvals[i];
    gsl_interp2d_free(LambdaS[i]);
    delete[] LambdaSinvfvals[i];
    gsl_interp2d_free(LambdaL[i]);
    delete[] LambdaLinvfvals[i];
  }
  delete[] QLSvals;
  delete[] LambdaS;
  delete[] LambdaSinvfvals;
  delete[] LambdaLinvfvals;
  delete[] LambdaL;
  for (i=0;i<2;i++){
    delete[] QLUvals[i];
    delete[] QSUvals[i];
    delete[] QUSvals[i];
  }
  delete [] QLUvals;
  delete[] QSUvals;
  delete[] QUSvals;
}

void iterate_inner_dynamics2(double **thetac,double **fvals,gsl_interp2d **f,gsl_interp2d **K,double **Kvals,string filename){
  //Unlike in iterate_inner_dynamics, we take a set of initial conditions in a
  //grid \theta x c and iterate them.
  int i,j,k;

  int numit=200;
  int nptheta=50;
  int npc=50;
  int *finaliter=new int[nptheta*npc];
  double ***iterates=new double**[nptheta*npc];
  for (i=0;i<nptheta*npc;i++){
    iterates[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iterates[i][j]=new double[5];
    }
  }
  gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
  gsl_interp_accel *caac=gsl_interp_accel_alloc();
  #pragma omp parallel for private(j,i)
  for (k=0;k<npc;k++){
    for (i=0;i<nptheta;i++){
      iterates[k*nptheta+i][0][0]=thetasimini+double(i)*(thetasimend-thetasimini)/double(nptheta);
      //iterates[i][0][1]=csimini +double(i)*(csimend-csimini)/double(numpoints-1);
      iterates[k*nptheta+i][0][1]=csimini +double(k)*(csimend-csimini)/double(npc);
      iterates[k*nptheta+i][0][2]=gsl_interp2d_eval_extrap(K[2],thetac[0],thetac[1],Kvals[2],modulo(iterates[k*nptheta+i][0][0],1.),iterates[k*nptheta+i][0][1],thetaaac,caac);
      iterates[k*nptheta+i][0][3]=gsl_interp2d_eval_extrap(K[3],thetac[0],thetac[1],Kvals[3],modulo(iterates[k*nptheta+i][0][0],1.),iterates[k*nptheta+i][0][1],thetaaac,caac);
      iterates[k*nptheta+i][0][4]=gsl_interp2d_eval_extrap(K[4],thetac[0],thetac[1],Kvals[4],modulo(iterates[k*nptheta+i][0][0],1.),iterates[k*nptheta+i][0][1],thetaaac,caac);
      //for (j=1;j<numit;j++){
      j=1;
      while(j<numit && fabs(iterates[k*nptheta+i][j-1][1])<5){
	iterates[k*nptheta+i][j][0]=modulo(gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],modulo(iterates[k*nptheta+i][j-1][0],1.),iterates[k*nptheta+i][j-1][1],thetaaac,caac),1.);
	iterates[k*nptheta+i][j][1]=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],modulo(iterates[k*nptheta+i][j-1][0],1.),iterates[k*nptheta+i][j-1][1],thetaaac,caac);
	iterates[k*nptheta+i][j][2]=gsl_interp2d_eval_extrap(K[2],thetac[0],thetac[1],Kvals[2],modulo(iterates[k*nptheta+i][j][0],1.),iterates[k*nptheta+i][j][1],thetaaac,caac);
	iterates[k*nptheta+i][j][3]=gsl_interp2d_eval_extrap(K[3],thetac[0],thetac[1],Kvals[3],modulo(iterates[k*nptheta+i][j][0],1.),iterates[k*nptheta+i][j][1],thetaaac,caac);
	iterates[k*nptheta+i][j][4]=gsl_interp2d_eval_extrap(K[4],thetac[0],thetac[1],Kvals[4],modulo(iterates[k*nptheta+i][j][0],1.),iterates[k*nptheta+i][j][1],thetaaac,caac);
	j++;
      }
      finaliter[k*nptheta+i]=j-1;
    }
  }
  ofstream fiterates;
  fiterates.open(filename.c_str());
  for (i=0;i<nptheta*npc;i++){
    //for (j=50;j<numit;j++){
    //for (j=50;j<finaliter[i];j++){
    for (j=0;j<finaliter[i];j++){
      fiterates<<i<<" "<<j<<" ";
      for (k=0;k<5;k++){
	fiterates<<iterates[i][j][k]<<" ";
      }
      fiterates<<endl;
    }
  }
  for (i=0;i<nptheta*npc;i++){
    for (j=0;j<numit;j++){
      delete[] iterates[i][j];
    }
    delete[] iterates[i];
  }
  delete[] iterates;
  delete[] finaliter;
  fiterates.close();
  gsl_interp_accel_free(thetaaac);
  gsl_interp_accel_free(caac);

}

void iterate_inner_dynamics(double **thetac,double **fvals,gsl_interp2d **f,gsl_interp2d **K,double **Kvals,string filename){
  int i,j,k;
  //Let's iterate the map:
  int numit=100000;
  int numpoints=50;
  int *finaliter=new int[numpoints];
  double ***iterates=new double**[numpoints];
  for (i=0;i<numpoints;i++){
    iterates[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iterates[i][j]=new double[5];
    }
  }
  gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
  gsl_interp_accel *caac=gsl_interp_accel_alloc();
  #pragma omp parallel for private(j)
  for (i=0;i<numpoints;i++){
    iterates[i][0][0]=0.5;
    //iterates[i][0][1]=csimini +double(i)*(csimend-csimini)/double(numpoints-1);
    //iterates[i][0][0]=thetasimini +double(i)*(thetasimend-thetasimini)/double(numpoints);
    iterates[i][0][1]=csimini +double(i)*(csimend-csimini)/double(numpoints);
    //iterates[i][0][1]=0.3466;
    iterates[i][0][2]=gsl_interp2d_eval_extrap(K[2],thetac[0],thetac[1],Kvals[2],modulo(iterates[i][0][0],1.),iterates[i][0][1],thetaaac,caac);
    iterates[i][0][3]=gsl_interp2d_eval_extrap(K[3],thetac[0],thetac[1],Kvals[3],modulo(iterates[i][0][0],1.),iterates[i][0][1],thetaaac,caac);
    iterates[i][0][4]=gsl_interp2d_eval_extrap(K[4],thetac[0],thetac[1],Kvals[4],modulo(iterates[i][0][0],1.),iterates[i][0][1],thetaaac,caac);
    //for (j=1;j<numit;j++){
    j=1;
    while(j<numit && fabs(iterates[i][j-1][1])<5){
      iterates[i][j][0]=modulo(gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],modulo(iterates[i][j-1][0],1.),iterates[i][j-1][1],thetaaac,caac),1.);
      iterates[i][j][1]=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],modulo(iterates[i][j-1][0],1.),iterates[i][j-1][1],thetaaac,caac);
      iterates[i][j][2]=gsl_interp2d_eval_extrap(K[2],thetac[0],thetac[1],Kvals[2],modulo(iterates[i][j][0],1.),iterates[i][j][1],thetaaac,caac);
      iterates[i][j][3]=gsl_interp2d_eval_extrap(K[3],thetac[0],thetac[1],Kvals[3],modulo(iterates[i][j][0],1.),iterates[i][j][1],thetaaac,caac);
      iterates[i][j][4]=gsl_interp2d_eval_extrap(K[4],thetac[0],thetac[1],Kvals[4],modulo(iterates[i][j][0],1.),iterates[i][j][1],thetaaac,caac);
      j++;
    }
    finaliter[i]=j-1;
  }
  ofstream fiterates;
  fiterates.open(filename.c_str());
  for (i=0;i<numpoints;i++){
    //for (j=50;j<numit;j++){
    //for (j=50;j<finaliter[i];j++){
    for (j=0;j<finaliter[i];j++){
      fiterates<<i<<" "<<j<<" ";
      for (k=0;k<5;k++){
	//fiterates<<iterates[i][j][k]<<" ";
	fiterates<<setprecision(15)<<iterates[i][j][k]<<" ";
      }
      fiterates<<endl;
    }
  }
  for (i=0;i<numpoints;i++){
    for (j=0;j<numit;j++){
      delete[] iterates[i][j];
    }
    delete[] iterates[i];
  }
  delete[] iterates;
  delete[] finaliter;
  fiterates.close();
  gsl_interp_accel_free(thetaaac);
  gsl_interp_accel_free(caac);

}

void write_data_in_file(double *theta, double *c,double **data, int ncols, string filename){
  int i,j,k;
  ofstream outfile;
  std:ostringstream s;
  s<<filename.c_str()<<"_"<<count<<".tna";
  filename=s.str();
  outfile.precision(10);
  outfile.open(filename.c_str());

  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      outfile<<theta[i]<<" "<<c[j]<<" ";
      for (k=0;k<ncols;k++){
	outfile<<data[k][j*Ntheta+i]<<" ";
      }
      outfile<<endl;
    }
  }
  outfile.close();
}

void write_data_in_file2(double *theta,double *c,double *data, string filename){
  int i,j;
  ofstream outfile;
  std:ostringstream s;
  s<<filename.c_str()<<"_"<<count<<".tna";
  filename=s.str();
  outfile.precision(10);
  outfile.open(filename.c_str());

  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      outfile<<modulo(theta[i],1.)<<" "<<c[j]<<" ";
      outfile<<data[j*Ntheta+i]<<" ";
      outfile<<endl;
    }
  }
  outfile.close();
}

void write_data_in_file3(double **fvals,double **data, int ncols, string filename){
  int i,j,k;
  ofstream outfile;
  std:ostringstream s;
  s<<filename.c_str()<<"_"<<count<<".tna";
  filename=s.str();
  outfile.precision(10);
  outfile.open(filename.c_str());

  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      outfile<<modulo(fvals[0][j*Ntheta+i],1.)<<" "<<fvals[1][j*Ntheta+i]<<" ";
      for (k=0;k<ncols;k++){
	outfile<<data[k][j*Ntheta+i]<<" ";
      }
      outfile<<endl;
    }
  }
  outfile.close();
}
double one_step_of_Newton(double **thetac,double **fvals,double **invfvals, double **Kvals,double **FKvals,double **LambdaLvals, double **LambdaSvals,double *LambdaUvals,double **Pvals, double **invPvals){
  int i,j,k,l;
  double **Deltafvals=new double*[2];
  Deltafvals[0]=new double[Ntheta*Nc];
  Deltafvals[1]=new double[Ntheta*Nc];
  //zetaU:
  double *zetaUvals=new double[Ntheta*Nc];
  //zetaS:
  double **zetaSvals=new double*[2];
  zetaSvals[0]=new double[Ntheta*Nc];
  zetaSvals[1]=new double[Ntheta*Nc];
  double **Evals=new double*[5];
  for (i=0;i<5;i++){
    Evals[i]=new double[Ntheta*Nc];
  }
  double **EredNvals=new double*[15];
  for (i=0;i<15;i++){
    EredNvals[i]=new double[Ntheta*Nc];
  }
  double **DFKvals=new double*[25];
  for (i=0;i<25;i++){
    DFKvals[i]=new double[Ntheta*Nc];
  }
  double tmp,maxerrE,maxerrEredN;
  string filename;

  //We will need to interpolate K both in compute_E and in
  //torus_correction_stable. So we do it only once here.
  //I think this should not be here, because the values Kvals we use in
  //compute_E and compute_Ered are different; in the latter they have
  //been corrected.
  gsl_interp2d **K=new gsl_interp2d*[5];
  #pragma omp parallel for
  for (i=0;i<5;i++){
    K[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(K[i],thetac[0],thetac[1],Kvals[i],Ntheta,Nc);
  }
  if (count==1){
    //For the first iteration of the Newton we need to compute FKvals.
    //
    cout <<"Computing FK for the first iteration...."<<endl;
    compute_FK_eps(Kvals,FKvals);
    cout <<"Done."<<endl;
    //Fkvals are written just before calling compute_FK_DFK.
  }

  //-------------------------------------------------------------------------------
  //--------------Computation of corrections for tori and inner dynamics-----------
  //-------------------------------------------------------------------------------

  cout <<"Computing Error in invariance equation..."<<endl;
  compute_E(thetac,Evals,fvals,Kvals,K,FKvals);
  clean_data(Evals,5);
  cout<<"Done."<<endl;

  //Writing data:
  filename="Evals";
  write_data_in_file(thetac[0],thetac[1],Evals,5,filename);

  cout <<"Computing torus correction in the unstable direction..."<<endl;
  torus_correction_unstable(thetac,LambdaUvals,zetaUvals,fvals,Kvals,invPvals,Evals);
  clean_data2(zetaUvals);
  filename="zetaUvals";
  write_data_in_file2(thetac[0],thetac[1],zetaUvals,filename);
  cout<<"Done."<<endl;

  cout <<"Computing torus correction in the stable direction..."<<endl;
  torus_correction_stable(thetac,LambdaSvals,zetaSvals,invfvals,Kvals,invPvals,Evals,K);
  clean_data(zetaSvals,2);
  filename="zetaSvals";
  write_data_in_file(thetac[0],thetac[1],zetaSvals,2,filename);
  cout<<"Done."<<endl;

  cout <<"Computing inner dynamics correction..."<<endl;
  innerdynamics_correction(thetac,Deltafvals,fvals,invPvals,Evals);
  clean_data(Deltafvals,2);
  cout<<"Done."<<endl;

  cout <<"Correcting torus and inner dynamics..."<<endl;
  //-----------------------------------------------------
  //---------------Correcting tori and inner dynamics 
  //-----------------------------------------------------

  //We now get new corrected values for fvals and Kvals:
  #pragma omp parallel for private(j,k,l)
  for (i=0;i<Ntheta;i++){
    for (j=0;j<Nc;j++){
      for (k=0;k<2;k++){
	fvals[k][j*Ntheta+i]+=Deltafvals[k][j*Ntheta+i];
      }
      //k=0 and k=1 are not corrected
      for (k=2;k<=4;k++){
	for (l=0;l<2;l++){
	  Kvals[k][j*Ntheta+i]+=Pvals[k*5+l+2][j*Ntheta+i]*zetaSvals[l][j*Ntheta+i];
	}
	  Kvals[k][j*Ntheta+i]+=Pvals[k*5+4][j*Ntheta+i]*zetaUvals[j*Ntheta+i];
      }
    }
  }
  lift_data(fvals[0]);
  cout <<"Done."<<endl;
  filename="fvals";
  write_data_in_file(thetac[0],thetac[1],fvals,2,filename);
  filename="Deltafvals";
  write_data_in_file(thetac[0],thetac[1],Deltafvals,2,filename);
  filename="Kvals";
  write_data_in_file(thetac[0],thetac[1],Kvals,5,filename);

  cout <<"Correcting L and LambdaL (differentiating K and f)..."<<endl;
  //We also compute the correction of L and LambdaL by differentiating
  //K and f. We compute interpolations for the new values fvals and
  //Kvals:
  gsl_interp2d **f=new gsl_interp2d*[2];
  f[0]=gsl_interp2d_alloc(T,Ntheta,Nc);
  f[1]=gsl_interp2d_alloc(T,Ntheta,Nc);
  #pragma omp parallel for
  for (i=0;i<2;i++){
    gsl_interp2d_init(f[i],thetac[0],thetac[1],fvals[i],Ntheta,Nc);
  }
  #pragma omp parallel for
  for (i=0;i<5;i++){
    gsl_interp2d_init(K[i],thetac[0],thetac[1],Kvals[i],Ntheta,Nc);
  }
  #pragma omp parallel for private(j,k,tmp)
  for (i=0;i<Ntheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    double *DeltaP=new double[25];
    //Corrected values for LambdaL:
    for (j=0;j<Nc;j++){
      LambdaLvals[0][j*Ntheta+i]=gsl_interp2d_eval_deriv_x(f[0],thetac[0],thetac[1],fvals[0],thetac[0][i],thetac[1][j],thetaaac,caac);
      LambdaLvals[1][j*Ntheta+i]=gsl_interp2d_eval_deriv_y(f[0],thetac[0],thetac[1],fvals[0],thetac[0][i],thetac[1][j],thetaaac,caac);
      LambdaLvals[2][j*Ntheta+i]=gsl_interp2d_eval_deriv_x(f[1],thetac[0],thetac[1],fvals[1],thetac[0][i],thetac[1][j],thetaaac,caac);
      LambdaLvals[3][j*Ntheta+i]=gsl_interp2d_eval_deriv_y(f[1],thetac[0],thetac[1],fvals[1],thetac[0][i],thetac[1][j],thetaaac,caac);
      //Corrected values for L:
      //Since K[0] and K[1] are not corrected, we know the derivatives
      //as well:
      for (k=0;k<25;k++){
	DeltaP[k]=0.;
      }
      Pvals[0][j*Ntheta+i]=1.;
      Pvals[1][j*Ntheta+i]=0.;
      Pvals[5][j*Ntheta+i]=0.;
      Pvals[6][j*Ntheta+i]=1.;
      for (k=2;k<5;k++){
	tmp=gsl_interp2d_eval_deriv_x(K[k],thetac[0],thetac[1],Kvals[k],thetac[0][i],thetac[1][j],thetaaac,caac);
	DeltaP[5*k]=tmp-Pvals[5*k][j*Ntheta+i];
	Pvals[5*k][j*Ntheta+i]=tmp;
	tmp=gsl_interp2d_eval_deriv_y(K[k],thetac[0],thetac[1],Kvals[k],thetac[0][i],thetac[1][j],thetaaac,caac);
	DeltaP[5*k+1]=tmp-Pvals[5*k+1][j*Ntheta+i];
	Pvals[5*k+1][j*Ntheta+i]=tmp;
      }
      invertPvals2(invPvals,DeltaP,j*Ntheta+i);
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
    delete[] DeltaP;
  }
  clean_data(Pvals,25);
  clean_data(LambdaLvals,4);
  filename="Pvalsstep1";
  write_data_in_file(thetac[0],thetac[1],Pvals,25,filename);
  //We now correct P^{-1}: (we invert P)
  //P^{-1}(f(theta,c)) will be otained later interpolating these
  //values.
  //We don't need to invert P here if we have used the first order correction
  //given by inverPvals2.
  //invertPvals(Pvals,invPvals);
  filename="invPvalsstep1";
  write_data_in_file(thetac[0],thetac[1],invPvals,25,filename);
  cout <<"Done."<<endl;
  
  cout <<"Iterating inner dynamics..."<<endl;
  //We iterate the map for several initial conditions:
  std:ostringstream s;
  s<<"iterates"<<"_f"<<count<<".tna";
  filename=s.str();
  iterate_inner_dynamics2(thetac,fvals,f,K,Kvals,filename);
  cout <<"Done."<<endl;
  
  //Computintg detDf:
  /*
  cout<<"Computing det Df..."<<endl;
  double *detDfvals=new double[Ntheta*Nc];
  for (j=0;j<Nc;j++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (i=0;i<Ntheta;i++){
      detDfvals[j*Ntheta+i]=gsl_interp2d_eval_deriv_x(f[0],thetac[0],thetac[1],fvals[0],thetac[0][i],thetac[1][j],thetaaac,caac)*gsl_interp2d_eval_deriv_y(f[1],thetac[0],thetac[1],fvals[1],thetac[0][i],thetac[1][j],thetaaac,caac);
      detDfvals[j*Ntheta+i]+=-gsl_interp2d_eval_deriv_x(f[1],thetac[0],thetac[1],fvals[1],thetac[0][i],thetac[1][j],thetaaac,caac)*gsl_interp2d_eval_deriv_y(f[0],thetac[0],thetac[1],fvals[0],thetac[0][i],thetac[1][j],thetaaac,caac);
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  filename="detDf";
  write_data_in_file2(thetac[0],thetac[1],detDfvals,filename);
  delete[] detDfvals;
  cout <<"Done"<<endl;
  */

  //We will need the new values of f^{-1} at the grid points:
  cout<<"Computing the inverse of f..."<<endl;
  compute_invf(thetac,fvals,f,invfvals);
  //filename="invf_noliftvals";
  //write_data_in_file(thetac[0],thetac[1],invfvals,2,filename);
  lift_data(invfvals[0]);
  clean_data(invfvals,2);
  cout<<"Done."<<endl;

  cout <<"Iterating the inverse..."<<endl;
  gsl_interp2d **invf=new gsl_interp2d*[2];
  invf[0]=gsl_interp2d_alloc(T,Ntheta,Nc);
  gsl_interp2d_init(invf[0],thetac[0],thetac[1],invfvals[0],Ntheta,Nc);
  invf[1]=gsl_interp2d_alloc(T,Ntheta,Nc);
  gsl_interp2d_init(invf[1],thetac[0],thetac[1],invfvals[1],Ntheta,Nc);
  s.str("");
  s.clear();
  s<<"iterates"<<"_invf"<<count<<".tna";
  filename=s.str();
  iterate_inner_dynamics2(thetac,invfvals,invf,K,Kvals,filename);
  cout<<"Done."<<endl;
  
  //Here we compuse f and invf to see if verything is fine:
  /*
  cout <<"Computing error in the inverse..."<<endl;
  compute_invferror(thetac,fvals,f,invfvals,invf);
  cout <<"Done."<<endl;
  */

  //We will need also the values of DF at Kvals. Here we also compute
  //F at Kvals, which will be used by compute_E in the next iteration.
  //We print before because FKvals computed below correspond to next
  //iteration.
  filename="FKvals";
  write_data_in_file(thetac[0],thetac[1],FKvals,5,filename);
  cout <<"Computing F and DF at K(theta)..."<<endl;
  compute_FK_DFK(Kvals,FKvals,DFKvals);
  clean_data(FKvals,5);
  clean_data(DFKvals,25);
  filename="DFKvals";
  write_data_in_file(thetac[0],thetac[1],DFKvals,25,filename);
  cout <<"Done."<<endl;

  cout<<"Correcting normal bundle..."<<endl;
  correct_normal_bundle(thetac,fvals,invfvals,Kvals,DFKvals,Pvals,invPvals,LambdaLvals,LambdaSvals,LambdaUvals,EredNvals);
  cout<<"Done correcting normal bundle."<<endl;

  filename="LambdaSvals";
  write_data_in_file(thetac[0],thetac[1],LambdaSvals,4,filename);
  filename="LambdaLvals";
  write_data_in_file(thetac[0],thetac[1],LambdaLvals,4,filename);
  filename="LambdaUvals";
  write_data_in_file2(thetac[0],thetac[1],LambdaUvals,filename);
  filename="Pvals";
  write_data_in_file(thetac[0],thetac[1],Pvals,25,filename);
  filename="invPvals";
  write_data_in_file(thetac[0],thetac[1],invPvals,25,filename);

  //Shotting along tangent directions to the stable and unstable
  //manifolds:
  cout<<"Shotting along stable and unstable directions..."<<endl;
  if (count>1){
    shot_manifolds2(thetac,fvals,f,invfvals,invf,Pvals,Kvals);
  }
  cout <<"Done."<<endl;

  maxerrE=0.;
  maxerrEredN=0.;
  int maxerrEk=0;
  int maxerrEredNk=0;
  for (i=0;i<Ntheta;i++){
    for (j=10;j<Nc-10;j++){
      for (k=0;k<5;k++){
	tmp=fabs(Evals[k][j*Ntheta+i]);
	if (tmp>maxerrE){
	  maxerrE=tmp;
	  maxerrEk=k;
	}
      }
      for (k=0;k<15;k++){
	tmp=fabs(EredNvals[k][j*Ntheta+i]);
	if (tmp>maxerrEredN){
	  maxerrEredN=tmp;
	  maxerrEredNk=k;
	}
      }
    }
  }
  cout <<"----------------------------- "<<endl;
  if (count==1){
    cout <<"Errors in 1st Newton step: "<<endl;
  }
  else if (count==2){
    cout <<"Errors in 2nd Newton step: "<<endl;
  }
  else if (count==3){
    cout <<"Errors in 3rd Newton step: "<<endl;
  }
  else{
    cout <<"Errors in "<<count<<"th Newton step: "<<endl;
  }
  cout <<"In E, error="<<maxerrE<<" found in "<<maxerrEk<<"th element of E"<<endl;
  cout <<"In Ered, error="<<maxerrEredN<<" found in "<<maxerrEredNk<<"th element of EredN"<<endl;
  cout <<"----------------------------- "<<endl<<endl;


  //Free memory:
  for (i=0;i<15;i++){
    delete[] EredNvals[i];
  }
  delete[] EredNvals;
  delete[] Deltafvals[0];
  delete[] Deltafvals[1];
  delete[] Deltafvals;
  delete[] zetaUvals;
  delete[] zetaSvals[0];
  delete[] zetaSvals[1];
  delete[] zetaSvals;
  for (i=0;i<25;i++){
    delete[] DFKvals[i];
  }
  delete[] DFKvals;
  for (i=0;i<5;i++){
    delete[] Evals[i];
    gsl_interp2d_free(K[i]);
  }
  delete[] Evals;
  delete[] K;
  gsl_interp2d_free(f[0]);
  gsl_interp2d_free(f[1]);
  delete[] f;
  gsl_interp2d_free(invf[0]);
  gsl_interp2d_free(invf[1]);
  delete[] invf;
  
  //Returning error:
  //return maxerrE;
  if (maxerrE>maxerrEredN){
    return maxerrE;
  }
  else{
    return maxerrEredN;
  }
}

void clean_data(double **data,int ncols){
  double prectol=1e-12;
  int i,j;
  for (i=0;i<ncols;i++){
    for (j=0;j<Ntheta*Nc;j++){
      if (fabs(data[i][j])<prectol){
	data[i][j]=0.;
      }
    }
  }
}

void clean_data2(double *data){
  double prectol=1e-12;
  int j;
  for (j=0;j<Ntheta*Nc;j++){
    if (fabs(data[j])<prectol){
      data[j]=0.;
    }
  }
}

void compute_DFeps(double **thetac, double **uvvals, double *alphavals,double *dalphavals, double **Kvals, double **Evals){
  //Here we compute partial F/partial eps at Kvals and store it in Evals.
  //uvvals comes from ini_functions2
  int i,j,k,ndim;
  double h,t;
  

  #pragma omp parallel for private(j,k,h,t,ndim)
  for (i=0;i<Ntheta;i++){
    ndim=42;
    double *z=new double[ndim];
    double *Duvthetac=new double[4];
    for (j=0;j<Nc;j++){
      z[0]=Kvals[2][j*Ntheta+i];
      z[1]=Kvals[3][j*Ntheta+i];
      z[2]=uvvals[0][j*Ntheta+i];
      z[3]=uvvals[1][j*Ntheta+i];
      z[4]=Kvals[4][j*Ntheta+i];
      z[5]=eps;
      for (k=6;k<42;k++){
	if ((k+1)%7==0){
	  z[k]=1.;
	}
	else {
	  z[k]=0;
	}
      }
      t=0;
      h=hini;
      ini_rk78(ndim);
      while (t<2.*pi/omega){
	rk78(&t,z,&h,tol,hmin,hmax,ndim,vfieldepsvar);
      }
      h=-(t-2*pi/omega);
      rk78(&t,z,&h,tol,fabs(h),fabs(h),ndim,vfieldepsvar);
      D_uv_thetac(z[2],z[3],alphavals[j],dalphavals[j],thetac[1][j],Duvthetac);
      Evals[0][j*Ntheta+i]=Duvthetac[0]*z[23]+Duvthetac[1]*z[29];
      Evals[1][j*Ntheta+i]=Duvthetac[2]*z[23]+Duvthetac[3]*z[29];
      Evals[2][j*Ntheta+i]=z[11];
      Evals[3][j*Ntheta+i]=z[17];
      Evals[4][j*Ntheta+i]=z[35];
    }
    delete[] z;
    delete[] Duvthetac;
  }
}

void iterate_FK(double **thetac, double **Kvals){
  int i,j,k;
  gsl_interp2d **K=new gsl_interp2d*[5];
  for (i=0;i<5;i++){
    K[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(K[i],thetac[0],thetac[1],Kvals[i],Ntheta,Nc);
  }
  int numit=100000;
  int numpoints=2;
  int *finaliter=new int[numpoints];
  double ***iterates=new double**[numpoints];

  for (i=0;i<numpoints;i++){
    iterates[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iterates[i][j]=new double[5];
    }
  }
  #pragma omp parallel for private(j)
  for (i=0;i<numpoints;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    double thetatmp,ctmp,xtmp,ytmp,wtmp;
    double *Fthetax=new double[5];
    iterates[i][0][0]=0.5;
    iterates[i][0][1]=csimini +double(i)*(csimend-csimini)/double(numpoints-1);
    for (j=2;j<5;j++){
      iterates[i][0][j]=gsl_interp2d_eval_extrap(K[j],thetac[0],thetac[1],Kvals[j],modulo(iterates[i][0][0],1.),iterates[i][0][1],thetaaac,caac);
    }
    j=1;
    while(j<numit && fabs(iterates[i][j-1][1])<5 &&  fabs(iterates[i][j-1][2])<5){
      thetatmp=iterates[i][j-1][0];
      ctmp=iterates[i][j-1][1];
      xtmp=iterates[i][j-1][2];
      ytmp=iterates[i][j-1][3];
      wtmp=iterates[i][j-1][4];
      F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax);
      iterates[i][j][0]=Fthetax[0];
      iterates[i][j][1]=Fthetax[1];
      iterates[i][j][2]=Fthetax[2];
      iterates[i][j][3]=Fthetax[3];
      iterates[i][j][4]=Fthetax[4];
      j++;
    }
    finaliter[i]=j-1;
    delete[] Fthetax;
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  string filename;
  ofstream fiterates;
  std:ostringstream s;
  s<<"iteratesF"<<"_"<<count<<".tna";
  filename=s.str();
  fiterates.open(filename.c_str());
  for (i=0;i<numpoints;i++){
    //for (j=50;j<numit;j++){
    for (j=0;j<finaliter[i];j++){
      fiterates<<i<<" "<<j<<" ";
      for (k=0;k<5;k++){
	fiterates<<iterates[i][j][k]<<" ";
      }
      fiterates<<endl;
    }
  }
  for (i=0;i<numpoints;i++){
    for (j=0;j<numit;j++){
      delete[] iterates[i][j];
    }
    delete[] iterates[i];
  }
  delete[] iterates;
  delete[] finaliter;
  fiterates.close();
}

void iterate_FK2(double **thetac, double **Kvals){
  //Unlike in iterate_FK, we take a set of initial conditions in a
  //grid \theta x c and iterate them.
  int i,j,k,l;
  gsl_interp2d **K=new gsl_interp2d*[5];
  for (i=0;i<5;i++){
    K[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(K[i],thetac[0],thetac[1],Kvals[i],Ntheta,Nc);
  }
  int numit=50;
  int numpointstheta=10;
  int numpointsc=100;
  int *finaliter=new int[numpointstheta*numpointsc];
  double ***iterates=new double**[numpointstheta*numpointsc];

  for (i=0;i<numpointstheta*numpointsc;i++){
    iterates[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iterates[i][j]=new double[5];
    }
  }
  #pragma omp parallel for private(j,k)
  for (i=0;i<numpointstheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    double thetatmp,ctmp,xtmp,ytmp,wtmp;
    double *Fthetax=new double[5];
    for (k=0;k<numpointsc;k++){
      iterates[k*numpointstheta+i][0][0]=double(i)/double(numpointstheta);
      iterates[k*numpointstheta+i][0][1]=csimini +double(k)*(csimend-csimini)/double(numpointsc-1);
      for (j=2;j<5;j++){
	iterates[k*numpointstheta+i][0][j]=gsl_interp2d_eval_extrap(K[j],thetac[0],thetac[1],Kvals[j],modulo(iterates[k*numpointstheta+i][0][0],1.),iterates[k*numpointstheta+i][0][1],thetaaac,caac);
      }
      j=1;
      while(j<numit && fabs(iterates[k*numpointstheta+i][j-1][1])<5 &&  fabs(iterates[k*numpointstheta+i][j-1][2])<5){
	thetatmp=iterates[k*numpointstheta+i][j-1][0];
	ctmp=iterates[k*numpointstheta+i][j-1][1];
	xtmp=iterates[k*numpointstheta+i][j-1][2];
	ytmp=iterates[k*numpointstheta+i][j-1][3];
	wtmp=iterates[k*numpointstheta+i][j-1][4];
	F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax);
	iterates[k*numpointstheta+i][j][0]=Fthetax[0];
	iterates[k*numpointstheta+i][j][1]=Fthetax[1];
	iterates[k*numpointstheta+i][j][2]=Fthetax[2];
	iterates[k*numpointstheta+i][j][3]=Fthetax[3];
	iterates[k*numpointstheta+i][j][4]=Fthetax[4];
	j++;
      }
      finaliter[k*numpointstheta+i]=j-1;
    }
    delete[] Fthetax;
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  string filename;
  ofstream fiterates;
  std:ostringstream s;
  s<<"iteratesF"<<"_"<<count<<".tna";
  filename=s.str();
  fiterates.open(filename.c_str());
  for (i=0;i<numpointstheta;i++){
    for(l=0;l<numpointsc;l++){
      //for (j=50;j<numit;j++){
      for (j=0;j<finaliter[l*numpointstheta+i];j++){
	fiterates<<j<<" ";
	for (k=0;k<5;k++){
	  fiterates<<iterates[l*numpointstheta+i][j][k]<<" ";
	}
	fiterates<<endl;
      }
    }
  }
  for (i=0;i<numpointstheta*numpointsc;i++){
    for (j=0;j<numit;j++){
      delete[] iterates[i][j];
    }
    delete[] iterates[i];
  }
  delete[] iterates;
  delete[] finaliter;
  fiterates.close();
}

void predict_fKeps(double **thetac,double **LambdaSvals, double *LambdaUvals, double **invPvals, double **Pvals, double **Kvals,double **fvals){
  int i,j;
  double **zetaSepsvals=new double*[2];
  zetaSepsvals[0]=new double[Ntheta*Nc];
  zetaSepsvals[1]=new double[Ntheta*Nc];
  double *zetaUepsvals=new double[Ntheta*Nc];
  double **EDepsFvals=new double*[5];
  for (i=0;i<5;i++){
    EDepsFvals[i]=new double[Ntheta*Nc];
  }
  cout <<"  Computing DepsF"<<endl;
  //compute_DepsF(thetac,Kvals,EDepsFvals);
  cout <<"  Done."<<endl;

  delete[] zetaSepsvals[0];
  delete[] zetaSepsvals[1];
  delete[] zetaSepsvals;
  delete[] zetaUepsvals;
  for (i=0;i<5;i++){
    delete[] EDepsFvals[i];
  }
  delete[] EDepsFvals;
}

void compute_DepsF(double **thetac,double **Kvals,double **EDepsFvals){
  //This is to compute \partial F_eps/\partial eps, which is stored
  //in the variable EDepsFvals.
  int i,j,k;
  double h,t,alpha,dalpha;
  int ndim;

  #pragma omp parallel for private(i,t,ndim,k,h,alpha,dalpha)
  for (j=0;j<Nc;j++){
    double *z=new double[42];
    double *uv=new double[2];
    double *Duvthetac=new double[4];
    //We cannot proceed as in ini_functions2 here because c is not
    //kept constant when varying only theta. So, we need to start
    //again at (0,sqrt(2c)) for every point.
    for (i=0;i<Ntheta;i++){
      alpha=periodporbit(sqrt(2.*Kvals[1][j*Ntheta+i]));
      dalpha=dperiodporbit(sqrt(2.*Kvals[1][j*Ntheta+i]));
      t=0.;
      h=hini;
      uv[0]=0;
      uv[1]=sqrt(2.*Kvals[1][j*Ntheta+i]);
      ndim=2;
      ini_rk78(ndim);
      h=hini;
      while (t<modulo(Kvals[0][j*Ntheta+i],1.)*alpha){
	rk78(&t,uv,&h,tol,hmin,hmax,ndim,vfieldU);
      }
      h=-(t-modulo(Kvals[0][j*Ntheta+i],1.)*alpha);
      rk78(&t,uv,&h,tol,fabs(h),fabs(h),ndim,vfieldU);
      end_rk78(ndim);
      //We compute D_{theta,c}uv:
      z[0]=Kvals[2][j*Ntheta+i];
      z[1]=Kvals[3][j*Ntheta+i];
      z[2]=uv[0];
      z[3]=uv[1];
      z[4]=Kvals[4][j*Ntheta+i];
      z[5]=eps;
      for (k=6;k<42;k++){
	if (k%6==1){
	  z[k]=1.;
	}
	else{
	  z[k]=0.;
	}
      }
      ndim=42;
      ini_rk78(ndim);
      t=0.;
      h=hini;
      while (t<2.*pi/omega){
	rk78(&t,z,&h,tol,hmin,hmax,ndim,vfieldepsvar);
      }
      h=-(t-2.*pi/omega);
      rk78(&t,z,&h,tol,fabs(h),fabs(h),ndim,vfieldepsvar);
      end_rk78(ndim);
      D_uv_thetac(z[2],z[3],alpha,dalpha,Kvals[1][j*Ntheta+i],Duvthetac);
      EDepsFvals[0][j*Ntheta+i]=Duvthetac[0]*z[23]+Duvthetac[1]*z[29];//part theta/part u * part u/part eps+ part theta/part v *part v/part eps
      EDepsFvals[1][j*Ntheta+i]=Duvthetac[2]*z[23]+Duvthetac[3]*z[29];//part c/part u * part u/part eps+ part c/part v *part v/part eps
      EDepsFvals[2][j*Ntheta+i]=z[11]; //part x/part eps
      EDepsFvals[3][j*Ntheta+i]=z[17]; //part y/part eps
      EDepsFvals[4][j*Ntheta+i]=z[35]; //part w/part eps
    }
    delete[] Duvthetac;
    delete[] z;
    delete[] uv;
  }
}

void shot_manifolds2(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals,gsl_interp2d **invf,double **Pvals,double **Kvals){
  //Here we get points at the tangent spaces to W^s and W^u for
  //serveral values of theta and c. We compute a few iterations of
  //these values.
  int i,j,k,l,currentindex,m;
  double thetaini=0;
  double thetaend=1;
  int numpointstheta=1;//Number of points in the manifold. For each of them we compute approximations of the stable and stable manifold.
  int numpointsc=100;//Number of points in the manifold. For each of them we compute approximations of the stable and stable manifold.
  int numpointsstable=1;//Number of points to generate the tangent space to the stable manifld (2d)
  int numpointsdomain=100;//Number of points for the parameters parameterizing the manifolds
  int numit=20;
  int numpoints=numpointstheta*numpointsc;
  int *finaliter=new int[numpoints*(numpointsstable+1)*numpointsdomain];
  int *finaliterstable=new int[numpoints*numpointsstable*numpointsdomain];
  int *finaliterunstable=new int[numpoints*numpointsdomain];
  double deltaini=1e-6;//Distance at which we start moving the parameter parameterizing the tangent space to the invariant manifolds
  double deltaend=1e-3;//deltaend-deltaini determines the "length" of domain to be iterated
  //We store space for the iterates of numpointsstable points for the stable
  //manifold and 1 for the unstable (1d)
  double ***iteratesstable=new double **[numpoints*(numpointsstable)*numpointsdomain];
  for (i=0;i<numpoints*(numpointsstable)*numpointsdomain;i++){
    iteratesstable[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iteratesstable[i][j]=new double[5];
    }
  }
  double ***iteratesunstable=new double **[numpoints*numpointsdomain];
  for (i=0;i<numpoints*numpointsdomain;i++){
    iteratesunstable[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iteratesunstable[i][j]=new double[5];
    }
  }

  gsl_interp2d **P=new gsl_interp2d*[15];
  gsl_interp2d **K=new gsl_interp2d*[5];
  #pragma omp parallel for
  for (i=0;i<15;i++){
    P[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(P[i],thetac[0],thetac[1],Pvals[2+i/3*5+i%3],Ntheta,Nc);
    if(i<5){
      K[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
      gsl_interp2d_init(K[i],thetac[0],thetac[1],Kvals[i],Ntheta,Nc);
    }
  }

  #pragma omp parallel for private(i,j,k,l,currentindex)
  for (m=0;m<numpointstheta;m++){
    for (i=0;i<numpointsc;i++){
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      double thetatmp,ctmp,xtmp,ytmp,wtmp;
      double *Fthetax=new double[5];
      double *bpoint=new double[5];
      double *vs1=new double[5];
      double *vs2=new double[5];
      double modvs1,modvs2;
      double Pivs1,Pivs2;
      double *vu=new double[5];
      double modvu;
      double *vtmp=new double[5];
      double modvtmp;

      bpoint[0]=thetaini+double(m)*(thetaend-thetaini)/double(numpointstheta);
      bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpointsc);
      for (j=2;j<5;j++){
	bpoint[j]=gsl_interp2d_eval_extrap(K[j],thetac[0],thetac[1],Kvals[j],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      }
      //--------------------------------------------
      //Stable manifold:-----------------------------
      //--------------------------------------------
      //Vectors generating the plane tangent to the stable manifold at bpoint:
      for (k=0;k<5;k++){
	vs1[k]=gsl_interp2d_eval_extrap(P[3*k],thetac[0],thetac[1],Pvals[2+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
	vs2[k]=gsl_interp2d_eval_extrap(P[1+3*k],thetac[0],thetac[1],Pvals[3+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      }
      modvs1=0.;
      modvs2=0.;
      for (k=0;k<5;k++){
	modvs1+=vs1[k]*vs1[k];
	modvs2+=vs2[k]*vs2[k];
      }
      modvs1=sqrt(modvs1);
      modvs2=sqrt(modvs2);
      //#pragma omp parallel for private(j,thetatmp,ctmp,xtmp,ytmp,wtmp,Pivs1,Pivs2,currentindex,modvs1)
      for (k=0;k<numpointsstable;k++){
	if (numpointsstable>1){
	  Pivs1=double(numpointsstable-1-k)/double(numpointsstable);
	  Pivs2=double(k)/double(numpointsstable);
	}
	else{
	  Pivs1=1.;
	  Pivs2=0.;
	}
	modvtmp=0.;
	for (j=0;j<5;j++){
	  vtmp[j]=Pivs1*vs1[j]+Pivs2*vs2[j];
	  modvtmp+=vtmp[j]*vtmp[j];
	}
	modvtmp=sqrt(modvtmp);
	currentindex=(i*numpointstheta+m)*numpointsstable*numpointsdomain+k*numpointsdomain;
	for (l=0;l<numpointsdomain;l++){
	  double *Fthetax2=new double[5];
	  for (j=0;j<5;j++){
	    iteratesstable[currentindex+l][0][j]=bpoint[j];
	    iteratesstable[currentindex+l][0][j]+=double(l)/double(numpointsdomain)*(deltaend-deltaini)*vtmp[j]/modvtmp;
	  }
	  j=1;
	  while(j<numit && fabs(iteratesstable[currentindex+l][j-1][1])<5 &&  fabs(iteratesstable[currentindex+l][j-1][2])<5&&  fabs(iteratesstable[currentindex+l][j-1][4])<10){
	    thetatmp=iteratesstable[currentindex+l][j-1][0];
	    ctmp=iteratesstable[currentindex+l][j-1][1];
	    xtmp=iteratesstable[currentindex+l][j-1][2];
	    ytmp=iteratesstable[currentindex+l][j-1][3];
	    wtmp=iteratesstable[currentindex+l][j-1][4];
	    //cout <<thetatmp<<" "<<ctmp<<" "<<xtmp<<" "<<ytmp<<" "<<wtmp<<endl;
	    invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax2);
	    iteratesstable[currentindex+l][j][0]=Fthetax2[0];
	    iteratesstable[currentindex+l][j][1]=Fthetax2[1];
	    iteratesstable[currentindex+l][j][2]=Fthetax2[2];
	    iteratesstable[currentindex+l][j][3]=Fthetax2[3];
	    iteratesstable[currentindex+l][j][4]=Fthetax2[4];
	    j++;
	  }
	  finaliterstable[currentindex+l]=j;
	  delete[] Fthetax2;
	}
      }

      //-----------------------------------------------
      //-----------Unstable manifold:-------------------
      //------------------------------------------------
      //We shot an initial condition in the direction tangent to the unstable
      //manifold:
      modvu=0.;
      for (k=0;k<5;k++){
	vu[k]=gsl_interp2d_eval_extrap(P[2+3*k],thetac[0],thetac[1],Pvals[4+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
	modvu+=vu[k]*vu[k];
      }
      modvu=sqrt(modvu);
      currentindex=(i*numpointstheta+m)*numpointsdomain;
      //#pragma omp parallel for private(j,thetatmp,ctmp,xtmp,ytmp,wtmp)
      for (l=0;l<numpointsdomain;l++){
	double *Fthetax2=new double[5];
	for (j=0;j<5;j++){
	  iteratesunstable[currentindex+l][0][j]=bpoint[j];
	  iteratesunstable[currentindex+l][0][j]+=double(l)/double(numpointsdomain)*(deltaend-deltaini)*vu[j]/modvu;
	}
	j=1;
	while(j<numit && fabs(iteratesunstable[currentindex+l][j-1][1])<5 &&  fabs(iteratesunstable[currentindex+l][j-1][2])<5){
	  thetatmp=iteratesunstable[currentindex+l][j-1][0];
	  ctmp=iteratesunstable[currentindex+l][j-1][1];
	  xtmp=iteratesunstable[currentindex+l][j-1][2];
	  ytmp=iteratesunstable[currentindex+l][j-1][3];
	  wtmp=iteratesunstable[currentindex+l][j-1][4];
	  F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax2);
	  iteratesunstable[currentindex+l][j][0]=Fthetax2[0];
	  iteratesunstable[currentindex+l][j][1]=Fthetax2[1];
	  iteratesunstable[currentindex+l][j][2]=Fthetax2[2];
	  iteratesunstable[currentindex+l][j][3]=Fthetax2[3];
	  iteratesunstable[currentindex+l][j][4]=Fthetax2[4];
	  j++;
	}
	finaliterunstable[currentindex+l]=j;
	delete[] Fthetax2;
      }
      delete[] bpoint;
      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
      delete[] Fthetax;
      delete[] vu;
      delete[] vs1;
      delete[] vs2;
      delete[] vtmp;
    }
  }

  //Writing in files:
  string filename;
  std:ostringstream s;
  //Stable Manifold:
  ofstream fstable;
  fstable.precision(10);
  s<<"stable"<<"_"<<count<<".tna";
  filename=s.str();
  fstable.open(filename.c_str());
  for (i=0;i<numpoints;i++){
    for (l=0;l<numpointsstable;l++){
      for(m=0;m<numpointsdomain;m++){
	currentindex=i*numpointsstable*numpointsdomain+l*numpointsdomain;
	for (j=0;j<finaliterstable[currentindex];j++){
	  fstable<<i<<" "<<l<<" "<<m<<" "<<j<<" ";
	  for (k=0;k<5;k++){
	    fstable<<iteratesstable[currentindex+m][j][k]<<" ";
	  }
	  fstable<<endl;
	}
      }
    }
  }
  fstable.close();
  //Unstable Manifold:
  s.str("");
  s.clear();
  s<<"unstable"<<"_"<<count<<".tna";
  filename=s.str();
  ofstream funstable;
  funstable.precision(10);
  funstable.open(filename.c_str());
  for (i=0;i<numpointsc*numpointstheta*numpointsdomain;i++){
    for (j=0;j<finaliterunstable[i];j++){
      funstable<<i<<" "<<j<<" ";
      for (k=0;k<5;k++){
	funstable<<iteratesunstable[i][j][k]<<" ";
      }
      funstable<<endl;
    }
  }
  funstable.close();

  for (i=0;i<15;i++){
    gsl_interp2d_free(P[i]);
  }
  delete[] P;
  for (i=0;i<5;i++){
    gsl_interp2d_free(K[i]);
  }
  delete [] K;
  delete[] finaliter;
  for (i=0;i<numpoints*(numpointsstable)*numpointsdomain;i++){
    for (j=0;j<numit;j++){
      delete[] iteratesstable[i][j];
    }
    delete[] iteratesstable[i];
  }
  delete[] iteratesstable;
  for (i=0;i<numpoints*numpointsdomain;i++){
    for (j=0;j<numit;j++){
      delete[] iteratesunstable[i][j];
    }
    delete[] iteratesunstable[i];
  }
  delete[] iteratesunstable;
  delete[] finaliterstable;
  delete[] finaliterunstable;
}

void shot_manifolds(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals,gsl_interp2d **invf,double **Pvals,double **Kvals){
  //This function is depreciated and replace by shot_manifolds2
  //Here we iterate F_eps through the stable and unstable directions.
  //We use the linear approximations given by P.
  int i,j,k,l,currentindex,ndim;
  int numpoints=100;//Number of points in the manifold. For each of them we compute approximations of the stable and stable manifold.
  int numpointsstable=20;//Number of points to generate the stable manifld (2d)
  int numpointsdomain=200;//Number of points in the fundamental domain
  int numit=15;
  int *finaliter=new int[numpoints*(numpointsstable+1)*numpointsdomain];
  int *finaliterstable=new int[numpoints*numpointsstable*numpointsdomain];
  int *finaliterunstable=new int[numpoints*numpointsdomain];
  double hv=1e-6;//Distance at which we start the fundamental domain tangent to the invariant directions
  //We store space for the iterates of numpointsstable points for the stable
  //manifold and 1 for the unstable (1d)
  double h,t,alpha,tf;
  double *z=new double[5];
  double *uv=new double[2];
  double *uv2=new double[6];
  double ***iteratesstable=new double **[numpoints*(numpointsstable)*numpointsdomain];
  for (i=0;i<numpoints*(numpointsstable)*numpointsdomain;i++){
    iteratesstable[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iteratesstable[i][j]=new double[5];
    }
  }
  double ***iteratesunstable=new double **[numpoints*numpointsdomain];
  for (i=0;i<numpoints*numpointsdomain;i++){
    iteratesunstable[i]=new double*[numit];
    for (j=0;j<numit;j++){
      iteratesunstable[i][j]=new double[5];
    }
  }

  gsl_interp2d **P=new gsl_interp2d*[15];
  gsl_interp2d **K=new gsl_interp2d*[5];
  #pragma omp parallel for
  for (i=0;i<15;i++){
    P[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
    gsl_interp2d_init(P[i],thetac[0],thetac[1],Pvals[2+i/3*5+i%3],Ntheta,Nc);
    if(i<5){
      K[i]=gsl_interp2d_alloc(T,Ntheta,Nc);
      gsl_interp2d_init(K[i],thetac[0],thetac[1],Kvals[i],Ntheta,Nc);
    }
  }
  float progress = 0.0;
  int barWidth = 70;
  int pos;

  //#pragma omp parallel for private(j,k,l,currentindex)
  for (i=0;i<numpoints;i++){
    /*
    //Progress bar:
    while (progress < 1.0) {
      std::cout << "[";
      pos = barWidth * progress;
      for (l = 0; l < barWidth; ++l) {
	if (l < pos) std::cout << "=";
	else if (l == pos) std::cout << ">";
	else std::cout << " ";
      }
      std::cout << "] " << int(progress * 100.0) << " %\r";
      std::cout.flush();
      progress = double (i)/double(numpoints); 
    }
    std::cout << std::endl;
    */
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    double thetatmp,ctmp,xtmp,ytmp,wtmp;
    double *Fthetax=new double[5];
    double *bpoint=new double[5];
    double *vs1=new double[5];
    double *vs2=new double[5];
    double modvs1,modvs2;
    double Pivs1,Pivs2;
    double *vu=new double[5];
    double modvu;
    double *vtmp=new double[5];

    //bpoint[0]=0.4995;
    bpoint[0]=0.5;
    //bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpoints-1);
    //bpoint[1]=0.2602;
    //bpoint[1]=0.3;
    bpoint[1]=0.4 +double(i)*(0.4-0.3)/double(numpoints-1);
    for (j=2;j<5;j++){
      bpoint[j]=gsl_interp2d_eval_extrap(K[j],thetac[0],thetac[1],Kvals[j],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
    }
    //--------------------------------------------
    //Stable manifold:-----------------------------
    //--------------------------------------------
    /*
    //Vectors generating the plane tangent to the stable manifold at bpoint:
    for (k=0;k<5;k++){
      vs1[k]=gsl_interp2d_eval_extrap(P[3*k],thetac[0],thetac[1],Pvals[2+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      vs2[k]=gsl_interp2d_eval_extrap(P[1+3*k],thetac[0],thetac[1],Pvals[3+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
    }
    modvs1=0.;
    modvs2=0.;
    for (k=0;k<5;k++){
      modvs1+=vs1[k]*vs1[k];
      modvs2+=vs2[k]*vs2[k];
    }
    modvs1=sqrt(modvs1);
    modvs2=sqrt(modvs2);
    //#pragma omp parallel for private(j,thetatmp,ctmp,xtmp,ytmp,wtmp,Pivs1,Pivs2,currentindex,modvs1)
    for (k=0;k<numpointsstable;k++){
      //We take an initial condition in plane tangent to the stable
      //manifold at bpoint.
      double *vtmp=new double[5];
      double modvtmp;
      if (numpointsstable>1){
	Pivs1=double(numpointsstable-1-k)/double(numpointsstable-1);
	Pivs2=double(k)/double(numpointsstable-1);
      }
      else{
	Pivs1=1.;
	Pivs2=0.;
      }
      modvtmp=0.;
      for (j=0;j<5;j++){
	vtmp[j]=Pivs1*vs1[j]+Pivs2*vs2[j];
	modvtmp+=vtmp[j]*vtmp[j];
      }
      modvtmp=sqrt(modvtmp);
      currentindex=i*numpointsstable+k*numpointsdomain;
      for (j=0;j<5;j++){
	iteratesstable[currentindex][0][j]=bpoint[j]+hv*vtmp[j]/modvtmp;
      }
      tauc2uv(uv2,iteratesstable[currentindex][0][0],iteratesstable[currentindex][0][1]);
      ndim=5;
      h=hini;
      t=0.;
      tf=0;
      z[0]=iteratesstable[currentindex][0][2];
      z[1]=iteratesstable[currentindex][0][3];
      z[2]=uv2[0];
      z[3]=uv2[1];
      //z[2]=uv[0];
      //z[3]=uv[1];
      z[4]=iteratesstable[currentindex][0][4];
      for(l=1;l<numpointsdomain;l++){
	h=hini;
	tf+=(2.*pi/omega)/double(numpointsdomain);
	ini_rk78(ndim);
	while(t<tf){
	  rk78(&t,z,&h,tol,hmin,hmax,ndim,vfieldpert);
	}
	h=-(t-tf);
	rk78(&t,z,&h,tol,fabs(h),fabs(h),ndim,vfieldpert);
	end_rk78(ndim);
	uv2[0]=z[2];
	uv2[1]=z[3];
	uv2[2]=uv2[5]=1.;
	uv2[3]=uv2[4]=0.;
	uv2tauc(uv2,&iteratesstable[currentindex+l][0][0],&iteratesstable[currentindex+l][0][1]);
	iteratesstable[currentindex+l][0][2]=z[0];
	iteratesstable[currentindex+l][0][3]=z[1];
	iteratesstable[currentindex+l][0][4]=z[4];
      }
      //We now iterate the fundemantal domain:
      ndim=5;
      #pragma omp parallel for private(j,thetatmp,ctmp,xtmp,ytmp,wtmp,t,tf,h)
      for (l=0;l<numpointsdomain;l++){
	double *uvtmp=new double[6];
	double *ztmp=new double[5];
	j=1;
	while(j<numit && fabs(iteratesstable[currentindex+l][j-1][1])<5 &&  fabs(iteratesstable[currentindex+l][j-1][2])<5){
	  thetatmp=iteratesstable[currentindex+l][j-1][0];
	  ctmp=iteratesstable[currentindex+l][j-1][1];
	  xtmp=iteratesstable[currentindex+l][j-1][2];
	  ytmp=iteratesstable[currentindex+l][j-1][3];
	  wtmp=iteratesstable[currentindex+l][j-1][4];
	  tauc2uv(uvtmp,thetatmp,ctmp);
	  ztmp[0]=xtmp;
	  ztmp[1]=ytmp;
	  ztmp[2]=uvtmp[0];
	  ztmp[3]=uvtmp[1];
	  ztmp[4]=wtmp;
	  h=-hini;
	  t=double(l)*2.*pi/omega/double(numpointsdomain);
	  tf=t-2.*pi/omega;
	  ini_rk78(ndim);
	  while (t>tf){
	    rk78(&t,ztmp,&h,tol,hmin,hmax,ndim,vfieldpert);
	  }
	  h=(t-tf);
	  rk78(&t,ztmp,&h,tol,fabs(h),fabs(h),ndim,vfieldpert);
	  end_rk78(ndim);
	  uvtmp[0]=ztmp[2];
	  uvtmp[1]=ztmp[3];
	  uvtmp[2]=uvtmp[5]=1.;
	  uvtmp[3]=uvtmp[4]=0.;
	  uv2tauc(uvtmp,&iteratesstable[currentindex+l][j][0],&iteratesstable[currentindex+l][j][1]);
	  iteratesstable[currentindex+l][j][2]=ztmp[0];
	  iteratesstable[currentindex+l][j][3]=ztmp[1];
	  iteratesstable[currentindex+l][j][4]=ztmp[4];
	  j++;
	}
	//finaliterstable[currentindex+l]=j-1;
	finaliterstable[currentindex+l]=j;
	delete[] uvtmp;
	delete[] ztmp;
      }
    }
    */

    //-----------------------------------------------
    //-----------Unstable manifold:-------------------
    //------------------------------------------------
    //We first shot an initial condition in the direction tangent to the unstable
    //manifold:
    cout <<"Shotting Unstable:"<<endl;
    modvu=0.;
    for (k=0;k<5;k++){
      vu[k]=gsl_interp2d_eval_extrap(P[2+3*k],thetac[0],thetac[1],Pvals[4+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      modvu+=vu[k]*vu[k];
    }
    modvu=sqrt(modvu);
    currentindex=i*numpointsdomain;
    for (j=0;j<5;j++){
      iteratesunstable[currentindex][0][j]=bpoint[j];
      //if (j==2||j==3){
	iteratesunstable[currentindex][0][j]+=hv*vu[j]/modvu;
      //}
    }
    /*
    //Temporal, flow inspection--------------
    ofstream funstabletmp;
    funstabletmp.open("unstable_flow.tna");
    double *ztmp=new double[5];
    double *uvtmp=new double[6];
    tauc2uv(uvtmp,bpoint[0]+hv*vu[0]/modvu,bpoint[1]+hv*vu[1]/modvu);
    ztmp[0]=bpoint[2]+hv*vu[2]/modvu;
    ztmp[1]=bpoint[3]+hv*vu[3]/modvu;
    ztmp[2]=uvtmp[0];
    ztmp[3]=uvtmp[1];
    ztmp[4]=bpoint[4]+hv*vu[4]/modvu;
    t=0;
    tf=1000;
    h=hini;
    ndim=5;
    ini_rk78(ndim);
    while (t<tf){
      funstabletmp<<t<<" "<<ztmp[0]<<" "<<ztmp[1]<<" "<<ztmp[2]<<" "<<ztmp[3]<<" "<<ztmp[4]<<endl;
      rk78(&t,ztmp,&h,tol,hmin,hmax,ndim,vfieldpert);
    }
    funstabletmp<<endl;
    funstabletmp.close();
    delete[] ztmp;
    delete[] uvtmp;
    //Up to here flow inspection-----------
    */

    /*
    //Trying to iterate the inner map first (co-cycle approach??)----------------
    double thetaprev,cprev,thetanext,cnext;
    int n=0;
    thetaprev=bpoint[0];
    cprev=bpoint[1];
    thetanext=modulo(gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],bpoint[0],bpoint[1],thetaaac,caac),1.);
    cnext=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],bpoint[0],bpoint[1],thetaaac,caac);
    while (fabs(thetanext-bpoint[0])>1e-5){
      thetaprev=thetanext;
      cprev=cnext;
      thetanext=modulo(gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],thetaprev,cprev,thetaaac,caac),1.);
      cnext=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],thetaprev,cprev,thetaaac,caac);
      n++;
    }
    xtmp=gsl_interp2d_eval_extrap(K[2],thetac[0],thetac[1],Kvals[2],thetanext,cnext,thetaaac,caac);
    ytmp=gsl_interp2d_eval_extrap(K[3],thetac[0],thetac[1],Kvals[3],thetanext,cnext,thetaaac,caac);
    wtmp=gsl_interp2d_eval_extrap(K[4],thetac[0],thetac[1],Kvals[4],thetanext,cnext,thetaaac,caac);
    cout<<"Base point: ";
    for (j=0;j<5;j++){
      cout <<bpoint[j]<<" ";
    }
    cout<<endl;
    cout <<n<<" "<<thetanext<<" "<<cnext<<" "<<xtmp<<" "<<ytmp<<" "<<wtmp<<endl;
    //Up to here inner iteration-----------------------------
    */
    /*
    j=1;
    while(j<numit && fabs(iteratesunstable[currentindex][j-1][1])<5 &&  fabs(iteratesunstable[currentindex][j-1][2])<5){
      thetatmp=iteratesunstable[currentindex][j-1][0];
      ctmp=iteratesunstable[currentindex][j-1][1];
      xtmp=iteratesunstable[currentindex][j-1][2];
      ytmp=iteratesunstable[currentindex][j-1][3];
      wtmp=iteratesunstable[currentindex][j-1][4];
      F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax);
      iteratesunstable[currentindex][j][0]=Fthetax[0];
      iteratesunstable[currentindex][j][1]=Fthetax[1];
      iteratesunstable[currentindex][j][2]=Fthetax[2];
      iteratesunstable[currentindex][j][3]=Fthetax[3];
      iteratesunstable[currentindex][j][4]=Fthetax[4];
      j++;
    }
    //finaliterunstable[currentindex]=j-1;
    finaliterunstable[currentindex]=j;
    */
    //Now we iterate forwards the fundamental domain
    for(j=0;j<5;j++){
      vtmp[j]=iteratesunstable[currentindex][1][j]-iteratesunstable[currentindex][0][j];
      /*
      if (j!=2 && j!=3){
	vtmp[j]=0.;
      }
      */
    }
    #pragma omp parallel for private(j,thetatmp,ctmp,xtmp,ytmp,wtmp)
    //for (l=1;l<numpointsdomain;l++){
    for (l=0;l<numpointsdomain;l++){
      double *Fthetax2=new double[5];
      for (j=0;j<5;j++){
	//iteratesunstable[currentindex+l][0][j]=iteratesunstable[currentindex][0][j];
	//iteratesunstable[currentindex+l][0][j]+=double(l)/double(numpointsdomain)*vtmp[j];
	iteratesunstable[currentindex+l][0][j]=bpoint[j];
	iteratesunstable[currentindex+l][0][j]+=double(l)/double(numpointsdomain-1)*1e-2*vu[j]/modvu;
      }
      j=1;
      while(j<numit && fabs(iteratesunstable[currentindex+l][j-1][1])<5 &&  fabs(iteratesunstable[currentindex+l][j-1][2])<5){
	thetatmp=iteratesunstable[currentindex+l][j-1][0];
	ctmp=iteratesunstable[currentindex+l][j-1][1];
	xtmp=iteratesunstable[currentindex+l][j-1][2];
	ytmp=iteratesunstable[currentindex+l][j-1][3];
	wtmp=iteratesunstable[currentindex+l][j-1][4];
	F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax2);
	iteratesunstable[currentindex+l][j][0]=Fthetax2[0];
	iteratesunstable[currentindex+l][j][1]=Fthetax2[1];
	iteratesunstable[currentindex+l][j][2]=Fthetax2[2];
	iteratesunstable[currentindex+l][j][3]=Fthetax2[3];
	iteratesunstable[currentindex+l][j][4]=Fthetax2[4];
	j++;
      }
      //finaliterunstable[currentindex+l]=j-1;
      finaliterunstable[currentindex+l]=j;
      delete[] Fthetax2;
    }
    delete[] bpoint;
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
    delete[] Fthetax;
    delete[] vu;
    delete[] vs1;
    delete[] vs2;
    delete[] vtmp;
  }
  //cout<<endl;//I think we need to print this to end the progress bar

  //Writing in files:
  string filename;
  std:ostringstream s;
  //Stable:
  /*
  ofstream fstable;
  fstable.precision(10);
  s<<"stable"<<"_"<<count<<".tna";
  filename=s.str();
  fstable.open(filename.c_str());
  for (i=0;i<numpoints*numpointsstable*numpointsdomain;i++){
    for (j=0;j<finaliterstable[i];j++){
      fstable<<j<<" ";
      for (k=0;k<5;k++){
	fstable<<iteratesstable[i][j][k]<<" ";
      }
      fstable<<endl;
    }
  }
  fstable.close();
  */
  //Unstable:
  s.str("");
  s.clear();
  s<<"unstable"<<"_"<<count<<".tna";
  filename=s.str();
  ofstream funstable;
  funstable.precision(10);
  funstable.open(filename.c_str());
  for (i=0;i<numpoints*numpointsdomain;i++){
    for (j=0;j<finaliterunstable[i];j++){
      //if (j%3==0){
	funstable<<j<<" ";
	for (k=0;k<5;k++){
	  funstable<<iteratesunstable[i][j][k]<<" ";
	}
	funstable<<endl;
      //}
    }
  }
  funstable.close();

  for (i=0;i<15;i++){
    gsl_interp2d_free(P[i]);
  }
  delete[] P;
  for (i=0;i<5;i++){
    gsl_interp2d_free(K[i]);
  }
  delete [] K;
  delete[] finaliter;
  for (i=0;i<numpoints*(numpointsstable)*numpointsdomain;i++){
    for (j=0;j<numit;j++){
      delete[] iteratesstable[i][j];
    }
    delete[] iteratesstable[i];
  }
  delete[] iteratesstable;
  for (i=0;i<numpoints*numpointsdomain;i++){
    for (j=0;j<numit;j++){
      delete[] iteratesunstable[i][j];
    }
    delete[] iteratesunstable[i];
  }
  delete[] iteratesunstable;
  delete[] finaliterstable;
  delete[] finaliterunstable;
  delete[] z;
  delete[] uv;
  delete[] uv2;
}

void compute_invferror(double **thetac, double **fvals, gsl_interp2d **f,double **invfvals,gsl_interp2d **invf){
  //Here we compose f and f^-1 to check the error.
  int i,j;
  double **invffvals=new double*[6];
  /*
  for (i=0;i<2;i++){
    gsl_interp2d_init(f[i],thetac[0],thetac[1],fvals[i],Ntheta,Nc);
    gsl_interp2d_init(invf[i],thetac[0],thetac[1],invfvals[i],Ntheta,Nc);
  }
  */
  for (i=0;i<6;i++){
    invffvals[i]=new double[Ntheta*Nc];
  }
  #pragma omp parallel for private(j)
  for (i=0;i<Ntheta;i++){
    gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
    gsl_interp_accel *caac=gsl_interp_accel_alloc();
    for (j=0;j<Nc;j++){
      invffvals[0][j*Ntheta+i]=gsl_interp2d_eval_extrap(f[0],thetac[0],thetac[1],fvals[0],thetac[0][i],thetac[1][j],thetaaac,caac);
      invffvals[0][j*Ntheta+i]=modulo(invffvals[0][j*Ntheta+i],1.);
      invffvals[1][j*Ntheta+i]=gsl_interp2d_eval_extrap(f[1],thetac[0],thetac[1],fvals[1],thetac[0][i],thetac[1][j],thetaaac,caac);
      invffvals[2][j*Ntheta+i]=gsl_interp2d_eval_extrap(invf[0],thetac[0],thetac[1],invfvals[0],invffvals[0][j*Ntheta+i],invffvals[1][j*Ntheta+i],thetaaac,caac);
      invffvals[2][j*Ntheta+i]=modulo(invffvals[2][j*Ntheta+i],1.);
      invffvals[3][j*Ntheta+i]=gsl_interp2d_eval_extrap(invf[1],thetac[0],thetac[1],invfvals[1],invffvals[0][j*Ntheta+i],invffvals[1][j*Ntheta+i],thetaaac,caac);
      invffvals[4][j*Ntheta+i]=invffvals[2][j*Ntheta+i]-thetac[0][i];
      invffvals[5][j*Ntheta+i]=invffvals[3][j*Ntheta+i]-thetac[1][j];
    }
    gsl_interp_accel_free(thetaaac);
    gsl_interp_accel_free(caac);
  }
  cout <<"caca"<<endl;
  string filename;
  filename="invffvals";
  write_data_in_file(thetac[0],thetac[1],invffvals,6,filename);

  for (i=0;i<6;i++){
    delete[] invffvals[i];
  }
  delete[] invffvals;
}
