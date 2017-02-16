#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rk78.h"
#include <iostream>
#include <fstream>
#include "lu.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_deriv.h>

using namespace std;


void multMatrix_array(double *C,double *A, int fA, int cA, double *B,int fB,int cB);
void multMatrix(double **C,double **A, int fA, int cA, double **B,int fB,int cB);
void F_DF_eps(double theta, double c,double x, double y, double w, double *Fthetax,double *DF);
void F_0(double *thetax,double **DF);
void F_eps(double theta, double c,double x, double y, double w, double *Fthetax);
void invF_eps(double theta, double c,double x, double y, double w, double *Fthetax);
void strobomap(double *x,double tf, void (*vfield)(double t, double *x, int ndim, double *dx),int ndim);
double periodporbit(double v0);
double period4numderiv( double v0,void *params);
double dperiodporbit(double v0);
void vfieldUvar(double t, double *x, int ndim, double *dx);
void vfieldUvar2(double t, double *x, int ndim, double *dx);
double omegap(double u0,double v0);
void vfieldUw(double t, double *x, int nidm, double *dx);
void vfieldU(double t, double *x, int ndim, double *dx);
void vfieldpert(double t, double *x, int ndim, double *dx);
void vfieldunpert(double t, double *x, int ndim, double *dx);
void vfieldunpertvar(double t, double *x, int ndim, double *dx);
void vfieldX(double t, double *x, int ndim, double *dx);
void vfieldepsvar(double t,double *x,int ndim,double *dx);
double omega_h(double u0,double v0, double xh,double yh);
double omegap_num(double u0,double v0);
double melnikov_function(double *zh,double s0);
void clean_matrix(gsl_matrix *M);
void uv2tauc(double *uv, double *tau, double *c);
void tauc2uv(double *uv,double tau, double c);
double cosangle(double *x1,double *x2,double *x3);
double modulo(double a, double b);
void write_data_in_file(double *theta,double *c,double **data, int ncols, string filename);
void write_data_in_file2(double *theta,double *c,double *data, string filename);
void write_data_in_file3(double **fvals,double **data, int ncols, string filename);


double melnikov_function(double *zh,double s0){
  //This computes the melnikov function at s0.
  //Note that (x[2],x[3])=\phi_U(s0;p(theta,c))

  double sum,xant,yant,tant;
  int i;
  int ndim=5;
  double *x=new double[ndim];
  double fant,fnew;
  double h,t;
  bool plotit=false;
  
  x[0]=zh[0];
  x[1]=zh[1];
  x[2]=zh[2];
  x[3]=zh[3];
  x[4]=zh[4];
  //x[4]=omega_h(x[0],x[1],x[2],x[3]);

  sum=0;
  t=0;
  //We start integrating forwards
  h=1e-1;
  xant=x[0];
  yant=x[1];
  fant=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
  if (plotit){
    cout<<"2 "<<t<<" ";
    for (i=0;i<5;i++){
      cout<<x[i]<<" ";
    }
    cout <<fant<<endl;
  }
  sum=sum+fabs(h)*fant;
  ini_rk78(ndim);
  rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldunpert);
  //x[4]=omega_h(x[0],x[1],x[2],x[3]);
  //fnew=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
  fnew=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
  sum=sum+fabs(h)*(fnew-fant)/2.0;

  while(xant*xant+yant*yant<x[0]*x[0]+x[1]*x[1]){//We integrate until the trajectory starts approaching Q0
    xant=x[0];
    yant=x[1];
    tant=t;
    //fant=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fant=fnew;
    if (plotit){
      cout<<"2 "<<t<<" ";
      for (i=0;i<5;i++){
	cout<<x[i]<<" ";
      }
      cout <<fant<<endl;
    }
    sum=sum+fabs(h)*fant;
    rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldunpert);
    //x[4]=omega_h(x[0],x[1],x[2],x[3]);
    //fnew=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fnew=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
    sum=sum+fabs(h)*(fnew-fant)/2.;
  }
  while(xant*xant+yant*yant>=x[0]*x[0]+x[1]*x[1]){//We integrate until we leave a neighbourhood of Q0
    xant=x[0];
    yant=x[1];
    tant=t;
    //fant=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fant=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
    if (plotit){
      cout<<"2 "<<t<<" ";
      for (i=0;i<5;i++){
	cout<<x[i]<<" ";
      }
      cout <<fant<<endl;
    }
    sum=sum+fabs(h)*fant;
    rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldunpert);
  //x[4]=omega_h(x[0],x[1],x[2],x[3]);
    //fnew=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fnew=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
    sum=sum+fabs(h)*(fnew-fant)/2.;
  }

  //We now integrate backwards
  t=0;
  h=-1e-1;
  x[0]=zh[0];
  x[1]=zh[1];
  x[2]=zh[2];
  x[3]=zh[3];
  x[4]=zh[4];
  xant=x[0];
  yant=x[1];
  //fant=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
  fant=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
  if (plotit){
    cout<<"2 "<<t<<" ";
    for (i=0;i<5;i++){
      cout<<x[i]<<" ";
    }
    cout <<fant<<endl;
  }
  sum=sum+fabs(h)*fant;
  rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldunpert);
  //x[4]=omega_h(x[0],x[1],x[2],x[3]);
  //fnew=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
  fnew=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
  sum=sum+fabs(h)*(fnew-fant)/2.0;

  while(xant*xant+yant*yant<x[0]*x[0]+x[1]*x[1]){//We integrate until the trajectory starts approaching Q0
    xant=x[0];
    yant=x[1];
    tant=t;
    //fant=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fant=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
    if(plotit){
    cout<<"2 "<<t<<" ";
      for (i=0;i<5;i++){
	cout<<x[i]<<" ";
      }
      cout <<fant<<endl;
    }
    sum=sum+fabs(h)*fant;
    rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldunpert);
    //x[4]=omega_h(x[0],x[1],x[2],x[3]);
    //fnew=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fnew=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
    sum=sum+fabs(h)*(fnew-fant)/2.;
  }
  while(xant*xant+yant*yant>=x[0]*x[0]+x[1]*x[1]){//We integrate until we leave a neighbourhood of Q0
    xant=x[0];
    yant=x[1];
    tant=t;
    //fant=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fant=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
    if (plotit){
      cout<<"2 "<<t<<" ";
      for (i=0;i<5;i++){
	cout<<x[i]<<" ";
      }
      cout <<fant<<endl;
    }
    sum=sum+fabs(h)*fant;
    rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldunpert);
    //x[4]=omega_h(x[0],x[1],x[2],x[3]);
    //fnew=x[1]*(tXi*x[4]+tF*sin(omega*(t+s0)));
    fnew=x[1]*(tF*sin(omega*(t+s0))-tK*(x[0]-x[2]));
    sum=sum+fabs(h)*(fnew-fant)/2.;
  }
  end_rk78(ndim);

  delete [] x;
  return sum;

}

double omega_h(double xh, double yh, double u0,double v0){
  //This returns the initial condition for w for the unstable
  //manifold.
  double h,sum,t,tant,xant,yant;
  double w0p,tf,fant,fnew,ynew,xnew;
  int ndim=2;
  double *x=new double[ndim];
  bool approaching,closetopoint;

  w0p=omegap(u0,v0);

  //sum=w0p;
  sum=0;
  tf=-1000;
  h=-1e-1;
  x[0]=xh;
  x[1]=yh;
  ndim=2;
  t=0;
  tant=t;
  xant=xh;
  yant=yh;
  fant=exp(lambda*tant)*yant;
  sum=sum+fabs(h)*fant;
  ini_rk78(ndim);
  rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldX);
  xnew=x[0];
  ynew=x[1];
  fnew=exp(lambda*t)*xnew;
  sum=sum+fabs(h)*(fnew-fant)/2.;

  if (xant*xant+yant*yant>xnew*xnew+ynew*ynew){
    approaching=true;
  }
  else{
    approaching=false;
  }

  if(!approaching){
    while( xant*xant+yant*yant<xnew*xnew+ynew*ynew){
      xant=x[0];
      yant=x[1];
      tant=t;
      fant=exp(lambda*tant)*yant;
      sum=sum+fabs(h)*fant;
      rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldX);
      xnew=x[0];
      ynew=x[1];
      fnew=exp(lambda*t)*x[1];
      sum=sum+fabs(h)*(fnew-fant)/2.;
    }
    //Now we need to integrate until we start getting away from Q0
  }

  //while (t>-tf && fabs(fant)>=fabs(fnew)){
  //while (t>tf && (xant*xant+yant*yant<xnew*xnew+ynew*ynew && ( yant<0 && xant>0 || yant>0 && xant<0) || xant*xant+yant*yant>xnew*xnew+ynew*ynew && (yant>0 && xant>0 || yant<0 && xant<0) || yant==0)){
  //while (t>tf && xant*xnew>=0){
  while(xant*xant+yant*yant>=xnew*xnew+ynew*ynew){
    xant=x[0];
    yant=x[1];
    tant=t;
    fant=exp(lambda*tant)*yant;
    sum=sum+fabs(h)*fant;
    rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldX);
    xnew=x[0];
    ynew=x[1];
    fnew=exp(lambda*t)*x[1];
    sum=sum+fabs(h)*(fnew-fant)/2.;
  }
  end_rk78(ndim);
  delete [] x;
  sum=w0p-tk*sum;
  return sum;
}

double omegap(double u0,double v0){
  //This returns the initial condition for the periodic orbit wp(t)
  //using the formula with the integral
  //
  
  double t,alphac,H,sum,h,tant,xant;
  double *x=new double[2];
  int i,numsteps;
  int ndim=2;
  
  numsteps=1000;
  H=0.5*v0*v0-u0*u0/4.0*beta*(1.0-u0*u0/2.0);
  alphac=periodporbit(sqrt(2.0*H));
  h=alphac/double(numsteps);
  t=0;
  sum=0;
  x[0]=u0;
  x[1]=v0;
  ini_rk78(ndim);
  for (i=0;i<numsteps;i++){
    tant=t;
    xant=x[1];
    sum=sum+h*x[1]*exp(lambda*t);
    rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldU);
    sum=sum+h*(x[1]*exp(lambda*t)-xant*exp(lambda*tant))/2.;
  }
  end_rk78(ndim);
  sum=-exp(-lambda*alphac)/(1-exp(-lambda*alphac))*tk*sum;

  delete [] x;
  return sum;
}



double omegap_num(double u0,double v0){
  //This function numerically computes the initial condition
  //for the periodic orbit wp(t).

  double H,alpha,dww,out;
  int n=3;
  double h=hini;
  int i,j,iter;
  double Newtol=1.0e-12;
  double dist,xn,xprev;
  int maxiterNexton;
  double *x=new double[n*n+n];
  int ndim=2;

  h=0.0001;
  maxiterNexton=200;

  H=0.5*v0*v0-u0*u0/4.0*beta*(1.0-u0*u0/2.0);
  alpha=periodporbit(sqrt(2.0*H));

  ndim=n*n+n;

  x[0]=u0;
  x[1]=v0;
  x[2]=0;
  for (i=n;i<ndim;i++){
    x[i]=0;
  }
  for (i=n;i<ndim;i=i+n+1){
    x[i]=1;
  }

  xprev=x[2];
  strobomap(x,alpha,vfieldUvar,ndim);
  dist=fabs(xprev-x[2]);
  iter=0;
  while(dist>Newtol && iter<maxiterNexton){
    dww=x[ndim-1];
    xn=xprev-(x[2]-xprev)/(dww-1);
    xprev=x[2];
    x[0]=u0;
    x[1]=v0;
    x[2]=xn;
    for (i=n;i<ndim;i++){
      x[i]=0;
    }
    for (i=n;i<ndim;i=i+n+1){
      x[i]=1;
    }
    strobomap(x,alpha,vfieldUvar,ndim);
    dist=fabs(xn-x[2]);
    iter++;
  }
  out=x[2];
  delete [] x;
  //end_rk78(ndim);
  if (iter>=maxiterNexton){
    cout <<"Newton failed when computing w^p"<<endl;
    exit(1);
  }
  else{
    return out;
  }

}

void vfieldUw(double t, double *x, int nidm, double *dx){
  //Unperturbed systems u-v-w

  dx[0]=x[1];
  dx[1]=0.5*beta*x[0]*(1-x[0]*x[0]);
  dx[2]=-lambda*x[2]-tk*x[1];
}


void vfieldUvar(double t, double *x, int ndim, double *dx){
  //Variatonal equations of the system u-v-w

  dx[0]=x[1];
  dx[1]=0.5*beta*x[0]*(1-x[0]*x[0]);
  dx[2]=-lambda*x[2]-tk*x[1];
  dx[3]=x[6];
  dx[4]=x[7];
  dx[5]=x[8];
  dx[6]=0.5*beta*(1-3*x[0]*x[0])*x[3];
  dx[7]=0.5*beta*(1-3*x[0]*x[0])*x[4];
  dx[8]=0.5*beta*(1-3*x[0]*x[0])*x[5];
  dx[9]=-tk*x[6]-lambda*x[9];
  dx[10]=-tk*x[7]-lambda*x[10];
  dx[11]=-tk*x[8]-lambda*x[11];
}

void vfieldUvar2(double t, double *x, int ndim, double *dx){

  dx[0]=x[1];
  dx[1]=0.5*beta*x[0]*(1.-x[0]*x[0]);
  dx[2]=x[4];
  dx[3]=x[5];
  dx[4]=0.5*beta*(1.-3.*x[0]*x[0])*x[2];
  dx[5]=0.5*beta*(1.-3.*x[0]*x[0])*x[3];
}


void vfieldunpertvar(double t, double *x, int ndim, double *dx){
  double a21,a43;
  //Variational equations of the unperturbed systems

  a21=0.5*(1.-3.*x[0]*x[0]);
  a43=0.5*beta*(1.-3.*x[2]*x[2]);

  dx[0]=x[1];
  dx[1]=0.5*x[0]*(1-x[0]*x[0]);
  dx[2]=x[3];
  dx[3]=0.5*beta*x[2]*(1-x[2]*x[2]);
  dx[4]=-lambda*x[4]-tk*(x[1]+x[3]);

  dx[5]=x[10];
  dx[6]=x[11];
  dx[7]=x[12];
  dx[8]=x[13];
  dx[9]=x[14];

  dx[10]=a21*x[5];
  dx[11]=a21*x[6];
  dx[12]=a21*x[7];
  dx[13]=a21*x[8];
  dx[14]=a21*x[9];

  dx[15]=x[20];
  dx[16]=x[21];
  dx[17]=x[22];
  dx[18]=x[23];
  dx[19]=x[24];

  dx[20]=a43*x[15];
  dx[21]=a43*x[16];
  dx[22]=a43*x[17];
  dx[23]=a43*x[18];
  dx[24]=a43*x[19];

  dx[25]=-tk*(x[10]+x[20])-lambda*x[25];
  dx[26]=-tk*(x[11]+x[21])-lambda*x[26];
  dx[27]=-tk*(x[12]+x[22])-lambda*x[27];
  dx[28]=-tk*(x[13]+x[23])-lambda*x[28];
  dx[29]=-tk*(x[14]+x[24])-lambda*x[29];

}

void vfieldunpert(double t, double *x, int ndim, double *dx){
  //Unperturbed system

  dx[0]=x[1];
  dx[1]=0.5*x[0]*(1-x[0]*x[0]);
  dx[2]=x[3];
  dx[3]=0.5*beta*x[2]*(1-x[2]*x[2]);
  dx[4]=-lambda*x[4]-tk*(x[1]+x[3]);

}

void vfieldpert(double t, double *x, int ndim, double *dx){
  //Full systems

  dx[0]=x[1];
  dx[1]=0.5*x[0]*(1-x[0]*x[0])+eps*(-2*tzeta*x[1]+tXi*x[4]+tK*(x[0]-x[2])+tF*sin(omega*t));
  dx[2]=x[3];
  dx[3]=0.5*beta*x[2]*(1-x[2]*x[2])+eps*(-2*tzeta*x[3]+tXi*x[4]+tK*(x[2]-x[0])+tF*sin(omega*t));
  dx[4]=-lambda*x[4]-tk*(x[1]+x[3]);

}

void vfieldpertvar(double t, double *x, int ndim, double *dx){
  double a21,a43;
  //Variational equations of the full perturbed field

  a21=0.5*(1.-3.*x[0]*x[0]);
  a43=0.5*beta*(1.-3.*x[2]*x[2]);

  dx[0]=x[1];
  dx[1]=0.5*x[0]*(1-x[0]*x[0])+eps*(-2*tzeta*x[1]+tXi*x[4]+tK*(x[0]-x[2])+tF*sin(omega*t));
  dx[2]=x[3];
  dx[3]=0.5*beta*x[2]*(1-x[2]*x[2])+eps*(-2*tzeta*x[3]+tXi*x[4]+tK*(x[2]-x[0])+tF*sin(omega*t));
  dx[4]=-lambda*x[4]-tk*(x[1]+x[3]);

  dx[5]=x[10];
  dx[6]=x[11];
  dx[7]=x[12];
  dx[8]=x[13];
  dx[9]=x[14];

  dx[10]=a21*x[5]-2.*eps*tzeta*x[10]+eps*tXi*x[25]+eps*tK*(x[5]-x[15]);
  dx[11]=a21*x[6]-2.*eps*tzeta*x[11]+eps*tXi*x[26]+eps*tK*(x[6]-x[16]);
  dx[12]=a21*x[7]-2.*eps*tzeta*x[12]+eps*tXi*x[27]+eps*tK*(x[7]-x[17]);
  dx[13]=a21*x[8]-2.*eps*tzeta*x[13]+eps*tXi*x[28]+eps*tK*(x[8]-x[18]);
  dx[14]=a21*x[9]-2.*eps*tzeta*x[14]+eps*tXi*x[29]+eps*tK*(x[9]-x[19]);

  dx[15]=x[20];
  dx[16]=x[21];
  dx[17]=x[22];
  dx[18]=x[23];
  dx[19]=x[24];

  dx[20]=a43*x[15]-2.*eps*tzeta*x[20]+eps*tXi*x[25]+eps*tK*(x[15]-x[5]);
  dx[21]=a43*x[16]-2.*eps*tzeta*x[21]+eps*tXi*x[26]+eps*tK*(x[16]-x[6]);
  dx[22]=a43*x[17]-2.*eps*tzeta*x[22]+eps*tXi*x[27]+eps*tK*(x[17]-x[7]);
  dx[23]=a43*x[18]-2.*eps*tzeta*x[23]+eps*tXi*x[28]+eps*tK*(x[18]-x[8]);
  dx[24]=a43*x[19]-2.*eps*tzeta*x[24]+eps*tXi*x[29]+eps*tK*(x[19]-x[9]);

  dx[25]=-tk*(x[10]+x[20])-lambda*x[25];
  dx[26]=-tk*(x[11]+x[21])-lambda*x[26];
  dx[27]=-tk*(x[12]+x[22])-lambda*x[27];
  dx[28]=-tk*(x[13]+x[23])-lambda*x[28];
  dx[29]=-tk*(x[14]+x[24])-lambda*x[29];

}

void vfieldepsvar(double t,double *x,int ndim,double *dx){
  double a21,a26,a43,a46,ft;
  //Variational equations of the full systems with eps as parameter
  //(for continuation pursposes)
  //Note: this is not used for now. The spring is missing.

  ft=sin(omega*t);
  a21=0.5*(1.-3.*x[0]*x[0]);
  a26=-2.*tzeta*x[1]+tXi*x[4]+tF*ft;
  a43=0.5*(1.-3.*x[2]*x[2]);
  a46=-2.*tzeta*x[3]+tXi*x[4]+tF*ft;

  dx[0]=x[1];
  dx[1]=0.5*x[0]*(1-x[0]*x[0])+eps*(-2*tzeta*x[1]+tXi*x[4]+tF*ft);
  dx[2]=x[3];
  dx[3]=0.5*beta*x[2]*(1-x[2]*x[2])+eps*(-2*tzeta*x[3]+tXi*x[4]+tF*ft);
  dx[4]=-lambda*x[4]-tk*(x[1]+x[3]);
  dx[5]=0.;

  dx[6]=x[12];
  dx[7]=x[13];
  dx[8]=x[14];
  dx[9]=x[15];
  dx[10]=x[16];
  dx[11]=x[17];

  dx[12]=a21*x[6]-2.*eps*tzeta*x[12]+eps*x[30]+a26*x[36];
  dx[13]=a21*x[7]-2.*eps*tzeta*x[13]+eps*x[31]+a26*x[37];
  dx[14]=a21*x[8]-2.*eps*tzeta*x[14]+eps*x[32]+a26*x[38];
  dx[15]=a21*x[9]-2.*eps*tzeta*x[15]+eps*x[33]+a26*x[39];
  dx[16]=a21*x[10]-2.*eps*tzeta*x[16]+eps*x[34]+a26*x[40];
  dx[17]=a21*x[11]-2.*eps*tzeta*x[17]+eps*x[35]+a26*x[41];

  dx[18]=x[24];
  dx[19]=x[25];
  dx[20]=x[26];
  dx[21]=x[27];
  dx[22]=x[28];
  dx[23]=x[29];

  dx[24]=a43*x[18]-2.*eps*tzeta*x[24]+eps*tXi*x[30]+a46*x[36];
  dx[25]=a43*x[19]-2.*eps*tzeta*x[25]+eps*tXi*x[31]+a46*x[37];
  dx[26]=a43*x[20]-2.*eps*tzeta*x[26]+eps*tXi*x[32]+a46*x[38];
  dx[27]=a43*x[21]-2.*eps*tzeta*x[27]+eps*tXi*x[33]+a46*x[39];
  dx[28]=a43*x[22]-2.*eps*tzeta*x[28]+eps*tXi*x[34]+a46*x[40];
  dx[29]=a43*x[23]-2.*eps*tzeta*x[29]+eps*tXi*x[35]+a46*x[41];

  dx[30]=-tk*(x[12]+x[24])-lambda*x[30];
  dx[31]=-tk*(x[13]+x[25])-lambda*x[31];
  dx[32]=-tk*(x[14]+x[26])-lambda*x[32];
  dx[33]=-tk*(x[15]+x[27])-lambda*x[33];
  dx[34]=-tk*(x[16]+x[28])-lambda*x[34];
  dx[35]=-tk*(x[17]+x[29])-lambda*x[35];

  dx[36]=0.;
  dx[37]=0.;
  dx[38]=0.;
  dx[39]=0.;
  dx[40]=0.;
  dx[41]=0.;
}

void vfieldU(double t, double *x, int ndim, double *dx){
  //Unperturbed system u-v

  dx[0]=x[1];
  dx[1]=0.5*beta*x[0]*(1-x[0]*x[0]);
}

void vfieldX(double t, double *x, int ndim, double *dx){
  //Unperturbed systems x-y

  dx[0]=x[1];
  dx[1]=0.5*x[0]*(1-x[0]*x[0]);

}


void strobomap(double *x,double tf, void (*vfield)(double t, double *x, int ndim, double *dx),int ndim){
  double t=0.0;
  double h=hini;
  
  ini_rk78(ndim);
  while (t<tf){
   rk78(&t,x,&h,tol,hmin,hmax,ndim,vfield);
  }
  h=-(t-tf);
  rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfield);
  end_rk78(ndim);
}

/*
double periodporbit(double v0){
  double u1,sum,h,u,c;
  int n,i;

  c=v0*v0/2;

  n=100;
  u1=sqrt(1+sqrt(1+8*c/beta));
  h=u1/n;
  sum=0;
  u=0;

  for (i=0;i<n;i++){
    u=u+h;
    sum=sum+h*1/sqrt(v0*v0+u*u/2*beta*(1-u*u/2));
    cout <<sum<<endl;
  }
  return 4*sum;
}

*/

/*
double dperiodporbit(double v0){
  //We compute the derivative of alpha(v0) with respect to v0 numerically
  //We get a neighbourhood of v0, get some values of alpha and
  //interpolate.
  double h=1e-5;
  double alpha1,alpha3;
  //double alpha2;
  ini_rk78(2);
  alpha1=periodporbit(v0-h);
  //alpha2=periodporbit(v0);
  alpha3=periodporbit(v0+h);
  end_rk78(2);

  return (alpha3-alpha1)/(2.*h);
}
*/

double period4numderiv( double v0,void *params){
  //This is to work with numerical differentiation of gsl
  (void)(params);
  double val;

  val=periodporbit(v0);
  return val;
}

double dperiodporbit(double v0){
  //We compute the derivative of alpha(v0) with respect to v0 numerically using numerical differentiaiton of gsl
  gsl_function F;
  F.function=&period4numderiv;
  F.params=0;
  double result,abserr;

  gsl_deriv_central(&F,v0,0.1,&result,&abserr);

  return result;


}

/*
double dperiodporbit(double v0){
  //Compute the derivative of alpha(c)
  //
  //This is not working properly because the integral is undefined
 
  double sum=0,sumprev=1.;
  double uprev,fprev;
  double unext,fnext;
  double c=v0*v0/2.;
  double h,up,u;
  int numintervs,i,j,Maxiter=200;
  double IntTol=1e-10;
  
  u=sqrt(1.+sqrt(1.+8.*c/beta));
  up=beta*sqrt(1.+8.*c/beta)/(8.*u);

  numintervs=10;
  h=u/double(numintervs);
  for (i=0;i<numintervs;i++){
    uprev=double(i)*h;
    unext=double(i+1)*h;
    fprev=-1./pow(sqrt(2.*c+uprev*uprev/2.*(1.-uprev*uprev/2.)),3.);
    //fnext=-1./pow(sqrt(2.*c+unext*unext/2.*(1.-unext*unext/2.)),3.);
    sum=sum+h*fprev;
    //sum=sum+h*(fnext-fprev)/2.;
  }
  cout <<sum<<endl;
  sumprev=sum+100*IntTol;
  j=0;
  while (fabs(sum-sumprev)>IntTol && j<Maxiter){
    numintervs=2*numintervs;
    h=u/double(numintervs);
    sumprev=sum;
    for (i=0;i<numintervs;i++){
      uprev=double(i)*h;
      unext=double(i+1)*h;
      fprev=-1./pow(sqrt(2.*c+uprev*uprev/2.*(1.-uprev*uprev/2.)),3.);
      //fnext=-1./pow(sqrt(2.*c+unext*unext/2.*(1.-unext*unext/2.)),3.);
      sum=sum+h*fprev;
      //sum=sum+h*(fnext-fprev)/2.;
      cout <<sum<<endl;
    }
    j++;
  }
  if( j>=Maxiter){
    cout <<"Can't reach desired precision when computing alpha'(c)"<<endl;
    exit(1);
  }
  sum=4.*(sum+up/sqrt(2.*c+u*u/2.*(1-u*u/2.)));
}
*/


double periodporbit(double v0){
  //This computes the period of the periodic orbit going through
  //(0,v0) numerically
  int ndim=2;
  double *x=new double[ndim];
  double *dx=new double[ndim];
  double t=0.0;
  double h=hini;
  double Newtol=1e-12;
  int Maxiter=100;
  int i;
  x[0]=0;
  x[1]=v0;
  //The flow twists clockwise
  ini_rk78(ndim);
  while (x[1]>=0){
    rk78(&t,x,&h,tol,hmin,hmax,ndim,vfieldU);
  }
  //We start Newton
  i=0;
  while(fabs(x[1])>Newtol && i <Maxiter){
    //vfieldpert(t,x,ndim,dx);
    vfieldU(t,x,ndim,dx);
    h=-x[1]/dx[1];
    rk78(&t,x,&h,tol,fabs(h),fabs(h),ndim,vfieldU);
    i++;
  }
  end_rk78(ndim);
  delete [] x;
  delete[] dx;

  if (i>=Maxiter){
    cout<< "Newton failed computing the period. Exiting"<<endl;
    exit(1);
  }
  else{
    return(4*t);
  }
}

void clean_matrix(gsl_matrix *M){
  size_t i ,j;
  double zerotol=1e-10;

  for (i=0;i<M->size1;i++){
    for (j=0;j<M->size2;j++){
      if (fabs(gsl_matrix_get(M,i,j))<zerotol){
	gsl_matrix_set(M,i,j,0.);
      }
    }
  }
}

void uv2tauc(double *uv, double *tau, double *c){
  double h,t;
  int i,Maxiter=200;
  double Newtol=1e-12;
  double uprev;
  double alpha,dalpha;
  double *duv=new double[6];
  double *duv2=new double[6];
  double *uvaux=new double[6];
  int ndim=6;

  *c=uv[1]*uv[1]/2.-0.25*beta*uv[0]*uv[0]*(1.-uv[0]*uv[0]/2.);
  ini_rk78(ndim);
  t=0.;
  h=-hini;
  uprev=uv[0];
  for (i=0;i<2;i++){
    uvaux[i]=uv[i];
  }
  uvaux[2]=uvaux[5]=1.;
  uvaux[3]=uvaux[4]=0.;
  rk78(&t,uvaux,&h,tol,hmin,hmax,ndim,vfieldUvar2);
  while (!(uvaux[0]*uprev<0 && uvaux[1]>0)){
    uprev=uvaux[0];
    rk78(&t,uvaux,&h,tol,hmin,hmax,ndim,vfieldUvar2);
  }
  while (fabs(uvaux[0])>Newtol && i<Maxiter){
    vfieldUvar2(t,uvaux,ndim,duv);
    h=-uvaux[0]/duv[0];
    rk78(&t,uvaux,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar2);
    i++;
  }
  end_rk78(ndim);
  if (i>=Maxiter){
    cout <<"Newton failed when computing tau"<<endl;
    exit(1);
  }
  alpha=periodporbit(sqrt(2.*(*c)));
  dalpha=dperiodporbit(sqrt(2.*(*c)));
  *tau=fabs(t)/alpha;
  vfieldUvar2(t,uv,ndim,duv);
  uv[4]=-duv[1];//\partial c/\partial u
  uv[5]=duv[0];//\partial c/\partial v
  uvaux[0]=0;
  uvaux[1]=sqrt(2.*(*c));
  vfieldUvar2(t,uvaux,ndim,duv2);//field at (0,sqrt(2*c))
  //uv[2]=-(uvaux[2]+duv[0]*(-*tau*dalpha/sqrt(2.*(*c)))*uv[4]);
  uv[2]=-(uvaux[2]+duv2[0]*(-*tau*dalpha/sqrt(2.*(*c)))*(-duv[1]));
  uv[2]=uv[2]/(-duv2[0]*alpha);//\partial \tau/\partial u
  //uv[3]=-(uvaux[3]+duv[0]*(-*tau*dalpha/sqrt(2.*(*c)))*uv[5]);
  uv[3]=-(uvaux[3]+duv2[0]*(-*tau*dalpha/sqrt(2.*(*c)))*duv[0]);
  uv[3]=uv[3]/(-duv2[0]*alpha);//\partial \tau/\partial v
  delete[] duv;
  delete[] duv2;
  delete[] uvaux;
}


void tauc2uv(double *uv,double tau, double c){
  //uv[2]-uv[5] contains the differential of uv with respect to tau c

  double alpha,dalpha,t,h;
  double *duv=new double[6];
  int ndim=2;
  alpha=periodporbit(sqrt(2.*c));
  dalpha=dperiodporbit(sqrt(2.*c));

  ndim=6;
  t=0;
  h=hini;
  uv[0]=0;
  uv[1]=sqrt(2.*c);
  uv[2]=1.;
  uv[3]=0.;
  uv[4]=0.;
  uv[5]=1.;
  ini_rk78(ndim);
  while (t<tau*alpha){
    rk78(&t,uv,&h,tol,hmin,hmax,ndim,vfieldUvar2);
  }
  h=-(t-tau*alpha);
  rk78(&t,uv,&h,tol,fabs(h),fabs(h),ndim,vfieldUvar2);
  end_rk78(ndim);

  vfieldUvar2(t,uv,ndim,duv);
  uv[2]=duv[0]*alpha;
  uv[3]=(duv[0]*dalpha*tau+uv[3])/sqrt(2.*c);
  uv[4]=duv[1]*alpha;
  uv[5]=(duv[1]*dalpha*tau+uv[5])/sqrt(2.*c);

  delete[] duv;
}

double cosangle(double *x1,double *x2,double *x3){
  //This returns the cosinus of the angle between the vectores defined by these
  //3 points. We only take into account coordinates x and y.
  double modv1=0;
  double modv2=0;
  double sproduct=0;
  int i;
  for (i=2;i<4;i++){
    modv1+=pow(x2[i]-x1[i],2.);
    modv2+=pow(x3[i]-x2[i],2.);
    sproduct+=(x2[i]-x1[i])*(x3[i]-x2[i]);
  }
  modv1=sqrt(modv1);
  modv2=sqrt(modv2);
  return fabs(sproduct/(modv1*modv2));
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

void write_data_in_file(double *theta, double *c,double **data, int ncols, string filename){
  int i,j,k;
  ofstream outfile;
  std:ostringstream s;
  s<<filename.c_str()<<"_"<<count<<".tna";
  filename=s.str();
  outfile.precision(20);
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
  outfile.precision(20);
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
  outfile.precision(20);
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
