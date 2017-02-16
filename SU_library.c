#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using std::cout;
//using std::cerr;
using std::endl;
//using std::setw;

using namespace std;
//#include "piezo_library.c"

void iterate_bundles2(double **thetac,double **Pvals,double **Kvals);
void iterate_bundles(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals,gsl_interp2d **invf,double **Pvals,double **Kvals);
void addnewpointsfiber(double **extrapoints,double ***pointsIM,int currentindex,int numpointsfiber, int numnewpoints);
void compute_fiber(double *bpoint,double *v,double deltaini,double deltaend,double ****pointsIM,int numit, int *numpointsfiber, int *maxnewpoints, int numpointsdomain,double maxangle,double maxdist, int stableorunstable);
void resample_segment_IM(double ***extrapoints, double *inipointNB,double *finalpointNB, double *inipointIM, double *finalpointIM,int finaliter,  double maxdist, double maxangle, int *numnewpoints,int maxnewpoints, int stableorunstable);
void introduce_element(double ***points,int n, int curpoint, double *newpoint);

void iterate_bundles2(double **thetac,double **Pvals,double **Kvals){
  //This routine obtains "the" stable and unstable manifolds of the NHIM.
  //We first obtain domains at the tangent spaces to W^s and W^u for
  //serveral values of theta and c. We compute a few iterations of
  //these values to generate manifolds close to the real stable and unstable
  //manifolds.
  //If, when iterating two consecutive points in the normal bundle,
  //they become to separated, then a resampling processes takes place
  //adding new points.
  //Each fiber will contain a different number of points depending on
  //its resampling needs. Hence, the fibers are interpolated using a
  //sort of length arc and homogeneously resampled for plotting
  //reasons.
  //All points are later saved in files in a proper format to plot the manifolds
  //in matlab using the plot_manifolds.m script.
  //
  //General parameters:
  double thetaini=0;
  double thetaend=1;
  int numpointstheta=1;//Number of points in the manifold. For each of them we compute approximations of the stable and stable manifold.
  int numpointsc=500;//Number of points in the manifold. For each of them we compute approximations of the stable and stable manifold.
  int numpointsstable=1;//Number of points to generate the tangent space to the stable manifold (2d)
  int numpointsdomain=100;//Number of points for the parameters parameterizing the manifolds
  int numit=8;//Number of iterations. It starts with 0.
  double deltaini=0.5e-6;//Distance at which we start moving the parameter parameterizing the tangent space to the invariant manifolds
  double deltaend=5e-3;//deltaend-deltaini determines the "length" of domain to be iterated
  //Resampling parameters:
  double maxdist=5e-2;//Maximal allowed distance between consecutive points in the leafs.
  double maxangle=10*pi/180;//Maximal allowed angle between three consecutive points in the fibers
  //Maximum number of points to be added in each resampling at iterate i.
  int i;
  int *maxnewpoints=new int[numit];
  maxnewpoints[0]=0;
  for (i=1;i<numit-1;i++){
    maxnewpoints[i]=10;
  }
  for (i=numit-2;i<numit;i++){
    maxnewpoints[i]=10000;
  }

  //const gsl_interp_type *Tfiber=gsl_interp_cspline;
  const gsl_interp_type *Tfiber=gsl_interp_cspline;
  int numpointsplot=1000;//Number of points per fiber that we want to user to plot in Matlab.

  //File string for writing:
  string filename;
  std:ostringstream s;

  int j,k,l,currentpoint,m,numnewpoints;
  int numpoints=numpointstheta*numpointsc;//Number of points in the manifold for which we compute the stable an unstable fibers.
  double curdist;
  int **numpointsUfiber=new int*[numpoints];
  double ****pointsUM=new double ***[numpoints];

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
  
  cout <<"  Iterating the unstable normal bundle."<<endl;
  //------------------------------------------------------------------------------------------
  //-------------Iteration of the unstable direction in the Normal bundle---------------------
  //------------------------------------------------------------------------------------------
  //We allocate temporary memory. This will be increased when
  //resampling is needed.
  for (j=0;j<numpoints;j++){
    pointsUM[j]=new double**[numit];
    numpointsUfiber[j]=new int[numit];
    for (k=0;k<numit;k++){
      pointsUM[j][k]=new double*[numpointsdomain];
      for (i=0;i<numpointsdomain;i++){
	pointsUM[j][k][i]=new double[5];
      }
      numpointsUfiber[j][k]=numpointsdomain;
    }
  }
  //#pragma omp parallel for private(i,j,k,l,currentindex)
  for (m=0;m<numpointstheta;m++){
    #pragma omp parallel for private(j,k,currentpoint)
    for (i=0;i<numpointsc;i++){
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      double *bpoint=new double[5];
      double *vu=new double[5];

      bpoint[0]=thetaini+double(m)*(thetaend-thetaini)/double(numpointstheta);
      if (numpointsc>1){
	bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpointsc-1);
      }
      else{
	bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpointsc);
      }
      for (j=2;j<5;j++){
	bpoint[j]=gsl_interp2d_eval_extrap(K[j],thetac[0],thetac[1],Kvals[j],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      }

      for (k=0;k<5;k++){
	vu[k]=gsl_interp2d_eval_extrap(P[2+3*k],thetac[0],thetac[1],Pvals[4+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      }
      currentpoint=i*numpointstheta+m;
      compute_fiber(bpoint,vu,deltaini,deltaend,&pointsUM[currentpoint],numit,numpointsUfiber[currentpoint],maxnewpoints,numpointsdomain,maxangle,maxdist,1);
      delete[] bpoint;
      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
      delete[] vu;
    }
  }
  cout<<"  Done."<<endl;

  //Interpolation of the unstable manifold:
  //For each point in the manifold we interpolate its unstable fiber. We use
  //arch parameter.
  //We then write data in files for Matlab plotting.
  //Careful, for now this only works for numpointstheta=1
  for (k=0;k<numit;k++){
    s.str("");
    s.clear();
    s<<"xfile-unstable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream xfile;
    xfile.precision(10);
    xfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"yfile-unstable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream yfile;
    yfile.precision(10);
    yfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"cfile-unstable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream cfile;
    cfile.precision(10);
    cfile.open(filename.c_str());
    for (i=0;i<numpoints;i++){
      gsl_interp **fiber=new gsl_interp*[3];
      for (j=0;j<3;j++){
	fiber[j]=gsl_interp_alloc(Tfiber,numpointsUfiber[i][k]);
      }
      gsl_interp_accel *arcacc=gsl_interp_accel_alloc();
      double *xfiber=new double[numpointsUfiber[i][k]];
      double *yfiber=new double[numpointsUfiber[i][k]];
      double *cfiber=new double[numpointsUfiber[i][k]];
      double *arcparam=new double[numpointsUfiber[i][k]];
      double fiberlength=0.;
      double dist=0;
      for (j=0;j<numpointsUfiber[i][k];j++){
	xfiber[j]=pointsUM[i][k][j][2];
	yfiber[j]=pointsUM[i][k][j][3];
	cfiber[j]=pointsUM[i][k][j][1];
	dist=0;
	if (j>0){
	  dist+=pow(pointsUM[i][k][j][2]-pointsUM[i][k][j-1][2],2.);
	  dist+=pow(pointsUM[i][k][j][3]-pointsUM[i][k][j-1][3],2.);
	  dist+=pow(pointsUM[i][k][j][4]-pointsUM[i][k][j-1][4],2.);
	  dist=sqrt(dist);
	}
	fiberlength+=dist;
	arcparam[j]=fiberlength;
      }
      gsl_interp_init(fiber[0],arcparam,xfiber,numpointsUfiber[i][k]);
      gsl_interp_init(fiber[1],arcparam,yfiber,numpointsUfiber[i][k]);
      gsl_interp_init(fiber[2],arcparam,cfiber,numpointsUfiber[i][k]);
      for (j=0;j<numpointsplot;j++){
	xfile<<gsl_interp_eval(fiber[0],arcparam,xfiber,double(j)*fiberlength/double(numpointsplot),arcacc)<<" ";
	yfile<<gsl_interp_eval(fiber[1],arcparam,yfiber,double(j)*fiberlength/double(numpointsplot),arcacc)<<" ";
	cfile<<gsl_interp_eval(fiber[2],arcparam,cfiber,double(j)*fiberlength/double(numpointsplot),arcacc)<<" ";
      }
      /*
      for (j=0;j<numpointsUfiber[i][k];j++){
	xfile<<arcparam[j]<<" "<<xfiber[j]<<" "<<yfiber[j]<<" "<<cfiber[j]<<" "<<pointsUM[i][k][j][0]<<" "<<pointsUM[i][k][j][1]<<endl;
      }
      */
      xfile<<endl;
      yfile<<endl;
      cfile<<endl;
      delete [] xfiber;
      delete[] yfiber;
      delete[] cfiber;
      delete [] arcparam;
      for (j=0;j<3;j++){
	gsl_interp_free(fiber[j]);
      }
      delete[] fiber;
      gsl_interp_accel_free(arcacc);
    }
    xfile.close();
    yfile.close();
    cfile.close();
  }

  for (j=0;j<numpoints;j++){
    for (i=0;i<numit;i++){
      for (k=0;k<numpointsUfiber[j][i];k++){
	delete[] pointsUM[j][i][k];
      }
      delete[] pointsUM[j][i];
    }
    delete[] pointsUM[j];
    delete[] numpointsUfiber[j];
  }
  delete[] pointsUM;
  delete[] numpointsUfiber;


  //------------------------------------------------------------------------------------------
  //-------------Iteration of the stable direction in the Normal bundle---------------------
  //------------------------------------------------------------------------------------------
  cout <<" Iterating the stable normal bundle."<<endl;
  //We allocate temporary memory. This will be increased when
  //resampling is needed.
  double *****pointsSM=new double****[numpoints];
  int ***numpointsSfiber=new int**[numpoints];
  for (j=0;j<numpoints;j++){
    pointsSM[j]=new double***[numpointsstable];
    numpointsSfiber[j]=new int*[numpointsstable];
    for (k=0;k<numpointsstable;k++){
      pointsSM[j][k]=new double**[numit];
      numpointsSfiber[j][k]=new int[numit];
      for (i=0;i<numit;i++){
	pointsSM[j][k][i]=new double*[numpointsdomain];
	for (m=0;m<numpointsdomain;m++){
	  pointsSM[j][k][i][m]=new double[5];
	}
	numpointsSfiber[j][k][i]=numpointsdomain;
      }
    }
  }
  //#pragma omp parallel for private(i,j,k,l,currentindex)
  for (m=0;m<numpointstheta;m++){
    #pragma omp parallel for private(j,k,currentpoint)
    for (i=0;i<numpointsc;i++){
      gsl_interp_accel *thetaaac=gsl_interp_accel_alloc();
      gsl_interp_accel *caac=gsl_interp_accel_alloc();
      double *bpoint=new double[5];
      double *vs1=new double[5];
      double *vs2=new double[5];
      double *vtmp=new double[5];
      double modvs1,modvs2;
      double Pivs1,Pivs2;

      bpoint[0]=thetaini+double(m)*(thetaend-thetaini)/double(numpointstheta);
      if (numpointsc>1){
	bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpointsc-1);
      }
      else{
	bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpointsc);
      }
      for (j=2;j<5;j++){
	bpoint[j]=gsl_interp2d_eval_extrap(K[j],thetac[0],thetac[1],Kvals[j],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      }

      modvs1=0.;
      modvs2=0.;
      for (k=0;k<5;k++){
	vs1[k]=gsl_interp2d_eval_extrap(P[3*k],thetac[0],thetac[1],Pvals[2+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
	vs2[k]=gsl_interp2d_eval_extrap(P[1+3*k],thetac[0],thetac[1],Pvals[3+k*5],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
	modvs1+=vs1[k]*vs1[k];
	modvs2+=vs2[k]*vs2[k];
      }
      modvs1=sqrt(modvs1);
      modvs2=sqrt(modvs2);
      currentpoint=i*numpointstheta+m;
      for (k=0;k<numpointsstable;k++){
	if (numpointsstable>1){
	  Pivs1=double(numpointsstable-1-k)/double(numpointsstable);
	  Pivs2=double(k)/double(numpointsstable);
	}
	else{
	  Pivs1=1.;
	  Pivs2=0.;
	}
	for (j=0;j<5;j++){
	  vtmp[j]=Pivs1*vs1[j]+Pivs2*vs2[j];
	}
	currentpoint=i*numpointstheta+m;
	compute_fiber(bpoint,vtmp,deltaini,deltaend,&pointsSM[currentpoint][k],numit,numpointsSfiber[currentpoint][k],maxnewpoints,numpointsdomain,maxangle,maxdist,0);
      }
      delete[] bpoint;
      gsl_interp_accel_free(thetaaac);
      gsl_interp_accel_free(caac);
      delete[] vtmp;
      delete[] vs1;
      delete[] vs2;
    }
  }
  
  cout <<" Done."<<endl;

  //Now we write data in a file. We use only the fiber given in the
  //direction of vs1.
  for (k=0;k<numit;k++){
    s.str("");
    s.clear();
    s<<"xfile-stable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream xfile;
    xfile.precision(10);
    xfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"yfile-stable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream yfile;
    yfile.precision(10);
    yfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"cfile-stable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream cfile;
    cfile.precision(10);
    cfile.open(filename.c_str());
    for (i=0;i<numpoints;i++){
      gsl_interp **fiber=new gsl_interp*[3];
      for (j=0;j<3;j++){
	fiber[j]=gsl_interp_alloc(Tfiber,numpointsSfiber[i][0][k]);
      }
      gsl_interp_accel *arcacc=gsl_interp_accel_alloc();
      double *xfiber=new double[numpointsSfiber[i][0][k]];
      double *yfiber=new double[numpointsSfiber[i][0][k]];
      double *cfiber=new double[numpointsSfiber[i][0][k]];
      double *arcparam=new double[numpointsSfiber[i][0][k]];
      double fiberlength=0.;
      double dist=0;
      for (j=0;j<numpointsSfiber[i][0][k];j++){
	xfiber[j]=pointsSM[i][0][k][j][2];
	yfiber[j]=pointsSM[i][0][k][j][3];
	cfiber[j]=pointsSM[i][0][k][j][1];
	dist=0;
	if (j>0){
	  dist+=pow(pointsSM[i][0][k][j][2]-pointsSM[i][0][k][j-1][2],2.);
	  dist+=pow(pointsSM[i][0][k][j][3]-pointsSM[i][0][k][j-1][3],2.);
	  dist+=pow(pointsSM[i][0][k][j][4]-pointsSM[i][0][k][j-1][4],2.);
	  dist=sqrt(dist);
	}
	fiberlength+=dist;
	arcparam[j]=fiberlength;
      }
      gsl_interp_init(fiber[0],arcparam,xfiber,numpointsSfiber[i][0][k]);
      gsl_interp_init(fiber[1],arcparam,yfiber,numpointsSfiber[i][0][k]);
      gsl_interp_init(fiber[2],arcparam,cfiber,numpointsSfiber[i][0][k]);
      for (j=0;j<numpointsplot;j++){
	xfile<<gsl_interp_eval(fiber[0],arcparam,xfiber,double(j)*fiberlength/double(numpointsplot),arcacc)<<" ";
	yfile<<gsl_interp_eval(fiber[1],arcparam,yfiber,double(j)*fiberlength/double(numpointsplot),arcacc)<<" ";
	cfile<<gsl_interp_eval(fiber[2],arcparam,cfiber,double(j)*fiberlength/double(numpointsplot),arcacc)<<" ";
      }
      /*
      for (j=0;j<numpointsSfiber[i][0][k];j++){
	xfile<<arcparam[j]<<" "<<xfiber[j]<<" "<<yfiber[j]<<" "<<cfiber[j]<<" "<<pointsSM[i][0][k][j][0]<<" "<<pointsSM[i][0][k][j][1]<<endl;
      }
      */
      xfile<<endl;
      yfile<<endl;
      cfile<<endl;
      delete [] xfiber;
      delete[] yfiber;
      delete[] cfiber;
      delete [] arcparam;
      for (j=0;j<3;j++){
	gsl_interp_free(fiber[j]);
      }
      delete[] fiber;
      gsl_interp_accel_free(arcacc);
    }
    xfile.close();
    yfile.close();
    cfile.close();
  }

  for (i=0;i<numpoints;i++){
    for (j=0;j<numpointsstable;j++){
      for (k=0;k<numit;k++){
	for (m=0;m<numpointsdomain;m++){
	  delete[] pointsSM[i][j][k][m];
	}
	delete[] pointsSM[i][j][k];
      }
      delete[] numpointsSfiber[i][j];
      delete[] pointsSM[i][j];
    }
    delete[] numpointsSfiber[i];
    delete[] pointsSM[i];
  }
  delete[] pointsSM;
  delete[] numpointsSfiber;


  delete[] maxnewpoints;
  for (i=0;i<15;i++){
    gsl_interp2d_free(P[i]);
  }
  delete[] P;
  for (i=0;i<5;i++){
    gsl_interp2d_free(K[i]);
  }
  delete [] K;
}

void iterate_bundles(double **thetac,double **fvals,gsl_interp2d **f,double **invfvals,gsl_interp2d **invf,double **Pvals,double **Kvals){
  //This function is depreciated, use version 2 instead, which is much more
  //memory efficient and faster.
  //
  //
  //This routine obtains "the" stable and unstable manifolds of the NHIM.
  //We first obtain domains at the tangent spaces to W^s and W^u for
  //serveral values of theta and c. We compute a few iterations of
  //these values to generate manifolds close to the real stable and unstable
  //manifolds.
  //Each iteration of the domain is later resampled to guarantee that points do
  //not split too much and we obtain a more homogeneous sampling.
  //All points are later saved in files in a proper format to plot the manifolds
  //in matlab using the plot_manifolds.m script.
  //
  //General parameters:
  double thetaini=0;
  double thetaend=1;
  int numpointstheta=1;//Number of points in the manifold. For each of them we compute approximations of the stable and stable manifold.
  int numpointsc=1000;//Number of points in the manifold. For each of them we compute approximations of the stable and stable manifold.
  int numpointsstable=1;//Number of points to generate the tangent space to the stable manifld (2d)
  int numpointsdomain=10;//Number of points for the parameters parameterizing the manifolds
  int numit=8;//Number of iterations. It starts with 0.
  double deltaini=1e-5;//Distance at which we start moving the parameter parameterizing the tangent space to the invariant manifolds
  double deltaend=5e-3;//deltaend-deltaini determines the "length" of domain to be iterated
  //Resampling parameters:
  double maxdist=1e-3;
  //int maxnewpoints=35000;//Maximum number of new points in each segment.
  //Maximum number of new points in each segment. Take into account that memory
  //is reserved for all iterates. The following allows to adapt the amount of
  //new samples at each iteration to avoid unnecessary memory reservation. Note
  //that the first iteration generally do not need resampling because the
  //dynmics is slow close to the NHIM. However, after a few iterations, the
  //iterates of the domains grow and need more resolution.
  //Adapt this to your needs.
  int i;
  int *maxnewpoints=new int[numit];
  maxnewpoints[0]=10;
  for (i=0;i<numit-1;i++){
    maxnewpoints[i]=1000000;
  }
  for (i=numit-1;i<numit;i++){
    maxnewpoints[i]=1000000;
  }
  //maxnewpoints[10]=2000000;

  int j,k,l,currentindex,m;
  int numpoints=numpointstheta*numpointsc;
  int *finaliter=new int[numpoints*(numpointsstable+1)*numpointsdomain];
  int *finaliterstable=new int[numpoints*numpointsstable*numpointsdomain];
  int *finaliterunstable=new int[numpoints*numpointsdomain];
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
  
  cout <<"  Iterating domains in the stable and unstable tangent spaces."<<endl;
  //#pragma omp parallel for private(i,j,k,l,currentindex)
  for (m=0;m<numpointstheta;m++){
    #pragma omp parallel for private(j,k,l,currentindex)
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
      if (numpointsc>1){
	bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpointsc-1);
      }
      else{
	bpoint[1]=csimini +double(i)*(csimend-csimini)/double(numpointsc);
      }
      for (j=2;j<5;j++){
	bpoint[j]=gsl_interp2d_eval_extrap(K[j],thetac[0],thetac[1],Kvals[j],modulo(bpoint[0],1.),bpoint[1],thetaaac,caac);
      }
      //------------------------------------------------------------------------------------------
      //-------------Iteration of the tangent space to the stable manifold:-----------------------
      //------------------------------------------------------------------------------------------
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
	    //-------use this to avoid iterating the base point (which belong to the
	    //manifold):----------
	    //iteratesstable[currentindex+l][0][j]+=double(l+1)/double(numpointsdomain)*(deltaend-deltaini)*vtmp[j]/modvtmp;
	    //------Use this to include the base point (for example to see the
	    //tangency)--------
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

      //------------------------------------------------------------------------------------------
      //-------------Iteration of the tangent space to the unstable manifold:-----------------------
      //------------------------------------------------------------------------------------------
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
	  //-------use this to avoid iterating the base point (which belong to the
	  //manifold):----------
	  //iteratesunstable[currentindex+l][0][j]+=double(l+1)/double(numpointsdomain)*(deltaend-deltaini)*vu[j]/modvu;
	  //------Use this to include the base point (for example to see the
	  //tangency)--------
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
  cout<<"  Done."<<endl;

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
  //for (i=0;i<numpointsc*numpointstheta*numpointsdomain;i++){
  for (i=0;i<numpoints;i++){
    for (m=0;m<numpointsdomain;m++){
      currentindex=i*numpointsdomain+m;
      for (j=0;j<finaliterunstable[currentindex];j++){
	funstable<<i<<" "<<m<<" "<<j<<" ";
	for (k=0;k<5;k++){
	  funstable<<iteratesunstable[currentindex][j][k]<<" ";
	}
	funstable<<endl;
      }
    }
  }
  funstable.close();

  //----------------------------------------------------------------------
  //-------------------RESAMPLING OF MANIFOLDS----------------------------
  //----------------------------------------------------------------------

  //We now resample those segments whose vertices are too far away one
  //another. When this occurs, we estimate the number of points we
  //need to add such that the distance between the new points is
  //below maxdist.
  //The number of resamples is decided checking only the leaf
  //c=csimini and theta=thetaini. Then, everything is reapeated for
  //the other values of theta and c.
  double curdist;

  //--------------------------------------------------------------------------------
  //------------------------Resampling the unstable manifold------------------------
  //--------------------------------------------------------------------------------
  cout <<"  Resampling the unstable manifold."<<endl;
  //New points will be contained here:
  int numresamples;
  double ****extraunstable=new double***[(numpointsdomain-1)*numpoints];
  for (i=0;i<(numpointsdomain-1)*numpoints;i++){
    extraunstable[i]=new double**[numit];
    for (j=0;j<numit;j++){
      extraunstable[i][j]=new double*[maxnewpoints[j]];
      for (k=0;k<maxnewpoints[j];k++){
	extraunstable[i][j][k]=new double[5];
      }
    }
  }
  //This contains the number of resamples added at each point segment
  int **numnewpointsunstable=new int*[numpointsdomain-1];
  for (i=0;i<numpointsdomain-1;i++){
    numnewpointsunstable[i]=new int[numit];
    for (j=0;j<numit;j++){
      numnewpointsunstable[i][j]=0;
    }
  }
  for (k=0;k<numit;k++){
    for (j=0;j<numpointsdomain-1;j++){
      double modvu;
      double *vu=new double[5];
      for (i=0;i<5;i++){
	curdist=pow(iteratesunstable[j+1][k][i]-iteratesunstable[j][k][i],2.);
      }
      curdist=sqrt(curdist);
      if (curdist>maxdist){
	//numnewpointsunstable[j][k]=int(curdist/maxdist)-1;
	numnewpointsunstable[j][k]=int(curdist/maxdist);
	if (numnewpointsunstable[j][k]>maxnewpoints[k]){
	  numnewpointsunstable[j][k]=maxnewpoints[k];
	}
	numresamples=1;
	while(curdist>maxdist && numresamples*numnewpointsunstable[j][k]<=maxnewpoints[k]){
	  numnewpointsunstable[j][k]=numresamples*numnewpointsunstable[j][k];
	  modvu=0.;
	  for (i=0;i<5;i++){
	    vu[i]=iteratesunstable[j+1][0][i]-iteratesunstable[j][0][i];
	    modvu+=vu[i]*vu[i];
	  }
	  modvu=sqrt(modvu);
	  curdist=modvu;
	  #pragma omp parallel for private(i)
	  for (l=0;l<numnewpointsunstable[j][k];l++){
	    double thetatmp,ctmp,xtmp,ytmp,wtmp,newinc;
	    //------Use this to avoid iterating the base point (which belongs to the-----
	    //manifold):----------
	    //newinc=double(l+1)*curdist/double(numnewpointsunstable[j][k]+1);
	    //------Use this to include the base point (for example to see the------
	    //tangency)--------
	    newinc=double(l)*curdist/double(numnewpointsunstable[j][k]+1);
	    thetatmp=iteratesunstable[j][0][0]+newinc*vu[0]/modvu;
	    ctmp=iteratesunstable[j][0][1]+newinc*vu[1]/modvu;
	    xtmp=iteratesunstable[j][0][2]+newinc*vu[2]/modvu;
	    ytmp=iteratesunstable[j][0][3]+newinc*vu[3]/modvu;
	    wtmp=iteratesunstable[j][0][4]+newinc*vu[4]/modvu;
	    extraunstable[j][k][l][0]=thetatmp;
	    extraunstable[j][k][l][1]=ctmp;
	    extraunstable[j][k][l][2]=xtmp;
	    extraunstable[j][k][l][3]=ytmp;
	    extraunstable[j][k][l][4]=wtmp;
	    for (i=0;i<k;i++){
	      F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,extraunstable[j][k][l]);
	      thetatmp=extraunstable[j][k][l][0];
	      ctmp=extraunstable[j][k][l][1];
	      xtmp=extraunstable[j][k][l][2];
	      ytmp=extraunstable[j][k][l][3];
	      wtmp=extraunstable[j][k][l][4];
	    }
	  }
	  //We now check that all distances between consecutive new points are below maxdist:
	  curdist=0;
	  for (l=0;l<5;l++){
	    curdist+=pow(extraunstable[j][k][1][l]-extraunstable[j][k][0][l],2.);
	  }
	  curdist=sqrt(curdist);
	  i=1;
	  while (curdist<=maxdist && i<(numnewpointsunstable[j][k]-1)){
	    curdist=0;
	    for (l=0;l<5;l++){
	      curdist+=pow(extraunstable[j][k][i+1][l]-extraunstable[j][k][i][l],2.);
	    }
	    curdist=sqrt(curdist);
	    i++;
	  }
	  numresamples++;
	}

	//We add numnewpointsunstable extra points to the rest of values of c
	#pragma omp parallel for private(l,i,curdist)
	for (m=1;m<numpointsc;m++){
	  double modvu;
	  double *vu=new double[5];
	  modvu=0.;
	  for (i=0;i<5;i++){
	    vu[i]=iteratesunstable[m*numpointsdomain+j+1][0][i]-iteratesunstable[m*numpointsdomain+j][0][i];
	    modvu+=vu[i]*vu[i];
	  }
	  modvu=sqrt(modvu);
	  curdist=modvu;
	  for (l=0;l<numnewpointsunstable[j][k];l++){
	    double thetatmp,ctmp,xtmp,ytmp,wtmp,newinc;
	    //-------use this to avoid iterating the base point (which belong to the
	    //manifold):----------
	    //newinc=double(l+1)*curdist/double(numnewpointsunstable[j][k]+1);
	    //------Use this to include the base point (for example to see the
	    //tangency)--------
	    newinc=double(l)*curdist/double(numnewpointsunstable[j][k]+1);
	    thetatmp=iteratesunstable[m*numpointsdomain+j][0][0]+newinc*vu[0]/modvu;
	    ctmp=iteratesunstable[m*numpointsdomain+j][0][1]+newinc*vu[1]/modvu;
	    xtmp=iteratesunstable[m*numpointsdomain+j][0][2]+newinc*vu[2]/modvu;
	    ytmp=iteratesunstable[m*numpointsdomain+j][0][3]+newinc*vu[3]/modvu;
	    wtmp=iteratesunstable[m*numpointsdomain+j][0][4]+newinc*vu[4]/modvu;
	    extraunstable[m*(numpointsdomain-1)+j][k][l][0]=thetatmp;
	    extraunstable[m*(numpointsdomain-1)+j][k][l][1]=ctmp;
	    extraunstable[m*(numpointsdomain-1)+j][k][l][2]=xtmp;
	    extraunstable[m*(numpointsdomain-1)+j][k][l][3]=ytmp;
	    extraunstable[m*(numpointsdomain-1)+j][k][l][4]=wtmp;
	    for (i=0;i<k;i++){
	      F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,extraunstable[m*(numpointsdomain-1)+j][k][l]);
	      thetatmp=extraunstable[m*(numpointsdomain-1)+j][k][l][0];
	      ctmp=extraunstable[m*(numpointsdomain-1)+j][k][l][1];
	      xtmp=extraunstable[m*(numpointsdomain-1)+j][k][l][2];
	      ytmp=extraunstable[m*(numpointsdomain-1)+j][k][l][3];
	      wtmp=extraunstable[m*(numpointsdomain-1)+j][k][l][4];
	    }
	  }
	  delete[] vu;
	}
      }
      delete[] vu;
    }
  }
  cout <<"  Done."<<endl;

  //For Matlab plotting:
  for (k=0;k<numit;k++){
    s.str("");
    s.clear();
    s<<"xfile-unstable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream xfile;
    xfile.precision(10);
    xfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"yfile-unstable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream yfile;
    yfile.precision(10);
    yfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"cfile-unstable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream cfile;
    cfile.precision(10);
    cfile.open(filename.c_str());
    for (i=0;i<numpoints;i++){
      for (j=0;j<numpointsdomain;j++){
	xfile<<iteratesunstable[i*numpointsdomain+j][k][2]<<" ";
	yfile<<iteratesunstable[i*numpointsdomain+j][k][3]<<" ";
	cfile<<iteratesunstable[i*numpointsdomain+j][k][1]<<" ";
	if (j<numpointsdomain-1){
	  for (l=0;l<numnewpointsunstable[j][k];l++){
	    xfile<<extraunstable[i*(numpointsdomain-1)+j][k][l][2]<<" ";
	    yfile<<extraunstable[i*(numpointsdomain-1)+j][k][l][3]<<" ";
	    cfile<<extraunstable[i*(numpointsdomain-1)+j][k][l][1]<<" ";
	  }
	}
      }
      xfile<<endl;
      yfile<<endl;
      cfile<<endl;
    }
    xfile.close();
    yfile.close();
    cfile.close();
  }

  for (i=0;i<(numpointsdomain-1)*numpoints;i++){
    for (j=0;j<numit;j++){
      for (l=0;l<maxnewpoints[j];l++){
	delete[] extraunstable[i][j][l];
      }
      delete[] extraunstable[i][j];
    }
    delete[] extraunstable[i];
  }
  delete[] extraunstable;
  for (i=0;i<numpointsdomain-1;i++){
    delete[] numnewpointsunstable[i];
  }
  delete[] numnewpointsunstable;


  //------------------------------------------------------------------------------
  //------------------------Resampling the stable manifold------------------------
  //------------------------------------------------------------------------------
  //For now this only supports the case numpointsstable=1: shotting along vs1.
  cout<<"  Resampling the Stable manifold"<<endl;
  //New points will be contained here:
  double ****extrastable=new double***[(numpointsdomain-1)*numpoints];
  for (i=0;i<(numpointsdomain-1)*numpoints;i++){
    extrastable[i]=new double**[numit];
    for (j=0;j<numit;j++){
      extrastable[i][j]=new double*[maxnewpoints[j]];
      for (k=0;k<maxnewpoints[j];k++){
	extrastable[i][j][k]=new double[5];
      }
    }
  }
  //This contains the number of resamples added at each point segment
  int **numnewpointsstable=new int*[numpointsdomain-1];
  for (i=0;i<numpointsdomain-1;i++){
    numnewpointsstable[i]=new int[numit];
    for (j=0;j<numit;j++){
      numnewpointsstable[i][j]=0;
    }
  }

  for (k=0;k<numit;k++){
    for (j=0;j<numpointsdomain-1;j++){
      curdist=pow(iteratesstable[j+1][k][2]-iteratesstable[j][k][2],2.);
      curdist+=pow(iteratesstable[j+1][k][3]-iteratesstable[j][k][3],2.);
      curdist=sqrt(curdist);
      double modvs1;
      double *vs1=new double[5];
      if (curdist>maxdist){//We resample
	//We first compute the number of new points.
	//numnewpointsstable[j][k]=int(curdist/maxdist)-1;//first estimate
	numnewpointsstable[j][k]=int(curdist/maxdist);//first estimate
	if (numnewpointsstable[j][k]>maxnewpoints[k]){
	  numnewpointsstable[j][k]=maxnewpoints[k];
	}
	numresamples=1;
	while(curdist>maxdist && numresamples*numnewpointsstable[j][k]<=maxnewpoints[k]){
	  numnewpointsstable[j][k]=numresamples*numnewpointsstable[j][k];
	  modvs1=0.;
	  for (i=0;i<5;i++){
	    vs1[i]=iteratesstable[j+1][0][i]-iteratesstable[j][0][i];
	    modvs1+=vs1[i]*vs1[i];
	  }
	  modvs1=sqrt(modvs1);
	  curdist=modvs1;
	  #pragma omp parallel for private(i)
	  for (l=0;l<numnewpointsstable[j][k];l++){
	    double thetatmp,ctmp,xtmp,ytmp,wtmp,newinc;
	    //-------use this to avoid iterating the base point (which belong to the
	    //manifold):----------
	    //newinc=double(l+1)*curdist/double(numnewpointsstable[j][k]+1);
	    //------Use this to include the base point (for example to see the
	    //tangency)--------
	    newinc=double(l)*curdist/double(numnewpointsstable[j][k]+1);
	    thetatmp=iteratesstable[j][0][0]+newinc*vs1[0]/modvs1;
	    ctmp=iteratesstable[j][0][1]+newinc*vs1[1]/modvs1;
	    xtmp=iteratesstable[j][0][2]+newinc*vs1[2]/modvs1;
	    ytmp=iteratesstable[j][0][3]+newinc*vs1[3]/modvs1;
	    wtmp=iteratesstable[j][0][4]+newinc*vs1[4]/modvs1;
	    extrastable[j][k][l][0]=thetatmp;
	    extrastable[j][k][l][1]=ctmp;
	    extrastable[j][k][l][2]=xtmp;
	    extrastable[j][k][l][3]=ytmp;
	    extrastable[j][k][l][4]=wtmp;
	    for (i=0;i<k;i++){
	      invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,extrastable[j][k][l]);
	      thetatmp=extrastable[j][k][l][0];
	      ctmp=extrastable[j][k][l][1];
	      xtmp=extrastable[j][k][l][2];
	      ytmp=extrastable[j][k][l][3];
	      wtmp=extrastable[j][k][l][4];
	    }
	  }
	  //We now check that all distances between consecutive new poins are below maxdist:
	  curdist=0;
	  for (l=0;l<5;l++){
	    curdist+=pow(extrastable[j][k][1][l]-extrastable[j][k][0][l],2.);
	  }
	  curdist=sqrt(curdist);
	  i=1;
	  while (curdist<=maxdist && i<(numnewpointsstable[j][k]-1)){
	    curdist=0;
	    for (l=0;l<5;l++){
	      curdist+=pow(extrastable[j][k][i+1][l]-extrastable[j][k][i][l],2.);
	    }
	    curdist=sqrt(curdist);
	    i++;
	  }
	  numresamples++;
	}
	//We add numnewpointsstable extra points
	#pragma omp parallel for private(l,i,curdist)
	for (m=1;m<numpointsc;m++){
	  double modvs1;
	  double *vs1=new double[5];
	  modvs1=0.;
	  for (i=0;i<5;i++){
	    vs1[i]=iteratesstable[m*numpointsdomain+j+1][0][i]-iteratesstable[m*numpointsdomain+j][0][i];
	    modvs1+=vs1[i]*vs1[i];
	  }
	  modvs1=sqrt(modvs1);
	  curdist=modvs1;
	  for (l=0;l<numnewpointsstable[j][k];l++){
	    double thetatmp,ctmp,xtmp,ytmp,wtmp,newinc;
	    //-------use this to avoid iterating the base point (which belong to the
	    //manifold):----------
	    newinc=double(l+1)*curdist/double(numnewpointsstable[j][k]+1);
	    //------Use this to include the base point (for example to see the
	    //tangency)--------
	    //newinc=double(l)*curdist/double(numnewpointsstable[j][k]+1);
	    thetatmp=iteratesstable[m*numpointsdomain+j][0][0]+newinc*vs1[0]/modvs1;
	    ctmp=iteratesstable[m*numpointsdomain+j][0][1]+newinc*vs1[1]/modvs1;
	    xtmp=iteratesstable[m*numpointsdomain+j][0][2]+newinc*vs1[2]/modvs1;
	    ytmp=iteratesstable[m*numpointsdomain+j][0][3]+newinc*vs1[3]/modvs1;
	    wtmp=iteratesstable[m*numpointsdomain+j][0][4]+newinc*vs1[4]/modvs1;
	    extrastable[m*(numpointsdomain-1)+j][k][l][0]=thetatmp;
	    extrastable[m*(numpointsdomain-1)+j][k][l][1]=ctmp;
	    extrastable[m*(numpointsdomain-1)+j][k][l][2]=xtmp;
	    extrastable[m*(numpointsdomain-1)+j][k][l][3]=ytmp;
	    extrastable[m*(numpointsdomain-1)+j][k][l][4]=wtmp;
	    for (i=0;i<k;i++){
	      invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,extrastable[m*(numpointsdomain-1)+j][k][l]);
	      thetatmp=extrastable[m*(numpointsdomain-1)+j][k][l][0];
	      ctmp=extrastable[m*(numpointsdomain-1)+j][k][l][1];
	      xtmp=extrastable[m*(numpointsdomain-1)+j][k][l][2];
	      ytmp=extrastable[m*(numpointsdomain-1)+j][k][l][3];
	      wtmp=extrastable[m*(numpointsdomain-1)+j][k][l][4];
	    }
	  }
	  delete[] vs1;
	}
      }
      delete[] vs1;
    }
  }
  cout <<"  Done."<<endl;

  //For Matlab plotting:
  for (k=0;k<numit;k++){
    s.str("");
    s.clear();
    s<<"xfile-stable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream xfile;
    xfile.precision(10);
    xfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"yfile-stable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream yfile;
    yfile.precision(10);
    yfile.open(filename.c_str());
    s.str("");
    s.clear();
    s<<"cfile-stable"<<"_"<<count<<"_"<<k<<".tna";
    filename=s.str();
    ofstream cfile;
    cfile.precision(10);
    cfile.open(filename.c_str());
    for (i=0;i<numpoints;i++){
      for (j=0;j<numpointsdomain;j++){
	xfile<<iteratesstable[i*numpointsdomain+j][k][2]<<" ";
	yfile<<iteratesstable[i*numpointsdomain+j][k][3]<<" ";
	cfile<<iteratesstable[i*numpointsdomain+j][k][1]<<" ";
	if (j<numpointsdomain-1){
	  for (l=0;l<numnewpointsstable[j][k];l++){
	    xfile<<extrastable[i*(numpointsdomain-1)+j][k][l][2]<<" ";
	    yfile<<extrastable[i*(numpointsdomain-1)+j][k][l][3]<<" ";
	    cfile<<extrastable[i*(numpointsdomain-1)+j][k][l][1]<<" ";
	  }
	}
      }
      xfile<<endl;
      yfile<<endl;
      cfile<<endl;
    }
    xfile.close();
    yfile.close();
    cfile.close();
  }

  for (i=0;i<(numpointsdomain-1)*numpoints;i++){
    for (j=0;j<numit;j++){
      for (l=0;l<maxnewpoints[j];l++){
	delete[] extrastable[i][j][l];
      }
      delete[] extrastable[i][j];
    }
    delete[] extrastable[i];
  }
  delete[] extrastable;
  for (i=0;i<numpointsdomain-1;i++){
    delete[] numnewpointsstable[i];
  }
  delete[] numnewpointsstable;

  delete[] maxnewpoints;


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

void compute_fiber(double *bpoint,double *v,double deltaini,double deltaend,double ****pointsIM,int numit, int *numpointsfiber, int *maxnewpoints, int numpointsdomain,double maxangle,double maxdist, int stableorunstable){
  //This function computes the stable/unstable leaf of the first numit
  //backards/forwards iterates of the base point given in bpoint.
  //The vector v gives the stable/unstable direction in the normal
  //bundle. Then we take numpointsdomain points in the direction of v at distance
  //between deltaini and deltaend. If deltaini=0 then the base point
  //will be part of the leaf.
  //These points are iterated numit times, and stored in pointsIM. At
  //each iteration, the obtained fiber is expanded so it becomes a
  //better approximation of the global fiber of the corresponding
  //iterated point by the inner dynamics.
  //At each iteration, some points are split dramatically. Hence a
  //resampling is needed to generate more points in the empty pieces.
  //This is done by imposing:
  //1) consecutive points are separated below maxdist
  //2) three consecutive points form an angle below maxangle
  //Finally, if stableorunstable=0, the stable leaf is computed
  //(backwards iterates), while if =1, then the unstable one is
  //computed by performing forward iterates.
  int k,l,j,lastindex;
  double thetatmp,ctmp,xtmp,ytmp,wtmp;
  double curdist;
  bool angleok;
  double *nextpoint=new double[5];
  double **extrapoints;
  double *Fthetax=new double[5];
  int numnewpoints;
  //This contains the indeces in pointsIM where the main domain points
  //are stored.
  int **domaindeces=new int*[numit];
  for (j=0;j<numit;j++){
    domaindeces[j]=new int[numpointsdomain];
    for (l=0;l<numpointsdomain;l++){
      domaindeces[j][l]=l;
    }
  }
  double modv=0;
  for (j=0;j<5;j++){
    modv+=pow(v[j],2.);
  }
  modv=sqrt(modv);
  //Note that, if deltini=0, the base point at the NHIM is included and
  //iterated
  for (l=0;l<numpointsdomain;l++){
    for (j=0;j<5;j++){
      (*pointsIM)[0][l][j]=bpoint[j];
      (*pointsIM)[0][l][j]+=double(l)/double(numpointsdomain)*(deltaend-deltaini)*v[j]/modv;
    }
  }


  for (k=1;k<numit;k++){
    thetatmp=(*pointsIM)[k-1][0][0];
    ctmp=(*pointsIM)[k-1][0][1];
    xtmp=(*pointsIM)[k-1][0][2];
    ytmp=(*pointsIM)[k-1][0][3];
    wtmp=(*pointsIM)[k-1][0][4];
    if (stableorunstable==0){
      invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax);
    }
    if (stableorunstable==1){ 
      F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,Fthetax);
    }
    (*pointsIM)[k][0][0]=Fthetax[0];
    (*pointsIM)[k][0][1]=Fthetax[1];
    (*pointsIM)[k][0][2]=Fthetax[2];
    (*pointsIM)[k][0][3]=Fthetax[3];
    (*pointsIM)[k][0][4]=Fthetax[4];
    domaindeces[k][0]=0;
    thetatmp=(*pointsIM)[k-1][domaindeces[k-1][1]][0];
    ctmp=(*pointsIM)[k-1][domaindeces[k-1][1]][1];
    xtmp=(*pointsIM)[k-1][domaindeces[k-1][1]][2];
    ytmp=(*pointsIM)[k-1][domaindeces[k-1][1]][3];
    wtmp=(*pointsIM)[k-1][domaindeces[k-1][1]][4];
    if (stableorunstable==0){
      invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,nextpoint);
    }
    if (stableorunstable==1){ 
      F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,nextpoint);
    }
    for (l=1;l<numpointsdomain;l++){
      lastindex=domaindeces[k][l-1];
      for (j=0;j<5;j++){
	(*pointsIM)[k][lastindex+1][j]=nextpoint[j];
      }
      //We check the distance:
      curdist=0;
      //Careful, distance is taken into account only in the x-y-w space, not
      //in inner coordinates
      for (j=2;j<5;j++){
	curdist+=pow((*pointsIM)[k][lastindex+1][j]-(*pointsIM)[k][lastindex][j],2.);
      }
      curdist=sqrt(curdist);
      //We check the angle.
      if (l<numpointsdomain-1){
	// compute next iterate to check the angle
	thetatmp=(*pointsIM)[k-1][domaindeces[k-1][l+1]][0];
	ctmp=(*pointsIM)[k-1][domaindeces[k-1][l+1]][1];
	xtmp=(*pointsIM)[k-1][domaindeces[k-1][l+1]][2];
	ytmp=(*pointsIM)[k-1][domaindeces[k-1][l+1]][3];
	wtmp=(*pointsIM)[k-1][domaindeces[k-1][l+1]][4];
	if (stableorunstable==0){//Stable manifold
	  invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,nextpoint);
	}
	if (stableorunstable==1){//Unstable manifold 
	  F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,nextpoint);
	}
	if (cosangle((*pointsIM)[k][lastindex],(*pointsIM)[k][lastindex+1],nextpoint)<cos(maxangle)){
	  angleok=false;
	}
	else {
	  angleok=true;
	}
      }
      else {
	angleok=true;
      }
      numnewpoints=2;
      if ((curdist>=maxdist || !angleok) && maxnewpoints[k]>0){
	resample_segment_IM(&extrapoints,(*pointsIM)[0][l-1],(*pointsIM)[0][l],(*pointsIM)[k][lastindex],(*pointsIM)[k][lastindex+1],k,maxdist,maxangle,&numnewpoints,maxnewpoints[k],stableorunstable);
	//Now we add the points in extrapoints into
	//pointsIM:
	addnewpointsfiber(extrapoints,&(*pointsIM)[k],lastindex,numpointsfiber[k],numnewpoints);
	numpointsfiber[k]+=numnewpoints-2;
	for (j=0;j<numnewpoints;j++){
	  delete[] extrapoints[j];
	}
	delete[] extrapoints;
      }
      domaindeces[k][l]=lastindex+1+numnewpoints-2;
    }
  }
  delete[] nextpoint;
  delete[] Fthetax;
  for (j=0;j<numit;j++){
    delete[] domaindeces[j];
  }
  delete[] domaindeces;
}

void resample_segment_IM(double ***extrapoints, double *inipointNB,double *finalpointNB, double *inipointIM, double *finalpointIM,int finaliter,  double maxdist, double maxangle, int *numnewpoints,int maxnewpoints, int stableorunstable){
  //This resamples the stable/unstable Invariant Manifolds between two given
  //points whose distance in the x-y-w space is larger than a tolerance or the
  //angle between three consecutive points is larger than maxangle (in radians).
  //We assume:
  //- hdeltaNM: this is the distance between the two corresponding points at the normal bundle
  //- v is the direction we are using at the normal bundle. v is
  //normalized!!! 
  //- stableorunstable=0 for stable manifold, 1 for the unstable one.
  //To simplify the algorithm, extrapoints includes both inipointIM
  //and finalpointIM in the first and last position.
  double mincosangle=cos(maxangle);//The normalized scalar product must be above this value.
  double curcosangle;

  int i,j,curextrapoint;
  double distprev,distnext;//Distances between the current new point and the previous and next one.
  double thetatmp,ctmp,xtmp,ytmp,wtmp;
  double **basepoints;
  double *newpoint=new double[5];
  *numnewpoints=2;
  (*extrapoints)=new double*[*numnewpoints];
  basepoints=new double*[*numnewpoints];
  for (i=0;i<*numnewpoints;i++){
    (*extrapoints)[i]=new double[5];
    basepoints[i]=new double[5];
  }
  for (i=0;i<5;i++){
    (*extrapoints)[0][i]=inipointIM[i];
    (*extrapoints)[1][i]=finalpointIM[i];
    basepoints[0][i]=inipointNB[i];
    basepoints[1][i]=finalpointNB[i];
  }
  //We assume current distance>maxist, so we start allocating memory
  //for one extra point (the middle one).
  curextrapoint=0;
  for (j=0;j<5;j++){
    newpoint[j]=basepoints[0][j]/2.+basepoints[1][j]/2.;
  }
  introduce_element(extrapoints,*numnewpoints,curextrapoint,newpoint);
  introduce_element(&basepoints,*numnewpoints,curextrapoint,newpoint);
  *numnewpoints=3;
  curextrapoint++;
  thetatmp=(*extrapoints)[curextrapoint][0];
  ctmp=(*extrapoints)[curextrapoint][1];
  xtmp=(*extrapoints)[curextrapoint][2];
  ytmp=(*extrapoints)[curextrapoint][3];
  wtmp=(*extrapoints)[curextrapoint][4];
  for (i=0;i<finaliter;i++){
    if (stableorunstable==0){//Stable Manifold
      invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,(*extrapoints)[curextrapoint]);
    }
    if (stableorunstable==1){//Unstable Manifold
      F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,(*extrapoints)[curextrapoint]);
    }
    thetatmp=(*extrapoints)[curextrapoint][0];
    ctmp=(*extrapoints)[curextrapoint][1];
    xtmp=(*extrapoints)[curextrapoint][2];
    ytmp=(*extrapoints)[curextrapoint][3];
    wtmp=(*extrapoints)[curextrapoint][4];
  }
  distprev=0;
  distnext=0;
  for (i=2;i<5;i++){
    distprev+=pow((*extrapoints)[curextrapoint-1][i]-(*extrapoints)[curextrapoint][i],2.);
    distnext+=pow((*extrapoints)[curextrapoint+1][i]-(*extrapoints)[curextrapoint][i],2.);
  }
  distprev=sqrt(distprev);
  distnext=sqrt(distnext);
  curcosangle=cosangle((*extrapoints)[curextrapoint-1],(*extrapoints)[curextrapoint], (*extrapoints)[curextrapoint+1]);
  if (distprev>=maxdist || distnext>=maxdist || curcosangle<mincosangle ){//We add a new extra point
    if (distprev>=maxdist || curcosangle<mincosangle  ){//Point will be added previous to the current one.
      curextrapoint+=-1;
    }
    for (j=0;j<5;j++){
      newpoint[j]=basepoints[curextrapoint][j]/2.+basepoints[curextrapoint+1][j]/2.;
    }
  }
  else {//Both distances to the next and previous points are below maxdist. So we are done because this is the first point addition.
    curextrapoint++;//This should avoid entering the next loop
  }
  while (curextrapoint<*numnewpoints-1 && *numnewpoints-2 <maxnewpoints){
    introduce_element(extrapoints,*numnewpoints,curextrapoint,newpoint);
    introduce_element(&basepoints,*numnewpoints,curextrapoint,newpoint);
    *numnewpoints=*numnewpoints+1;
    curextrapoint++;
    thetatmp=(*extrapoints)[curextrapoint][0];
    ctmp=(*extrapoints)[curextrapoint][1];
    xtmp=(*extrapoints)[curextrapoint][2];
    ytmp=(*extrapoints)[curextrapoint][3];
    wtmp=(*extrapoints)[curextrapoint][4];
    for (i=0;i<finaliter;i++){
      if (stableorunstable==0){//Stable Manifold
	invF_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,(*extrapoints)[curextrapoint]);
      }
      if (stableorunstable==1){//Unstable Manifold
	F_eps(thetatmp,ctmp,xtmp,ytmp,wtmp,(*extrapoints)[curextrapoint]);
      }
      thetatmp=(*extrapoints)[curextrapoint][0];
      ctmp=(*extrapoints)[curextrapoint][1];
      xtmp=(*extrapoints)[curextrapoint][2];
      ytmp=(*extrapoints)[curextrapoint][3];
      wtmp=(*extrapoints)[curextrapoint][4];
    }
    distprev=0;
    distnext=0;
    for (i=2;i<5;i++){
      distprev+=pow((*extrapoints)[curextrapoint-1][i]-(*extrapoints)[curextrapoint][i],2.);
      distnext+=pow((*extrapoints)[curextrapoint+1][i]-(*extrapoints)[curextrapoint][i],2.);
    }
    distprev=sqrt(distprev);
    distnext=sqrt(distnext);
    curcosangle=cosangle((*extrapoints)[curextrapoint-1],(*extrapoints)[curextrapoint], (*extrapoints)[curextrapoint+1]);
    if (distprev>=maxdist || distnext>=maxdist || curcosangle<mincosangle){//We add a new extra point
      if (distprev>=maxdist){//Point will be added previous to the current one.
	curextrapoint+=-1;
      }
      for (j=0;j<5;j++){
	newpoint[j]=basepoints[curextrapoint][j]/2.+basepoints[curextrapoint+1][j]/2.;
      }
    }
    else {//Both distances to the next and previous points are below maxdist and the angle is below the maximum. So we move on until we find some distance above the threshold.
      while (distnext<maxdist && curcosangle>=mincosangle && curextrapoint<*numnewpoints-2){
	curextrapoint++;
	distnext=0.;
	for (j=2;j<5;j++){
	  distnext+=pow((*extrapoints)[curextrapoint][j]-(*extrapoints)[curextrapoint+1][j],2.);
	}
	distnext=sqrt(distnext);
	curcosangle=cosangle((*extrapoints)[curextrapoint-1],(*extrapoints)[curextrapoint], (*extrapoints)[curextrapoint+1]);
      }
      if (distnext>=maxdist || curcosangle<mincosangle){//We stopped because we found two points that need resampling
	for (j=0;j<5;j++){
	  newpoint[j]=basepoints[curextrapoint][j]/2.+basepoints[curextrapoint+1][j]/2.;
	}
      }
      else {//We stopped because we finished checking all points and all are below maxdist.
	curextrapoint++;
      }
    }
  }
  //*numnewpoints=*numnewpoints-1;
  delete[] newpoint;
  for (i=0;i<*numnewpoints;i++){
    delete [] basepoints[i];
  }
  delete [] basepoints;
}

void introduce_element(double ***points,int n, int curpoint, double *newpoint){
  //Given n points, this adds newpoint between curpoint and curpoint+1.
  double **pointsaux=new double*[n];
  int i,j;
  int ndim=5;

  for (i=0;i<n;i++){
    pointsaux[i]=new double[ndim];
    for (j=0;j<ndim;j++){
      pointsaux[i][j]=(*points)[i][j];
    }
    delete []  (*points)[i];
  }
  delete [] *points;
  (*points)=new double*[n+1];
  for(i=0;i<=curpoint;i++){
    (*points)[i]=new double[ndim];
    for (j=0;j<ndim;j++){
      (*points)[i][j]=pointsaux[i][j];
    }
    delete [] pointsaux[i];
  }
  (*points)[curpoint+1]=new double[ndim];
  for (j=0;j<ndim;j++){
    (*points)[curpoint+1][j]=newpoint[j];
  }
  for (i=curpoint+1;i<n;i++){
    (*points)[i+1]=new double[ndim];
    for (j=0;j<ndim;j++){
      (*points)[i+1][j]=pointsaux[i][j];
    }
    delete[] pointsaux[i];
  }
  delete [] pointsaux;
}

void addnewpointsfiber(double **extrapoints,double ***pointsIM,int currentindex,int numpointsfiber, int numnewpoints){
  //This will add the points contained in extrapoints to the array
  //pointsIM. The total size of pointsIM is numpointsfiber, and will
  //be increased in numnewpoints-2 points. Note that the first and
  //last points in numnewpoints are already contained in
  //pointsIM[currentindex] and pointsIM[currentindex+1], respectively.
  //The rest of points from currentindex+numnewpoints to
  //numpointsfiber+numnewpoints are set to 0 because nothing important
  //is supposed to be ther at this point.
  int i,j;
  double **auxpoints=new double*[currentindex];
  for (i=0;i<currentindex;i++){
    auxpoints[i]=new double[5];
    for (j=0;j<5;j++){
    auxpoints[i][j]=(*pointsIM)[i][j];
    }
    delete [] (*pointsIM)[i];
  }
  for (i=currentindex;i<numpointsfiber;i++){
    delete[] (*pointsIM)[i];
  }
  delete[] (*pointsIM);

  (*pointsIM)=new double*[numpointsfiber+numnewpoints-2];//First and last points in extrapoints are already included in pointsIM
  for (i=0;i<currentindex;i++){
    (*pointsIM)[i]=new double[5];
    for (j=0;j<5;j++){
      (*pointsIM)[i][j]=auxpoints[i][j];
    }
    delete [] auxpoints[i];
  }
  delete[] auxpoints;
  for (i=0;i<numnewpoints;i++){
    (*pointsIM)[currentindex+i]=new double[5];
    for (j=0;j<5;j++){
      (*pointsIM)[currentindex+i][j]=extrapoints[i][j];
    }
    if (i>0){
      if (extrapoints[i][2]==extrapoints[i-1][2]){
	cout<<"Oju cagada: "<<extrapoints[i-1][2]<<" "<<extrapoints[i][2]<<endl;
      }
    }
  }
  for (i=currentindex+numnewpoints;i<numpointsfiber+numnewpoints-2;i++){
    (*pointsIM)[i]=new double[5];
    for (j=0;j<5;j++){
      (*pointsIM)[i][j]=0.;
    }
  }
}
