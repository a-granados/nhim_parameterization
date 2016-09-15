#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include "memoria.h"


int lu(double **a, int n, int *perm, double tol){

 int i,j,i2,j2,t,k;
 int m,q;
 double max,r;
 double *aux;
 m=0;

  for(j=0; j<n; j++){
	max=a[j][j];
	k=j;
	  for(t=j+1;t<n;t++){
		   if(fabs(a[t][j])>fabs(max)){
				max=a[t][j];
				k=t;
			  }
			}
		if (fabs(max)<tol){
		  return 0;
			  }

	   if(k!=j){
		q=perm[j];
		perm[j]=perm[k];
		perm[k]=q;

		aux=a[k];
		a[k]=a[j];
		a[j]=aux;

		m=m+1;
			}

		r=a[j][j];

		for(i2=j+1; i2<n ; i2++){
		a[i2][j]=a[i2][j]/r;
		for(j2=j+1; j2<n; j2++){
		a[i2][j2]=a[i2][j2]-(a[i2][j]*a[j][j2]);
				}
			  }
		   }

		if(fmod(m,2)==1){
		 return -1;
		  }
		else {
		 return 1;
		 }
	}


void resol(double **a, double *x, double *b, int n, int *perm){
   double *y;
   double *bp;
   int i, j;
   double l;
   double tol;

   bp=crear_vector(0,n-1);
   for(i=0; i<n; i++){
	  bp[i]=b[perm[i]];
	  }
   for(i=0; i<n; i++){
	  b[i]=bp[i];
	  }

   alliberar_vector(bp, 0);


   y=crear_vector(0, n-1);

   y[0]=b[0];
   for(i=1; i<n; i++){
	l=0;
	for(j=0; j<i; j++){
	  l=l+(a[i][j]*y[j]);
	  }
	y[i]=b[i]-l;
	}

   x[n-1]=y[n-1]/a[n-1][n-1];
   for(i=n-2; i>=0; i--){
	 l=0;
	 for(j=i+1; j<n; j++){
	  l=l+(a[i][j]*x[j]);
	  }
	 x[i]=(1/a[i][i])*(y[i]-l);
	 }

   alliberar_vector(y,0);

}









