#include <stdio.h>
#include <math.h>
#include <stdlib.h>



double **crear_matriu (int nf1, int nf2, int nc1, int nc2)
{
 int i;
 double **m;
 m=(double **)malloc((nf2-nf1+1)*sizeof(double *));
 if (m==NULL)
   {
	printf("error en alocatar la matriu\n");
	exit(1);
   }
 m-=nf1;
 for (i=nf1; i<=nf2; i++)
   {
	m[i]=(double *)calloc(nc2-nc1+1, sizeof(double));
	if (m[i]==NULL)
	  {
	   printf("error en alocatar la fila %d de la matriu\n",i);
	   exit(1);
	  }
	m[i]-=nc1;
   }
 return m;
}





void alliberar_matriu (double ** m, int nf1, int nf2, int nc1)
{
 int i;
 for (i=nf2; i>=nf1; i--)
   {
	free(m[i]+nc1);
   }
 free(m+nf1);
}



double *crear_vector (int nc1, int nc2)
{
 double *v;
 v=(double *)calloc(nc2-nc1+1, sizeof(double));
 if (v==NULL)
   {
	printf("error en alocatar el vector\n");
	exit(1);
   }
 return (v-nc1);
}



void alliberar_vector (double *v, int nc1)
{
 free(v+nc1);
}

int *crear_vector_enter (int nc1, int nc2)
{
 int *v;
 v=(int *)calloc(nc2-nc1+1, sizeof(int));
 if (v==NULL)
   {
	printf("error en alocatar el vector\n");
	exit(1);
   }
 return (v-nc1);
}

void alliberar_vector_enter (int *v, int nc1)
{
 free(v+nc1);
}



