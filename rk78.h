double rk78(double *at, double x[], double *ah, double tol,double hmin,double hmax, int n,
				void (*deriv)(double t, double *x, int n, double *dx));

void ini_rk78(int n);
void end_rk78(int n);

