#!apt install texlive-fonts-recommended texlive-fonts-extra cm-super dvipng


def FD2(func,x,h):
    return (-func(x-2.0*h)+16.0*func(x-h)-30.0*func(x)+16.0*func(x+h)-func(x+2.0*h))/(12.0*h**2)
if __name__ == "__main__":
    import numpy as np;
    import matplotlib;
    import matplotlib.pylab as plt;
    matplotlib.rcParams.update({'font.size':14,'font.family':'serif'})
    matplotlib.rcParams.update({'text.usetex':'true'})
    # create my function
    def myfunc(x):
        return np.exp(x);
    # point where I want to approx derivative
    x0 = 1.0;
    # exact derivative
    df2_exact = np.exp(x0);
    # approximate optimal spacing and minimal error
    eps = 2.0**(-52);
    M = myfunc(x0);
#   hopt = (144.0*eps/M)**(1/6);
    hopt = (45*eps/2*M)**(1/5);

    Eopt = (eps/hopt) + M*(hopt**4.0)/90.0;
    # numerical experiment
    num_tests = 1001;
    hvec = np.zeros(num_tests);
    evec = np.zeros(num_tests);
    for n in range(0,num_tests):
        hvec[n] = (1.05)**(1-n);
        evec[n] = np.abs(df2_exact - FD2(myfunc,x0,hvec[n]));
    # plot
    plt.figure(1)
    plt.clf()
    plt.grid()
    plt.loglog(hvec,evec,'b-')
    plt.title(r'Loglog error vs h for FD of $e^x$ at $x=1$')
    plt.xlabel(r'$h$: grid spacing');
    plt.ylabel('absolute error');
    plt.loglog(hopt,Eopt,'ro');
    #plt.show()
    plt.savefig('roundoff_example.pdf',format='pdf',bbox_inches='tight')

