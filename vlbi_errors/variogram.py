#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
from pylab import *




def variogram(data):
    n = data.shape[0]
    x_coords, y_coords = mgrid[0:n,0:n]
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    #print x_coords_flat, y_coords_flat
    flat_data = data.flatten()
    std_data = std(data)
    var_data = var(data)
    mean_data = mean(data)
    print "Mean, Std & Variance"
    print mean_data, std_data, var_data
    #print flat_data
    #result = array(n*(n-1)/2)
    result = []
    distance = []
    for i in range(n):
        result.append((delete(flat_data,i) - flat_data[i])**2)
        distance.append(sqrt((delete(x_coords_flat,i) - x_coords_flat[i])**2 + (delete(y_coords_flat,i) - y_coords_flat[i])**2 ))
    result = array(result).flatten()
    distance = array(distance).flatten()

    #sorted_distance = sort(distance)

    unique_distance, unique_indexes = unique(distance,return_index=True)
    final_distances = [0.0]
    final_varys = [0.0]
    number_of_distances = [n*n]
    for unique_d in unique_distance:
        result_4unique_d = where(distance == unique_d)[0]
        len_4unique_d = len(where(distance == unique_d)[0])
        vary_4unique_d = sum(result[result_4unique_d])/(2.0*len_4unique_d)
        #print "DEBUG"
        #print unique_d, vary_4unique_d
        final_distances.append(unique_d)
        final_varys.append(vary_4unique_d)
        number_of_distances.append(len_4unique_d)
        print   len_4unique_d, unique_d, sum(result[result_4unique_d])/(2.0*len_4unique_d)

    print "DEBUG"

    print type(final_distances),type(final_varys), type(number_of_distances)
    print final_distances, number_of_distances

    final_distances = array(final_distances)
    final_varys = array(final_varys)
    number_of_distances = array(number_of_distances)
    print "Counting only big numbers"
    print sort(number_of_distances)[::-1]
    print final_distances[argsort(number_of_distances)][::-1]
    print "???????????????"
    #final_distances = final_distances[0:n+1]
    #final_varys = final_varys[0:n+1]
    #number_of_distances = number_of_distances[0:n+1]

    def fit_variogramm(final_distances, p0, y = "None", y_errors = ones(len(final_distances)), model = "gauss"):
        """Function to fit of isotropic variogramm models: gaussian & power:
        y - variogramm (list or array)
        x - distances (list or array)
        y_errors - optional errors of variogramm values
        \"p0\" - initial values of parameters
        """
        from scipy.optimize import leastsq

        def fitting_func_gauss(p0):
            return p0[0]*(1 - exp(-(final_distances/p0[1])**2))

        def fitting_func_power(p0):
            return p0[0]*(1 - exp(-(final_distances/p0[1])))

        def error_func_gauss(p0):
            return (y - fitting_func_gauss(p0))/y_errors

        def error_func_power(p0):
            return (y - fitting_func_power(p0))/y_errors

        if y == "None":
            return fitting_func_gauss(p0)

        else:
            p,cov,info,mesg,success = leastsq(error_func_gauss,p0,full_output = 1)
            chisq                   = sum((error_func_gauss(p))**2)
            dof                     = len(final_distances) - len(p)
            redchisq                = ('%.3f' % float(chisq/dof))
            print "DEBUG"
            print "chisq, len(final_distances), dof, redchisq"
            print chisq, len(final_distances), dof, redchisq

            return p, chisq, redchisq, fitting_func_gauss(p), final_varys - fitting_func_gauss(p)

    distance_4_plot = arange(0,max(final_distances),0.01,dtype=float32)
    #choosing initial values of pars. based on variogramm
    print final_distances
    print final_varys
    p0 = [var_data,0.3*(max(final_distances)-min(final_distances))]
    print "/////////////// "
    print "Initial pars. :"
    print 0.0,p0
    print "///////////////"

    #Give more weigths to small distnces
    sigma_varys = arange(len(final_varys), dtype=final_varys.dtype)
    sigma_varys[1:] = (final_distances[1:])**2
    sigma_varys[0] = sigma_varys[1]/2.0
    p1, chisq1, redchisq1, variogram1, residuals1 = fit_variogramm(final_distances, p0, y = final_varys, y_errors = sigma_varys)
    print p1

    plot(final_distances, final_varys, "ro")
    plot(distance_4_plot, fit_variogramm(distance_4_plot, p1, y = "None"))
    xlabel('distance', size=13)
    ylabel('Variogramm values', size=13)
    title(' Variogram fiting  w rchsq :'+str(redchisq1))
    savefig('Variogram_initial_fit.png')
    close()


    print "Calculating uncertaintes of variogram for second (weighted) fit"
    sigma_varys = sqrt(2.0*(variogram1)**2.0/number_of_distances)
    sigma_varys[0] = sigma_varys[1]
    print final_varys, sigma_varys

    p2, chisq2, redchisq2, variogram2, residuals2 = fit_variogramm(final_distances, p1, y = final_varys, y_errors = sigma_varys)

    errorbar(final_distances,final_varys,sigma_varys,fmt='.k',linewidth=0.5)
    plot(distance_4_plot, fit_variogramm(distance_4_plot, p2, y = "None"))
    xlabel('distance', size=13)
    ylabel('Variogramm values', size=13)
    title(' Variogram fiting  w rchsq :'+str(redchisq2))
    savefig('Variogram_weighted1_fit.png')
    close()

    print "Calculating uncertaintes of variogram for final (weighted) fit"
    sigma_varys = sqrt(2.0*(variogram2)**2.0/number_of_distances)
    sigma_varys[0] = sigma_varys[1]
    print "DEBUG DOUBLE"
    print final_distances
    print final_varys
    print sigma_varys
    print len(sigma_varys), shape(sigma_varys)


    p3, chisq3, redchisq3, variogram3, residuals3 = fit_variogramm(final_distances, p2, y = final_varys, y_errors = sigma_varys)
    print p3

    errorbar(final_distances,final_varys,sigma_varys,fmt='.k',linewidth=0.5)
    plot(distance_4_plot, fit_variogramm(distance_4_plot, p3, y = "None"))
    xlabel('distance', size=13)
    ylabel('Variogramm values', size=13)
    title(' Variogram fiting  w rchsq :'+str(redchisq3))
    savefig('Variogram_final_fit.png')
    close()

    print "Estimating covariance matrix C(d_{ij})"
    Cov_plane = arange(len(distance_4_plot),dtype=final_varys.dtype)
    Cov_plane = p3[0] - fit_variogramm(distance_4_plot, p3, y = "None")

    def Cov(k,m,n,output=""):
        i1 = k//n
        j1 = k - i1*n
        i2 = m//n
        j2 = m - i2*n
        d_ij = sqrt((i1-i2)**2+(j1-j2)**2)
        if output == "dist":
            return d_ij
        return p3[0] - fit_variogramm(d_ij, p3, y = "None")

    plot(distance_4_plot, Cov_plane)
    xlabel('distance', size=13)
    ylabel('Covariogramm values', size=13)
    title('Covariogramm ')
    savefig('Covariogram_.png')
    close()

    print "//////////////////////////////"
    print "Now building Covariance matrix"
    print "//////////////////////////////"
    Cov_matrix = zeros((n*n,n*n),dtype=float32)
    for i in range(n*n):
        for j in range(n*n):
            Cov_matrix[i,j] = Cov(i,j,n,output="")

    #for i in range(n*n):
    #    for k in range((i//n)*n,((i//n)+1)*n):
    #        for m in range(mod(i,n)*n,(mod(i,n)+1)*n):
    #            Cov_matrix[k,m] = Cov(i,(k-(i//n)*n)*n+(m-mod(i,n)*n),n,output="")
    #return Cov_matrix


    #Cov_matrix = matrix(Cov_matrix)
    print Cov_matrix
    print shape(Cov_matrix)

#    print "Trying to estimate covariance matrix explicitly"
#    cov_expl = zeros((n*n,n*n),dtype=float32)
#    for i in range(n*n):
#        print "BIG Cycle :",i
#        for j in range(n*n):
#            print "little :",j
#            d_ij = Cov(i,j,n,output = "dist")
#            print d_ij
#            print len(where(distance == d_ij))
#            cov_expl[i,j] = sum([flat_data[ii]*flat_data[jj] for ii in range(n*n) for jj in range(n*n) if Cov(ii,jj,n,output = "dist") == d_ij])/len(where(distance == d_ij)) - mean(data)**2
#

    try:
        L = cholesky(Cov_matrix)
    except LinAlgError:
        print "Matrix is not positive definite!"
        w,v=eig(Cov_matrix)
        A = dot(v,sqrt(diag(w)))

        data_newa = []
        mean_newa = []
        std_newa = []
        res_newa = []
        for i in xrange(1000):
            resa = dot(randn(n*n),A.transpose())
            data_newa.append(mean_data + resa)
            mean_newa.append(mean(mean_data + resa))
            std_newa.append(std(mean_data + resa))
            res_newa.append(resa)

        return array(data_newa),array(res_newa),array(mean_newa),Cov_matrix,A.transpose()


    w,v=eig(Cov_matrix)
    A = dot(v,sqrt(diag(w)))

    data_new = []
    data_newa = []
    mean_new = []
    mean_newa = []
    std_new = []
    std_newa = []
    res_new =[]
    res_newa = []
    for i in xrange(1000):
        res = dot(randn(n*n),L)
        resa = dot(randn(n*n),A.transpose())
        data_new.append(mean_data + res)
        data_newa.append(mean_data + resa)
        mean_new.append(mean(mean_data + res))
        mean_newa.append(mean(mean_data + resa))
        std_new.append(std(mean_data + res))
        std_newa.append(std(mean_data + resa))
        res_new.append(res)
        res_newa.append(resa)

    return array(data_new),array(res_new),array(mean_new),Cov_matrix,A.transpose()
