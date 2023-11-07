import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import slogdet, pinv
from statsmodels.formula.api import ols
from statsmodels.tsa.arima.model import ARIMA


# ---------------------------------------------------
# reshape dataframes
# ---------------------------------------------------
def reshape_dataframe(df):
    """
    Reshape the dataframe into state-space form
    
    Args:
        df:           dataframe to be reshaped
        
    Return:
        df_trans:     N x T if endogenous, (number of exogenous * N) x T if exogenous
    """
    
    # add a new column that cumcounts the observations in one quarter
    df['n'] = df.groupby(df.index).cumcount()
    # convert the DataFrame from a wide format to a long format
    df_long = df.set_index([df.index, 'n']).stack(dropna=False)
    # convert it back into separate columns with "vertrag" values
    df_trans = df_long.unstack('period')
    
    df_trans.index = df_trans.index.set_names(["n", "variables"])
    
    return df_trans.fillna(0)

# ---------------------------------------------------
# generate the measurement matrix
# ---------------------------------------------------
def generate_measurement_matrix(X, N, time_length, n_invariant, n_variant):
    """
    Reshape the measurement matrix
    
    Args:
        X:            N x T, input matrix
        N:            scalar, number of observations in each period N
        time_length:  scalar
        n_invariant:  scalar, number of time-invariant entries
        n_variant:    scalar, number of time-variant entries
        
        
    Return: 
        Z:            N x K x T, design matrix
    """
    
    # constant part
    Z_constant = np.zeros((N, n_invariant, time_length))
    for t in range(time_length):
        Z_constant[:, :, t] = np.tile([1, 1], (N, 1))
    
    # time-varying part
    Z_time_varing = np.zeros((N, n_variant, time_length))
    for n in range(N):
        Z_time_varing[n] = X.loc[n]
    
    Z = np.concatenate((Z_constant, Z_time_varing), axis=1)
    
    return Z

# ---------------------------------------------------
# vec operation
# ---------------------------------------------------
def vec(matrix):
    """
    Turns a matrix into a column vector by vertically stacking the columns on top of each other.
    """
    
    column_vector = matrix.flatten(order="F").reshape(-1, 1)
    
    return column_vector

# ---------------------------------------------------
# kalman filter prediction-step
# ---------------------------------------------------
def kalman_filter_predict(a, P, T, R):
    """
    Args:
        a(t-1):    K x 1, estimated state mean at time t-1
        P(t-1):    K x K, estimated state covariance at time t-1
        T:         K x K, transition matrix
        R:         K x K, process noise covariance matrix
        
    Return:
        a(t|t-1):  K x 1, prior of state mean state at time t
        P(t|t-1):  K x K, prior of state covariance at time t
    """
    
    # priors
    a = np.dot(T, a)
    P = np.dot(T, np.dot(P, T.T)) + R
    
    return a, P

# ---------------------------------------------------
# kalman filter update-step
# ---------------------------------------------------
def kalman_filter_update(a, P, yt, Zt, H):
    """
    Args:
        a(t|t-1):   K x 1, prior state mean at time t
        P(t|t-1):   K x K, prior state covariance at time t
        yt:         N x 1, measurement vector at time t
        Zt:         N x K, measurement matrix at time t
        H:          N x N, measurement covariance matrix
        
    Return:
        a(t):       K x 1, posterior at time t
        P(t):       K x K, posterior at time t
        Kt:         K x N, Kalman Gain at time t
        logL_t:     scalar, loglikelihood at time t
    """
    
    # predicted y_t
    yt_pred = np.dot(Zt, a)
    # innovation
    vt = yt - yt_pred
    # Covariace matrix of the innovation
    Ft = np.dot(Zt, np.dot(P, Zt.T)) + H
    # Kalman Gain
    Kt = np.dot(P, np.dot(Zt.T, pinv(Ft)))
    
    # log-likelihood
    logL_t = (-0.5 * slogdet(Ft)[1]) + (-0.5 * np.dot(vt.T, np.dot(pinv(Ft), vt)))
    
    # posteriors
    a = a + np.dot(Kt, vt)
    P = P - np.dot(Kt, np.dot(Zt, P))
    
    return a, P, Kt, vt, Ft, logL_t[0]

# ---------------------------------------------------
# kalman filter
# ---------------------------------------------------
def kalman_filter(mu, Sig, T, R, Z, H, y):
    """
    Args:
        mu:           K x 1, initial state mean
        Sig:          K x K, initial state covariance
        T:            K x K, transition matrix
        R:            K x K, process noise covariance matrix
        Z:            N x K x T, measurement matrix
        H:            N x N, measurement covariance matrix
        Y:            N x T, measurement vector
    
    Return:
        a_pred:       list of K x 1 arrays, predicted state mean
        P_pred:       list of K x K arrays, predicted state covariance
        a_filt:       list of K x 1 arrays, filtered state mean
        P_filt:       list of K x K arrays, filtered state covariance
        W:            list of K x N arrays, Kalman Gain
        logL:         scalar, loglikelihood summed over t
    """
    
    a_pred = [None]
    P_pred = [None]
    a_filt = [mu]
    P_filt = [Sig]
    K = []
    F = []
    v = []
    logL = 0
    
    a = mu
    P = Sig
    
    for t in np.arange(y.shape[1]): 
        
        yt = y[:, t, np.newaxis]
        Zt = Z[:, :, t]
        
        # prediction step
        a, P = kalman_filter_predict(a, P, T, R)
        a_pred.append(a)
        P_pred.append(P)
        
        # update step
        a, P, Kt, vt, Ft, logL_t = kalman_filter_update(a, P, yt, Zt, H)
        
        # log-likelihood
        logL += logL_t
        
        # kalman gain, innovations, covariance matrix of innovations
        K.append(Kt)
        v.append(vt)
        F.append(Ft)
        
        a_filt.append(a)
        P_filt.append(P)
    
    return a_pred, P_pred, a_filt, P_filt, K, v, F, logL

# ---------------------------------------------------
# kalman smoother
# ---------------------------------------------------
def kalman_smoother(a_pred, P_pred, a_filt, P_filt, K, T, Z):
    """
    Args:
        a_pred:         list of K x 1 arrays, predicted state mean
        P_pred:         list of K x K arrays, predicted state covariance
        a_filt:         list of K x 1 arrays, filtered state mean
        P_filt:         list of K x K arrays, filtered state covariance
        W:              list of K x N arrays, Kalman Gain
        T:              K x K, transition matrix
        Z:              N x K x T, measurement matrix
    
    Return:
        a_smooth:     list of K x 1 arrays, smoothed state mean
        P_smooth:     list of K x K arrays, smoothed state covariance
        cov_lag1:     list of K x K arrays, smoothed lag one covariance P_{t-1, t-2 | T}
    """
    
    # initialize with the last filtered state
    a_smooth = [a_filt[-1]]
    P_smooth = [P_filt[-1]]
    # initialize
    J = []
    I = np.eye(a_filt[0].shape[0])
    cov_lag1 = [np.dot(np.dot((I - K[-1].dot(Z[:,:,-1])), T), P_filt[-2])]
    
    for t in range(len(a_filt)-2, -1, -1): # for T-1 -> 0 (85 -> 0)
        
        # smoothing gain
        Jt = np.dot(P_filt[t], np.dot(T.T, pinv(P_pred[t+1])))
        # smoothed estimates
        a_smooth_t = a_filt[t] + np.dot(Jt, (a_smooth[-1] - np.dot(T, a_filt[t])))
        P_smooth_t = P_filt[t] + np.dot(Jt, np.dot(P_smooth[-1] - P_pred[t+1], Jt.T))
        
        J.append(Jt)
        a_smooth.append(a_smooth_t)
        P_smooth.append(P_smooth_t)
    
    #J.reverse()
    
    for t in range(len(a_filt)-2, 0, -1): # T-1 -> 1 (85 -> 1)
        
        # smoothed lag one covariance
        cov_lag1_t = np.dot(P_filt[t], J[t-1].T) + np.dot(np.dot(J[t], (cov_lag1[-1] - T.dot(P_filt[t]))), J[t-1].T)
        cov_lag1.append(cov_lag1_t)
    
    # reverse the order of the lists 
    a_smooth.reverse()
    P_smooth.reverse()
    cov_lag1.reverse()
    
    return a_smooth, P_smooth, cov_lag1

# ---------------------------------------------------
# calculate ABC
# ---------------------------------------------------
def calculate_ABC(a_smooth, P_smooth, cov_lag1):
    """
    Args:
        a_smooth:     list of k x 1 arrays, smoothed state mean
        P_smooth:     list of k x k arrays, smoothed state covariance
        cov_lag1:     list of k x k arrays, smoothed lag one covariance P_{t-1, t-2 | T}
        
    Returns:
        A, B, C:      k x k array
    """

    A = 0
    B = 0
    C = 0
    
    for t in range(len(a_smooth)-1): # 0 -> T-1 = 0 -> 85
        
        A += P_smooth[t] + np.dot(a_smooth[t], a_smooth[t].T)
        B += cov_lag1[t] + np.dot(a_smooth[t+1], a_smooth[t].T)
        C += P_smooth[t+1] + np.dot(a_smooth[t+1], a_smooth[t+1].T)
    
    return A, B, C

# ---------------------------------------------------
# calculate auxiliary matrices
# ---------------------------------------------------
def calculate_auxiliary_matrices(T, R, H):
    
    # for matrix T
    f_T = vec(T)
    f_T[0, 0] = 0
    
    D_T = np.zeros((len(f_T), 1))
    D_T[0] = 1
    
    # for R
    D_R = vec(R)
    D_R[0] = 1
    
    # for matrix H
    D_H = vec(np.identity(H.shape[0]))
    
    return f_T, D_T, D_R, D_H

# ---------------------------------------------------
# maximization step
# ---------------------------------------------------
def maximization_step(a_smooth, P_smooth, cov_lag1, T, R, Z, H, y):
    """
    Args:
        a_smooth:       list of K x 1 arrays, smoothed state mean
        P_smooth:       list of K x K arrays, smoothed state covariance
        cov_lag1:       list of K x K arrays, smoothed lag one covariance P_{t-1, t-2 | T}
        R:              K x K array, transition noise covariance matrix
        
    Returns:
        mu_new:         K x 1 array, initial state mean
        Sig_new:        K x K array, initial state covariance
        T_new:          K x K array, transition matrix
        R_new:          K x K array, process noise covariance matrix
        H_new:          N x N array, measurement covariance matrix
    """
    
    T_new = T.copy()
    R_new = R.copy()
    H_new = H.copy()
    
    # calculate the necessary matrices
    A, B, C = calculate_ABC(a_smooth, P_smooth, cov_lag1)
    f_T, D_T, D_R, D_H = calculate_auxiliary_matrices(T, R, H)
    
    # maximizing mu
    mu_new = a_smooth[0]
    
    # maximizing Sigma
    Sig_new = P_smooth[0]
    
    # maximizing T
    vec_invR_B = vec(np.dot(pinv(R), B)) #36x1
    A_kr_invR = np.kron(A, pinv(R)) #36x36
    
    inversion_term = pinv(np.dot(D_T.T, A_kr_invR.dot(D_T))) #2x2
    vec_term = vec(np.dot(pinv(R), B)) - A_kr_invR.dot(f_T) #36x1
    Phi = np.dot(inversion_term.dot(D_T.T), vec_term) #2x1
    
    T_new[0, 0] = Phi[0, 0]
    
    # maximizing R
    sigma_nu_2 = 1/(len(a_smooth)-1) * (C - np.dot(B, T.T) - np.dot(T, B.T) + np.dot(T.dot(A), T.T))[0,0]
    R_new[0, 0] = sigma_nu_2
    
    # maximizing H
    sum_term = 0
    for t in range(len(a_smooth)-1): # 0 -> 85
        pi_t = y[:, t, np.newaxis] - np.dot(Z[:, :, t], a_smooth[t+1])
        ZPZ_t = np.dot(Z[:, :, t], P_smooth[t+1].dot(Z[:, :, t].T))
        sum_term += np.dot(pi_t, pi_t.T) + ZPZ_t
        
    sigma_omega_2 = np.dot(D_H.T, vec(sum_term))[0, 0] / ((len(a_smooth)-1) * y.shape[0])
    np.fill_diagonal(H_new, sigma_omega_2)
        
    return mu_new, Sig_new, T_new, R_new, H_new

# ---------------------------------------------------
# expectation step
# ---------------------------------------------------
def expectation_step(mu, Sig, T, R, Z, H, y):
    
    a_pred, P_pred, a_filt, P_filt, K, v, F, logL = kalman_filter(mu, Sig, T, R, Z, H, y)
    a_smooth, P_smooth, cov_lag1 = kalman_smoother(a_pred, P_pred, a_filt, P_filt, K, T, Z)
    
    return a_smooth, P_smooth, cov_lag1, v, F, logL

# ---------------------------------------------------
# em
# ---------------------------------------------------
def EM(initial_mu, initial_Sig, initial_T, initial_R, initial_H, Z, y, max_iterations=100, tol=0.01):
    
    mu_estimated = initial_mu
    Sig_estimated = initial_Sig
    T_estimated = initial_T
    R_estimated = initial_R
    H_estimated = initial_H
    
    best_mu = mu_estimated
    best_Sig = Sig_estimated
    best_T = T_estimated
    best_R = R_estimated
    best_H = H_estimated
    
    for it in range(max_iterations):
        
        # E-step
        a_smooth, P_smooth, cov_lag1, v, F, logL = expectation_step(mu_estimated, Sig_estimated, T_estimated, R_estimated, Z, H_estimated, y)
        if it == 0: 
            print(f'intinal log-likelihood: {logL}')
            prev_logL = logL - 0.1
        
        # M-step
        mu_estimated, Sig_estimated, T_estimated, R_estimated, H_estimated = maximization_step(a_smooth, P_smooth, cov_lag1, T_estimated, R_estimated, Z, H_estimated, y)
        
        if (logL - prev_logL) < tol:
            print('converged! wowho!')
            print(f': Final iteration: {it}')
            break
    
        best_mu = mu_estimated
        best_Sig = Sig_estimated
        best_T = T_estimated
        best_R = R_estimated
        best_H = H_estimated
        prev_logL = logL
        
    return best_mu, best_Sig, best_T, best_R, best_H, v, F