import torch


def op_norm(mat,init):
    '''
    Compute spectral norm of linear operator
    Initialised with previous eigen vectors during training mode
    '''
    xtmp,val = init
    tol = 1e-12
    max_iter = 5000
    with torch.no_grad():
        if xtmp == None:
            xtmp = torch.randn(mat.shape[1]).to(torch.device('cuda'))
            xtmp = xtmp / torch.norm(xtmp)
            val = 1
        else:
            xtmp = xtmp.to(torch.device('cuda'))
        
        for k in range(max_iter):
            old_val = val
            xtmp = torch.matmul(mat.T, torch.matmul(mat, xtmp))
            val = torch.norm(xtmp)
            rel_val = torch.absolute(val - old_val) / old_val
            if rel_val < tol:
                break
            xtmp = xtmp / val
    return xtmp, torch.sqrt(val)


def op_norm_fct(fct, fct_star, init_size, init, end):
    '''
    Compute spectral norm of linear operator
    Initialised with previous eigen vectors during training mode
    '''
    tol = 1e-12
    max_iter = 500
    with torch.no_grad():
        if type(init_size) == list:
            if len(init_size) == 2:
                X,val = init
                xtmp,utmp = X

                if xtmp == None:
                    N,S = init_size[0],init_size[1]
                    xtmp, utmp = torch.randn(N).to(torch.device('cuda')), torch.randn(S).to(torch.device('cuda'))
                    val = torch.sqrt(torch.norm(xtmp)**2+torch.norm(utmp)**2)
                    xtmp = xtmp / val
                    utmp = utmp / val
                    val = 1
                for k in range(max_iter):
                    old_val = val
                    if end:
                        xtmp,utmp = fct_star(fct(xtmp,utmp))
                    else:
                        xtmp,utmp = fct_star(*fct(xtmp,utmp))
                    val = torch.sqrt(torch.norm(xtmp)**2+torch.norm(utmp)**2)
                    rel_val = torch.absolute(val - old_val) / old_val
                    if rel_val < tol:
                        break
                    xtmp = xtmp / val
                    utmp = utmp / val
                return ((xtmp,utmp), torch.sqrt(val))
            else:
                X,val = init
                xtmp,x_tmp,utmp = X
                
                if xtmp == None:
                    N,N,S = init_size[0],init_size[1],init_size[2]
                    xtmp,x_tmp, utmp = torch.randn(N).to(torch.device('cuda')), torch.randn(N).to(torch.device('cuda')), torch.randn(S).to(torch.device('cuda'))
                    val = torch.sqrt(torch.norm(xtmp)**2+torch.norm(x_tmp)**2+torch.norm(utmp)**2)
                    xtmp = xtmp / val
                    x_tmp = x_tmp / val
                    utmp = utmp / val
                    val = 1
                for k in range(max_iter):
                    old_val = val
                    if end:
                        xtmp,x_tmp,utmp = fct_star(fct(xtmp,x_tmp,utmp))
                    else:
                        xtmp,x_tmp,utmp = fct_star(*fct(xtmp,x_tmp,utmp))
                    val = torch.sqrt(torch.norm(xtmp)**2+torch.norm(x_tmp)**2+torch.norm(utmp)**2)
                    rel_val = torch.absolute(val - old_val) / old_val
                    if rel_val < tol:
                        break
                    xtmp = xtmp / val
                    x_tmp = x_tmp / val
                    utmp = utmp / val
                return ((xtmp,x_tmp,utmp), torch.sqrt(val))
        else:
            xtmp,val = init
                
            if xtmp == None:
                xtmp = torch.randn(init_size).to(torch.device('cuda'))
                val = torch.norm(xtmp)
                xtmp = xtmp / val
                val = 1
            for k in range(max_iter):
                old_val = val
                if end:
                    xtmp = fct_star(fct(xtmp))
                else:
                    xtmp = fct_star(*fct(xtmp))
                val = torch.norm(xtmp)
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val
            return (xtmp, torch.sqrt(val))
            





def Lips_cst(norm_W_n_i_,K):
    theta = torch.zeros(2*K+2).to(torch.device('cuda'))
    theta[-1] = 1
    for n in range(0,2*K+1):
        for i in range(0,n+1):
            theta[n] += theta[i-1] * norm_W_n_i_[n,i]
    return theta[2*K]/(2**(2*K))


def Lips_cte_Arthur(K,N,r,tau,sigma,L):
    #here we don't compute H
    #eigenvalues of H^*H inpainting :
    h = torch.zeros(N).to(torch.device('cuda'))
    h[:r] = torch.ones(r).to(torch.device('cuda'))

    #here we don't compute P

    #the tau_i are given by the NN,
    #same for sigma_i

    #The matrix L is s.t. L^*L and H^*H commute :
    #L = U Diagblock(B_1,B_2) P^*
    #L^*L = P Diagblock(B_1',B_2') P^*, B_1' and B_2' symmetric
    #The e.v. of L^*L are the one of B_1' and B_2'.
    #we don't have to explicit the eigen basis v_p !

    #indeed we have L^*L = PQ D (PQ)^*, with Q = Diagblock(Q_1,Q_2)
    # B_i_ = Q_i D_i Q_i^*
    #the basis (v_p) is given by PQ
    #we have well : H^*H = PQ J_r (PQ)^* = P(QJ_rQ^*)P^* = P Diagblock(Q_1 Q_1^*,(0)) P^* = P J_r P^*


    # #B_1,B_2 are given by the NN
    # B_1_ = torch.matmul(B_1.T,B_1)
    # B_2_ = torch.matmul(B_2.T,B_2)
    L_star_L = torch.matmul(L.T,L)


    #we compute the eigen values of L^*L :
    #can we do better ? B_i_ is symmetric and we just need its e. values, not e. vectors...
    # eigenvalue1, eigenvectors1 = torch.linalg.eig(B_1_)
    # eigenvalue2, eigenvectors2 = torch.linalg.eig(B_2_)
    l, eigenvectors1 = torch.linalg.eig(L_star_L)
    l = torch.relu(torch.real(l))


    #the ev are real non negative numbers
    # eigenvalue1 = torch.nn.functional.relu(eigenvalue1.real)
    # eigenvalue2 = torch.nn.functional.relu(eigenvalue2.real)

    # l = torch.zeros(N).to(torch.device('cuda'))
    # l[:r] = eigenvalue1
    # l[r:] = eigenvalue2

    #the ev of L^*L are non negative
    norm_L_star_L = torch.max(l)

    #now we compute the approximate Lipschitz constant with my formula :

    #we define a_{n,i,p},...,d_{n,i,p} :
    a = torch.zeros((K+1,K+1,N)).to(torch.device('cuda'))
    b = torch.zeros((K+1,K+1,N)).to(torch.device('cuda'))
    c = torch.zeros((K+1,K+1,N)).to(torch.device('cuda'))
    d = torch.zeros((K+1,K+1,N)).to(torch.device('cuda'))

    for i in range(0,K+1):
        a[i-1,i] = torch.ones(N).to(torch.device('cuda'))
        b[i-1,i] = torch.zeros(N).to(torch.device('cuda'))
        c[i-1,i] = torch.zeros(N).to(torch.device('cuda'))
        d[i-1,i] = torch.zeros(N).to(torch.device('cuda'))

    for i in range(1,K):
        a[i,i] = 1-tau[i] * h
        b[i,i] = -tau[i] * torch.ones(N).to(torch.device('cuda'))
        c[i,i] = sigma[i] * (1-2*tau[i] * h)
        d[i,i] = 2 * tau[i] * sigma[i] * torch.ones(N).to(torch.device('cuda'))

        for n in range(i,K-1):
            a[n+1,i] = (1-tau[n+1] * h) * a[n,i] - tau[n+1] * l * c[n,i]
            b[n+1,i] = (1-tau[n+1] * h) * b[n,i] - tau[n+1] * (l * d[n,i] + 1)
            c[n+1,i] = sigma[n+1] * (1-2*tau[n+1] * h) * a[n,i] + (1 - 2 * tau[n+1] * sigma[n+1] * l) * c[n,i]
            d[n+1,i] = sigma[n+1] * ((1-2*tau[n+1] * h) * b[n,i] - 2 * tau[n+1]) + (1 - 2 * tau[n+1] * sigma[n+1] * l) * d[n,i]
    #i = 0
    for n in range(K):
        a[n,0] = a[n,1]*(1-tau[0]*(h+l)) + b[n,1]*l*((sigma[0]+1)-2*sigma[0]*tau[0]*(h+l))
        c[n,0] = c[n,1]*(1-tau[0]*(h+l)) + (1+d[n,1]*l)*((sigma[0]+1)-2*sigma[0]*tau[0]*(h+l))

    #norm of a symmetric 2X2 matrix : (a & b \\ b & c)
    def norm_sym_matrix(alpha,beta,gamma):
        return 1/2 * (torch.abs(alpha+gamma) + torch.sqrt((alpha-gamma)**2 + 4 * beta**2)), 1/2 * (alpha+gamma + torch.sqrt((alpha-gamma)**2 + 4 * beta**2))


    def norm_cal_A(n,i):
        alpha = a[n,i]**2 + c[n,i]**2 * l
        beta = a[n,i] * b[n,i] + c[n,i] * (1+d[n,i] * l)
        gamma = d[n,i]*(d[n,i]*l+2) + b[n,i]**2
        norm_p, max_vp = norm_sym_matrix(alpha,beta,gamma)
        return torch.max(norm_p), torch.max(max_vp)

    def norm_cal_B(n,i):
        alpha = (a[n,i]+2*sigma[i-1]*b[n,i]*l)**2 + (sigma[i-1]*b[n,i]*l)**2 + b[n,i]**2 * l
        beta = (a[n,i] +2*sigma[i-1] * b[n,i] * l) * (c[n,i]+2*sigma[i-1]*(1+d[n,i]*l))+sigma[i-1]**2 * b[n,i] * l * (1+d[n,i]*l) + b[n,i] * (1+l * d[n,i])
        gamma = (c[n,i]+2*sigma[i-1]*(1+d[n,i]*l))**2 + sigma[i-1]**2 * (1+d[n,i] * l)**2+ 2*d[n,i] + d[n,i]**2 * l
        norm_p, max_vp = norm_sym_matrix(alpha,beta,gamma)
        return torch.max(norm_p), torch.max(max_vp)

    def norm_cal_C(n,i):
        alpha = a[n,i]**2 + ((1-tau[n+1] * h)*a[n,i] - tau[n+1] * c[n,i] * l)**2
        beta = ((1-tau[n+1] * h)*a[n,i] - tau[n+1] * c[n,i] * l) * ((1-tau[n+1] * h) * b[n,i] - tau[n+1] * (1 + l * d[n,i]))
        gamma = d[n,i]*(d[n,i]*l + 2) + ((1-tau[n+1] * h) * b[n,i] - tau[n+1] * (1 + l * d[n,i]))**2
        norm_p, max_vp = norm_sym_matrix(alpha,beta,gamma)
        return torch.max(norm_p), torch.max(max_vp)

    def norm_cal_D(n,i):
        mat = torch.zeros((3,3,N)).to(torch.device('cuda'))
        mat[0,0] = ((1- tau[n+1] *h) *a[n,i] -tau[n+1] *l* c[n,i]+ 2* sigma[i-1] *((1- tau[n+1]* h) *b[n,i] -tau[n+1]* (1 + l* d[n,i]))*l)**2 + a[n,i]**2 + 4*sigma[i-1]**2 * l * (1 + d[n,i] * l)**2
        mat[0,1] = - sigma[i-1]* ((1- tau[n+1]* h) *a[n,i] -tau[n+1]* l* c[n,i] + 2* sigma[i-1] * ((1- tau[n+1]* h)* b[n,i] -tau[n+1]* (1 + l* d[n,i]))*l) * ((1- tau[n+1]* h)* b[n,i] -tau[n+1]* (1 + l* d[n,i]))* l - 2 *sigma[i-1]**2 *(1+d[n,i] *l)**2 * l
        mat[1,0] = d[0,1]
        mat[0,2] = ((1- tau[n+1] *h)* a[n,i] -tau[n+1]* l* c[n,i] + 2* sigma[i-1]* ((1- tau[n+1]* h) *b[n,i] -tau[n+1]* (1 + l* d[n,i]))*l) * ((1- tau[n+1]* h)* b[n,i] -tau[n+1]* (1 + l* d[n,i])) +2 *sigma[i-1]* (1+d[n,i]*l)**2
        mat[2,0] = d[0,2]
        mat[1,1] = sigma[i-1]**2 *((1- tau[n+1]* h)* b[n,i] -tau[n+1]* (1 + l* d[n,i]))**2 * l**2 +sigma[i-1]**2 *l* (1 + d[n,i]* l)**2
        mat[1,2] = -sigma[i-1]* l* ((1- tau[n+1]* h) *b[n,i] -tau[n+1]* (1 + l* d[n,i])) * ((1- tau[n+1]* h) *b[n,i] -tau[n+1]* (1 + l* d[n,i])) -sigma[i-1]* (1+d[n,i]*l)**2
        mat[2,1] = d[1,2]
        mat[2,2] = ((1- tau[n+1]* h) *b[n,i] -tau[n+1]* (1 + l* d[n,i]))**2 + d[n,i] * (2+ l* d[n,i])
        norm_p = torch.zeros(1).to(torch.device('cuda'))
        max_vp = torch.zeros(1).to(torch.device('cuda'))
        for p in range(N):
            evalue, evector = torch.linalg.eigh(mat[:,:,p])
            norm_p, max_vp = max(norm_p, torch.max(torch.abs(evalue))), max(max_vp, torch.max(evalue))
        return norm_p, max_vp


    def maj(norm,max_vp):
        #return torch.sqrt(min(max(1+norm_L_star_L * norm, norm),max(one,norm_L_star_L * max_vp, norm_L_star_L, max_vp)))
        return torch.sqrt(max(1+norm_L_star_L * max_vp, max_vp))


    norm_W_n_i_arthur = torch.zeros((2*K+1,2*K+1))

    #i = 0 :
    #n = 0 :
    norm_W_n_i_arthur[0,0] = torch.sqrt(torch.max(torch.abs((1-(tau[0] * (h+l)))**2 + 1 + l)))
    
    #n >= 1 : 
    for n in range(1,2*K):
        if n%2 == 1:
            #n = 2*n_+1
            norm_W_n_i_arthur[n,0] = torch.max(torch.sqrt(a[(n-1)//2,0]**2+c[(n-1)//2,0]**2 * l))
        else:
            #n = 2(n_+1)
            norm_W_n_i_arthur[n,0] = torch.max(torch.sqrt(((1-tau[n//2]*h)*a[n//2-1,0] - tau[n//2]*l*c[n//2-1,0])**2 + a[n//2-1,0]**2 + c[n//2-1,0]**2 * l))
    #n = 2K
    norm_W_n_i_arthur[2*K,0] = torch.max(torch.abs(a[K-1,0]))


    for i in range(1,2*K):
        if i%2 == 0:
            #i = 2 * i_
            for n in range(i,2*K):
                if n % 2 == 1:
                    #n = 2n_+1
                    norm_cal_A_,max_vp = norm_cal_A((n-1)//2,i//2)
                    norm_W_n_i_arthur[n,i] = maj(norm_cal_A_,max_vp)
                else:
                    #n = 2(n_+1)
                    norm_cal_C_,max_vp = norm_cal_C(n//2 - 1,i//2)
                    norm_W_n_i_arthur[n,i] = maj(norm_cal_C_,max_vp)
            #n = 2*K
            norm_W_n_i_arthur[2*K,i] = torch.max(torch.sqrt(a[K-1,i//2]**2 + b[K-1,i//2]**2 * l))
        
        else:
            #i = 2 i_ -1
            for n in range(i,2*K):
                if n % 2 == 1:
                    #n = 2n_+1
                    norm_cal_B_,max_vp = norm_cal_B((n-1)//2,(i+1)//2)
                    norm_W_n_i_arthur[n,i] = maj(norm_cal_B_,max_vp)
                else:
                    #n = 2(n_+1)
                    norm_cal_D_,max_vp = norm_cal_D(n//2 - 1,(i+1)//2)
                    norm_W_n_i_arthur[n,i] = maj(norm_cal_D_,max_vp)
            #n = 2*K
            norm_W_n_i_arthur[2*K,i] = torch.max(torch.sqrt((a[K-1,(i+1)//2]**2 + 2*sigma[(i-1)//2]*b[K-1,(i+1)//2] * l)**2 + sigma[(i-1)//2]**2 * b[K-1,(i+1)//2]**2 * l**2 + b[K-1,(i+1)//2]**2 * l))

    #i = 2*K : 
    norm_W_n_i_arthur[2*K,2*K] = 1


    return Lips_cst(norm_W_n_i_arthur,K)





def Lips_cte_Classic(N, S, K, r, iava, L, tau, sigma):

    # H_star_H_ = Diag(e1,...,eN) where ek = 1 if k\in iava, ek = 0 otherwise.
    H_star_H = torch.zeros((N,N)).to(torch.device('cuda'))

    for i in range(r):
        H_star_H[iava[i],iava[i]] = 1


    #eigenvalues of H^*H inpainting :
    h = torch.zeros(N).to(torch.device('cuda'))
    h[:r] = torch.ones(r).to(torch.device('cuda'))



    #we choose a matrix L s.t. L^*L and H^*H commute :
    #L = U Diagblock(B_1,B_2) P^*
    #L^*L = P Diagblock(B_1',B_2') P^*, B_1' and B_2' symmetric

    L_star = L.T


    V_0_0 = torch.zeros(2*N+S,N).to(torch.device('cuda'))
    V_0_0[:N] = torch.eye(N).to(torch.device('cuda')) - tau[0] * (H_star_H + torch.matmul(L_star,L))
    V_0_0[N:2*N] = torch.eye(N).to(torch.device('cuda'))
    V_0_0[2*N:] = L

    V_0 = torch.zeros((K-1,2*N+S,N+S)).to(torch.device('cuda'))

    for k in range(K-1):
        V_0[k,:N,:N] = torch.eye(N).to(torch.device('cuda')) - tau[k+1] * H_star_H
        V_0[k,N:2*N,:N] = torch.eye(N).to(torch.device('cuda'))
        V_0[k,:N,N:] = - tau[k+1] * L_star
        V_0[k,2*N:,N:] = torch.eye(S).to(torch.device('cuda'))


    V_1 = torch.zeros((K,N+S,2*N+S)).to(torch.device('cuda'))
    for k in range(K):
        V_1[k,:N,:N] = torch.eye(N).to(torch.device('cuda'))
        V_1[k,N:,:N] = 2*sigma[k] * L
        V_1[k,N:,N:2*N] = - sigma[k] * L
        V_1[k,N:,2*N:] = torch.eye(S).to(torch.device('cuda'))


    Out = torch.zeros(N,N+S).to(torch.device('cuda'))
    Out[:,:N] = torch.eye(N).to(torch.device('cuda'))

    #now we compute the Lipschitz constant of the PNN :
    #notice that it depends a lot on the choose of L

    norm_W_n_i_classical = torch.zeros((2*K+1,2*K+1)).to(torch.device('cuda'))

    #i=0
    xtmp = None
    val = None

    #i = 0 : 
    W = V_0_0
    #n = 0 : 
    xtmp, val = op_norm(W,(xtmp, val))
    norm_W_n_i_classical[0,0] = val

    for n in range(1,2*K):
        if n % 2 == 0:
            W = torch.matmul(V_0[n//2 - 1],W)
            xtmp, val = op_norm(W,(xtmp, val))
            norm_W_n_i_classical[n,0] = val
        else:
            W = torch.matmul(V_1[n//2],W)
            xtmp, val = op_norm(W,(xtmp, val))
            norm_W_n_i_classical[n,0] = val

    #n = 2K : 
    W = torch.matmul(Out,W)
    xtmp, val = op_norm(W,(xtmp, val))
    norm_W_n_i_classical[2*K,0] = val


    for i in range(1,2*K):
        xtmp = None
        val = None
        if i % 2 == 0:
            #n = i : 
            W = V_0[i//2 - 1]
            xtmp, val = op_norm(W,(xtmp, val))
            norm_W_n_i_classical[i,i] = val
            for n in range(i+1,2*K):
                if n % 2 == 0:
                    W = torch.matmul(V_0[n//2 -1],W)
                    xtmp, val = op_norm(W,(xtmp, val))
                    norm_W_n_i_classical[n,i] = val
                else:
                    W = torch.matmul(V_1[n//2],W)
                    xtmp, val = op_norm(W,(xtmp, val))
                    norm_W_n_i_classical[n,i] = val
            #n = 2*K : 
            W = torch.matmul(Out,W)
            xtmp, val = op_norm(W,(xtmp, val))
            norm_W_n_i_classical[2*K,i] = val
        else:
            #n = i : 
            W = V_1[i//2]
            xtmp, val = op_norm(W,(xtmp, val))
            norm_W_n_i_classical[i,i] = val
            for n in range(i+1,2*K):
                if n % 2 == 0:
                    W = torch.matmul(V_0[n//2 -1],W)
                    xtmp, val = op_norm(W,(xtmp, val))
                    norm_W_n_i_classical[n,i] = val
                else:
                    W = torch.matmul(V_1[n//2],W)
                    xtmp, val = op_norm(W,(xtmp, val))
                    norm_W_n_i_classical[n,i] = val
            #n = 2K : 
            W = torch.matmul(Out,W)
            xtmp, val = op_norm(W,(xtmp, val))
            norm_W_n_i_classical[2*K,i] = val
        
    #i = 2K : 
    xtmp = None
    val = None
    W = Out
    xtmp, val = op_norm(W,(xtmp, val))
    norm_W_n_i_classical[2*K,2*K] = val


    return Lips_cst(norm_W_n_i_classical,K)



import copy


def Lips_cte_Classic_conv(N, S, K, iava, L, L_star, tau, sigma):

    def H_star_H(x):
        y = torch.zeros(x.shape).to(torch.device('cuda'))
        y[iava] = x[iava]
        return y

    def V_0_0(x):
        x_ = x.view(1,1,28,28)
        x__ = L_star(L(x_)).view(28*28)
        u = L(x_)
        return (x-tau[0]* (H_star_H(x) + x__), x, u)
    
    def V_0_0_star(x,x_,u):
        x__ = x.view(1,1,28,28)
        x__ = L_star(L(x__)).view(28*28)
        return x-tau[0]* (H_star_H(x) + x__) + x_ + L_star(u).view(28*28)
    

    def V_0(k,x,u):
        return (x-tau[k]* (H_star_H(x) + L_star(u).view(28*28)), x, u)

    def V_0_star(k,x,x_,u):
        u_ = x.view(1,1,28,28)
        u_ = L(u_)
        return (x-tau[k]* H_star_H(x) + x_, -tau[k] * u_+u)
    
    def V_1(k,x,x_,u):
        u_ = x.view(1,1,28,28)
        u_ = L(u_)
        u__ = x_.view(1,1,28,28)
        u__ = L(u__)
        return(x, 2*sigma[k] * u_- sigma[k] * u__+u)
    
    def V_1_star(k,x,u):
        return(x + 2*sigma[k] * L_star(u).view(28*28), - sigma[k] * L_star(u).view(28*28),u)

    def Out(x,u):
        return x

    def Out_star(x):
        return (x,torch.zeros(1,64,28,28).to(torch.device('cuda')))

    #now we compute the Lipschitz constant of the PNN :
    #notice that it depends a lot on the choose of L

    norm_W_n_i_classical = torch.zeros((2*K+1,2*K+1)).to(torch.device('cuda'))

    #i=0
    xtmp = None
    x_tmp= None
    utmp = None
    val = None

    #i = 0 : 
    W = V_0_0
    W_star = V_0_0_star
    #n = 0 :
    xtmp, val = op_norm_fct(W,W_star, N,(xtmp, val),0)
    norm_W_n_i_classical[0,0] = val

    for n in range(1,2*K):
        if n % 2 == 0:
            W_old_1 = copy.deepcopy(W)
            W_old_star_1 = copy.deepcopy(W_star)
            def W_new(x):
                x_,u = W_old_1(x)
                return V_0(n//2 - 1,x_,u)
            def W_new_star(x,x_,u):
                x__,u_ = V_0_star(n//2 - 1,x,x_,u)
                return W_old_star_1(x__,u_)
            W = copy.deepcopy(W_new)
            W_star = copy.deepcopy(W_new_star)

            xtmp, val = op_norm_fct(W,W_star, N,(xtmp, val),0)
            norm_W_n_i_classical[n,0] = val
        else:
            W_old_2 = copy.deepcopy(W)
            W_old_star_2 = copy.deepcopy(W_star)
            def W_new(x):
                x_,x__,u = W_old_2(x)
                return V_1(n//2,x_,x__,u)
            def W_new_star(x,u):
                x_,x__,u_ = V_1_star(n//2,x,u)
                return W_old_star_2(x_,x__,u_)
            W = copy.deepcopy(W_new)
            W_star = copy.deepcopy(W_new_star)

            xtmp, val = op_norm_fct(W,W_star, N,(xtmp, val),0)
            norm_W_n_i_classical[n,0] = val

    #n = 2K : 
    W_old_3 = copy.deepcopy(W)
    W_old_star_3 = copy.deepcopy(W_star)
    def W_new(x):
        x_,u = W_old_3(x)
        return x_
    def W_new_star(x):
        x_,u = Out_star(x)
        return W_old_star_3(x_,u)
    W = copy.deepcopy(W_new)
    W_star = copy.deepcopy(W_new_star)

    xtmp, val = op_norm_fct(W,W_star, N,(xtmp, val),1)
    norm_W_n_i_classical[2*K,0] = val

    for i in range(1,2*K):
        xtmp = None
        val = None
        if i % 2 == 0:
            #n = i : 
            def W_new(x,u):
                return V_0(i//2 - 1,x,u)
            def W_new_star(x,x_,u):
                return V_0_star(i//2 - 1,x,x_,u)
            W = copy.deepcopy(W_new)
            W_star = copy.deepcopy(W_new_star)
            X, val = op_norm_fct(W,W_star,[N,S],((xtmp,utmp), val),0)
            xtmp, utmp = X
            norm_W_n_i_classical[i,i] = val
            for n in range(i+1,2*K):
                if n % 2 == 0:
                    W_old_4 = copy.deepcopy(W)
                    W_old_star_4 = copy.deepcopy(W_star)
                    def W_new(x,u):
                        x_,u_ = W_old_4(x,u)
                        return V_0(n//2 -1,x_,u_)
                    def W_new_star(x,x_,u):
                        x__,u_ = V_0_star(n//2 -1,x,x_,u)
                        return W_old_star_4(x__,u_)
                    W = copy.deepcopy(W_new)
                    W_star = copy.deepcopy(W_new_star)
                    
                    X, val = op_norm_fct(W,W_star,[N,S],((xtmp,utmp), val),0)
                    xtmp, utmp = X
                    norm_W_n_i_classical[n,i] = val
                else:
                    W_old_5 = copy.deepcopy(W)
                    W_old_star_5 = copy.deepcopy(W_star)
                    def W_new(x,u):
                        x_,x__,u_ = W_old_5(x,u)
                        return V_1(n//2,x_,x__,u_)
                    def W_new_star(x,u):
                        x_,x__,u_ = V_1_star(n//2,x,u)
                        return W_old_star_5(x_,x__,u_)
                    W = copy.deepcopy(W_new)
                    W_star = copy.deepcopy(W_new_star)
                    
                    X, val = op_norm_fct(W,W_star,[N,S],((xtmp,utmp), val),0)
                    xtmp, utmp = X
                    norm_W_n_i_classical[n,i] = val
            #n = 2*K :
            W_old_6 = copy.deepcopy(W)
            W_old_star_6 = copy.deepcopy(W_star)
            def W_new(x,u):
                x_,u = W_old_6(x,u)
                return x_
            def W_new_star(x):
                x_,u = Out_star(x)
                return W_old_star_6(x_,u)
            W = copy.deepcopy(W_new)
            W_star = copy.deepcopy(W_new_star)
            
            X, val = op_norm_fct(W,W_star,[N,S],((xtmp,utmp), val),1)
            xtmp, utmp = X
            norm_W_n_i_classical[2*K,i] = val
        else:
            #n = i :
            def W_new(x,x_,u):
                return V_1(i//2,x,x_,u)
            def W_new_star(x,u):
                return V_1_star(i//2,x,u)
            W = copy.deepcopy(W_new)
            W_star = copy.deepcopy(W_new_star)
            
            X, val = op_norm_fct(W,W_star,[N,N,S],((xtmp,x_tmp,utmp), val),0)
            xtmp, x_tmp, utmp = X
            norm_W_n_i_classical[i,i] = val
            for n in range(i+1,2*K):
                if n % 2 == 0:
                    W_old_7 = copy.deepcopy(W)
                    W_old_star_7 = copy.deepcopy(W_star)
                    def W_new(x,x_,u):
                        x_,u_ = W_old_7(x,x_,u)
                        return V_0(n//2 -1,x_,u_)
                    def W_new_star(x,x_,u):
                        x__,u_ = V_0_star(n//2 -1,x,x_,u)
                        return W_old_star_7(x__,u_)
                    W = copy.deepcopy(W_new)
                    W_star = copy.deepcopy(W_new_star)
                    
                    X, val = op_norm_fct(W,W_star,[N,N,S],((xtmp, x_tmp, utmp), val),0)
                    xtmp, x_tmp, utmp = X
                    norm_W_n_i_classical[n,i] = val
                else:
                    W_old_8 = copy.deepcopy(W)
                    W_old_star_8 = copy.deepcopy(W_star)
                    def W_new(x,x_,u):
                        x_,x__,u_ = W_old_8(x,x_,u)
                        return V_1(n//2,x_,x__,u_)
                    def W_new_star(x,u):
                        x_,x__,u_ = V_1_star(n//2,x,u)
                        return W_old_star_8(x_,x__,u_)
                    W = copy.deepcopy(W_new)
                    W_star = copy.deepcopy(W_new_star)
                    
                    X, val = op_norm_fct(W,W_star,[N,N,S],((xtmp, x_tmp, utmp), val),0)
                    xtmp, x_tmp, utmp = X
                    norm_W_n_i_classical[n,i] = val
            #n = 2K :
            W_old_9 = copy.deepcopy(W)
            W_old_star_9 = copy.deepcopy(W_star) 
            def W_new(x,x_,u):
                x_,u = W_old_9(x,x_,u)
                return x_
            def W_new_star(x):
                x_,u = Out_star(x)
                return W_old_star_9(x_,u)
            W = copy.deepcopy(W_new)
            W_star = copy.deepcopy(W_new_star)
            
            X, val = op_norm_fct(W,W_star,[N,N,S],((xtmp, x_tmp, utmp), val),1)
            xtmp, x_tmp, utmp = X
            norm_W_n_i_classical[2*K,i] = val
        
    #i = 2K : 
    xtmp = None
    val = None
    W = Out
    norm_W_n_i_classical[2*K,2*K] = 1


    return Lips_cst(norm_W_n_i_classical,K)