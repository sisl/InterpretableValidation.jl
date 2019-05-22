## This file is adapted from the code: https://www.mathworks.com/matlabcentral/fileexchange/53796-truncated-normal-and-student-s-t-distribution-toolbox
using Distributions
using LinearAlgebra
using SpecialFunctions
using NLsolve
using DataStructures

# Computes logarithm of tail of Z~N(0,1) mitigating numerical roundoff errors
lnPhi(x) = -0.5 * x^2 - 0.69314718055994530941723212 +
            log(erfcx(x / 1.4142135623730950488016887242))

# Computes ln(P(a<Z<b)) where Z~N(0,1) very accurately for any 'a', 'b'
function lnNpr(a, b)
    pa, pb = lnPhi(abs(a)), lnPhi(abs(b))
    (a > 0) && return pa + log1p(-exp(pb - pa)) # case b > a > 0
    (b < 0) && return pb + log1p(-exp(pa - pb)) # case a < b < 0
    # case a < 0 < b
    log1p(-erfc(-a / 1.4142135623730950488016887242) / 2 -
           erfc(b / 1.4142135623730950488016887242) / 2)
end

# Sample random number from truncated normal distribution with l < x < u
trandn(l, u) = rand(Truncated(Normal(), l, u), 1)[1]

# Generates the proposals from the exponentially tilted sequential importance sampling pdf
# input: n is the number of samples
#        L is ?
#        l is the lower bound
#        u is the upper bound
#        mu is ?
# output:    'p', log-likelihood of sample
#             Z, random sample
function mvnrnd(n, L, l, u, mu)
    d = length(l)
    p, Z = 0, zeros(d,n)
    for k = 1:d
        # compute matrix multiplication L*Z)
        col = dropdims(L[k:k, 1:k]*Z[1:k, :], dims = 1)

        # compute limits of truncation
        tl = l[k] .- mu[k] .- col
        tu = u[k] .- mu[k] .- col

        # simulate N(mu,1) conditional on [tl,tu]
        Z[k,:] .= mu[k] .+ trandn.(tl, tu)

        # update likelihood ratio
        p = p .+ lnNpr.(tl, tu) .+ 0.5*mu[k]^2 .- mu[k]*Z[k,:]
    end
    p, Z
end

# Implements psi(x,mu). assumes scaled 'L' without diagonal
function psy(x, L, l, u, mu)
    d = length(u)

    # compute  ~l and ~u
    c = L * x
    nl = l - mu - c
    nu = u - mu - c
    sum(lnNpr.(nl,nu) .+ .5*mu.^2 .- x.*mu)
end

# implements gradient of psi(x) to find optimal exponential twisting
# assumes scaled 'L' with zero diagonal
function gradpsi(y, L, l, u)
    d = length(u)
    c = zeros(d,1)
    x, mu = deepcopy(c), deepcopy(c)
    x[1:(d-1)] = y[1:(d-1)]
    mu[1:(d-1)] = y[d:end]

    # compute now ~l and ~u
    c[2:d] = L[2:d,:]*x
    lt = l - mu - c
    ut = u - mu - c

    # compute gradients avoiding catastrophic cancellation
    w = lnNpr.(lt, ut)
    pl = exp.(-0.5*lt.^2 .- w)./sqrt(2*pi)
    pu = exp.(-0.5*ut.^2 .- w)./sqrt(2*pi)
    P = pl - pu

    # output the gradient
    dfdx = -mu[1:(d-1)] .+ (P'*L[:,1:(d-1)])'
    dfdm = mu - x + P
    grad = [dfdx; dfdm[1:(d-1)]]

    # Compute the jacbian
    lt[isinf.(lt)] .= 0
    ut[isinf.(ut)] .= 0
    dP = -P.^2 + lt.*pl .- ut.*pu # dPdm
    DL = repeat(dP, 1, d).*L
    mx = -Matrix{Float64}(I, d, d) + DL
    xx = L'*DL
    mx = mx[1:d-1,1:d-1]

    xx = xx[1:d-1,1:d-1]

    J = [xx mx'; mx diagm(0 => (1 .+ dP[1:d-1]))]

    grad, J
end


#  Computes permuted lower Cholesky factor L for Σ
#  by permuting integration limit vectors l and u.
#  Outputs perm, such that Σ(perm,perm)=L*L'.
#
# Reference:
#  Gibson G. J., Glasbey C. A., Elston D. A. (1994),
#  "Monte Carlo evaluation of multivariate normal integrals and
#  sensitivity to variate ordering",
#  In: Advances in Numerical Methods and Applications, pages 120--126
function cholperm(Σ_in, l, u)
    Σ = copy(Σ_in)
    d = length(l)
    perm, L, z = [1:d...], zeros(d,d), zeros(d,1)
    for j = 1:d
        pr = Inf*ones(d) # compute marginal prob.
        rd = j:d # search remaining dimensions
        D = diag(Σ)
        s = D[rd] .- sum(L[rd,1:j-1].^2, dims=2)
        s[s.<0] .= 1e-16
        s = sqrt.(s)

        tl = (l[rd] .- L[rd,1:j-1]*z[1:j-1])./s
        tu = (u[rd] .- L[rd,1:j-1]*z[1:j-1])./s

        pr[rd] = lnNpr.(tl,tu)

        # find smallest marginal dimension
        k = argmin(pr)

        # flip dimensions k-->j
        jk, kj = [j,k], [k,j]

        # update rows and cols of Σ
        Σ[jk, :] .= Σ[kj, :]
        Σ[:, jk] .= Σ[:, kj]

        # update only rows of L
        L[jk,:] .= L[kj,:]

        # update integration limits
        l[jk] .= l[kj]
        u[jk] .= u[kj]
        perm[jk] .= perm[kj] # keep track of permutation

        # construct L sequentially via Cholesky computation
        s = Σ[j,j] - sum(L[j,1:j-1].^2)

        if s<-0.01 error("Σ is not positive semi-definite") end
        if s < 0 s = 1e-16 end
        L[j,j] = sqrt(s)

        L[j+1:d,j:j] .= (Σ[j+1:d,j]-L[j+1:d,1:j-1]*L[j:j,1:j-1]')./L[j,j]

        # find mean value, z(j), of truncated normal:
        tl = (l[j].-L[j:j,1:j-1]*z[1:j-1])./L[j,j]
        tu = (u[j].-L[j:j,1:j-1]*z[1:j-1])./L[j,j]
        w = lnNpr.(tl,tu) # aids in computing expected value of trunc. normal

        @assert length(tl) == 1 && length(tu) == 1 && length(w) == 1
        z[j] = (exp(-.5*tl[1]^2 - w[1]) - exp(-5*tu[1]^2. - w[1]))/sqrt(2*pi)
    end
    return L, l, u, perm
end


# truncated multivariate normal generator
# simulates 'n' random vectors exactly/perfectly distributed
# from the d-dimensional N(0,Sig) distribution (zero-mean normal
# with covariance 'Sig') conditional on l<X<u;
# infinite values for 'l' and 'u' are accepted;
# output:   'd' times 'n' array 'rv' storing random vectors;
#
# * Example:
#  d=60;n=10^3;Sig=0.9*ones(d,d)+0.1*eye(d);l=(1:d)/d*4;u=l+2;
#  X=mvrandn(l,u,Sig,n);boxplot(X','plotstyle','compact')
# plot marginals
#
# * Notes: Algorithm may not work if 'Sig' is close to being rank deficient.
# See also: mvNcdf, mvNqmc, mvrorth
# For more help, see <a href="matlab:
# doc">Truncated Multivariate Student & Normal</a> documentation at the bottom.
function mvrandn(l, u, Σ, n)
    # basic input check
    d = length(l)
    if  length(u) != d || size(Σ) != (d,d) || any(l > u)
        error("l, u, and Σ have to match in dimension with u>l")
    end

    # Cholesky decomposition of matrix with permuation
    Lfull, l, u, perm = cholperm(Σ,l,u) # outputs the permutation
    D = diag(Lfull)
    if any(D .< 1e-16)
        warning("Method may fail as covariance matrix is singular!")
    end
    L = Lfull ./ repeat(D,1,d)

    u = u ./ D
    l = l ./ D #rescale
    L = L .- Matrix{Float64}(I, d, d) # remove diagonal

    # find optimal tilting parameter non-linear equation solver
    soln = nlsolve(only_fj((x) -> gradpsi(x, L, l, u)), zeros(2*(d-1),1))



    x = [soln.zero[1:(d-1)]..., 0]
    mu = [soln.zero[d:(2*d-2)]..., 0]

    # compute psi star
    psistar = psy(x, L, l, u, mu)

    # start acceptance rejection sampling
    rv, accept, iter = zeros(d, 0), 0, 0

    while accept < n #  while # of accepted is less than n
        logpr, Z = mvnrnd(n, L, l, u, mu) # simulate n proposals
        idx = -log.(rand(n)) .> (psistar .- logpr) #acceptance tests
        rv = hcat(rv, Z[:, idx]) #accumulate accepted
        accept = size(rv,2) #keep track of # of accepted
        iter = iter+1 #keep track of while loop iterations
        if iter > 1e3 # if iterations too large, seek approximation only
            accept = n
            rv = hcat(rv, Z) # add the approximate samples
            # println("WARNING: Sample is only approximately distributed.")
        end
    end
    # finish sampling postprocessing
    order = sortperm(perm)

    rv = rv[:, 1:n] # cut-down the array to desired n samples
    rv = Lfull*rv # reverse scaling of L
    rv = rv[order,:] # reverse the Cholesky permutation
end

function mvrandn_μ(μ, l, u, Σ, n)
    Y = mvrandn(l - μ, u - μ, Σ, n)
    return Y .+ μ
end

