### Tersoff potential functions

function Tersoff_energy(x_input::IntervalArithmetic.Interval, parameters, connectivity)

    e = zero(eltype(x_input[1]))
    @unpack a, b, λ₁, λ₂, β, n, h, c, d = parameters

    ### Bond Energy with Tersoff Potential with Carbon parameters 
    x = transpose(reshape(x_input, :, 3))
    N = size(x, 2)

    ### Bond Energy with Tersoff Potential with Carbon parameters.

    ### Iterating over particles
    @inbounds for i = 1:N

        y = x[:, view(connectivity, i, :)]

        ### First Bond  
        @inbounds for m = 1:3

            index1 = mod(m, 3) + 1
            index2 = mod(index1, 3) + 1

            ### Computing neighbors
            vⱼ = view(y, :, m) - view(x, :, i)
            vₖ = view(y, :, index1) - view(x, :, i)
            vₗ = view(y, :, index2) - view(x, :, i)

            rⱼ = sqrt(dot(vⱼ, vⱼ))
            rₖ = sqrt(dot(vₖ, vₖ))
            rₗ = sqrt(dot(vₗ, vₗ))

            ### First neighbor
            nrmₖ = rⱼ * rₖ
            projₖ = dot(vⱼ, vₖ) / nrmₖ
            angleₖ = acos(projₖ)

            ### Second neighbor
            nrmₗ = rⱼ * rₗ
            projₗ = dot(vⱼ, vₗ) / nrmₗ
            angleₗ = acos(projₗ)


            ### Interactions in the bond order function
            r = (interval(1) + (interval(c) / interval(d))^2 - (interval(c)^2) / (interval(d)^2 + (interval(h) - cos(angleₖ))^2)) + (interval(1) + (interval(c) / interval(d))^2 - (interval(c)^2) / (interval(d)^2 + (interval(h) - cos(angleₗ))^2))

            fA = -interval(b) * exp(-interval(λ₂) * rⱼ)

            e = e + interval(a) * exp(-interval(λ₁) * rⱼ) + ((interval(1) + (interval(β) * r)^interval(n))^(-interval(1) / interval(2 * n))) * fA


        end

    end


    e = e * interval(0.5)



end

function Tersoff_energy(x_input, parameters, connectivity)

    e = zero(eltype(x_input[1]))
    @unpack a, b, λ₁, λ₂, β, n, h, c, d = parameters

    ### Bond Energy with Tersoff Potential with Carbon parameters 
    x = transpose(reshape(x_input, :, 3))
    N = size(x, 2)

    ### Bond Energy with Tersoff Potential with Carbon parameters.

    ### Iterating over particles
    @inbounds for i = 1:N

        y = x[:, view(connectivity, i, :)]

        ### First Bond  
        @inbounds for m = 1:3

            index1 = mod(m, 3) + 1
            index2 = mod(index1, 3) + 1

            ### Computing neighbors
            vⱼ = view(y, :, m) - view(x, :, i)
            vₖ = view(y, :, index1) - view(x, :, i)
            vₗ = view(y, :, index2) - view(x, :, i)

            rⱼ = sqrt(dot(vⱼ, vⱼ))
            rₖ = sqrt(dot(vₖ, vₖ))
            rₗ = sqrt(dot(vₗ, vₗ))

            ### First neighbor
            nrmₖ = rⱼ * rₖ
            projₖ = dot(vⱼ, vₖ) / nrmₖ
            angleₖ = acos(projₖ)

            ### Second neighbor
            nrmₗ = rⱼ * rₗ
            projₗ = dot(vⱼ, vₗ) / nrmₗ
            angleₗ = acos(projₗ)


            ### Interactions in the bond order function
            r = (1 + (c / d)^2 - (c^2) / (d^2 + (h - cos(angleₖ))^2)) + (1 + (c / d)^2 - (c^2) / (d^2 + (h - cos(angleₗ))^2))

            fA = -b * exp(-λ₂ * rⱼ)

            e = e + a * exp(-λ₁ * rⱼ) + ((1 + (β * r)^n)^(-1 / (2 * n))) * fA

        end

    end


    e = e * 0.5



end

function Tersoff(x_input, p)

    @unpack N, R, D, a, b, λ₁, λ₂, β, n, h, c, d = p

    ### Input handling
    x = transpose(reshape(x_input, N, 3))

    ### Bond Energy with Tersoff Potential with Carbon parameters. 

    e = zero(eltype(x[1]))

    ### Distances 
    dist = zeros(eltype(x[1]), N, N)
    pairwise!(dist, Euclidean(1e-14), x, dims=2)
    cutoff = dist .<= R + D

    ### Energy computation

    @inbounds for i ∈ 1:N

        @inbounds for j ∈ 1:N

            if cutoff[i, j] == 1 && i != j

                ### Bond order function

                B = zero(eltype(x[1]))
                ### Bond order
                @inbounds for k ∈ 1:N

                    if k != i && k != j

                        if cutoff[i, k] == 1

                            ### Angle
                            nrm = dist[i, j] * dist[i, k]
                            proj = dot(view(x, :, i) - view(x, :, k), view(x, :, i) - view(x, :, j)) / nrm
                            angle = acos(proj)

                            B = B + smooth_cutoff(dist[i, k], R, D) * (1 + (c / d)^2 - (c^2) / (d^2 + (h - cos(angle))^2))

                        end

                    end

                end

                e = e + smooth_cutoff(dist[i, j], R, D) * (a * exp(-λ₁ * dist[i, j]) + ((1 + (β * B)^n)^(-1 / (2 * n))) * -b * exp(-λ₂ * dist[i, j]))

            end

        end

    end


    e = e * 0.5

    return e

end


@inline function smooth_cutoff(r, R, D)
    T = typeof(r)
    if r >= R - D
        return (one(T)/2) * (one(T) - sin(T(π) * (r - R) / (2D)))
    else
        return one(T)
    end
end


# function smooth_cutoff(r, R, D)


#     if R - D <= r
#         return 0.5 * (1 - sin(π * (r - R) / (2 * D)))
#     else
#         return 1.0
#     end

# end

function Energy_Grad_Tersoff(x, parameters, connectivity)

    e = Tersoff_energy(x, parameters, connectivity)

    g = ForwardDiff.gradient(x_input -> Tersoff_energy(x_input, parameters, connectivity), x)

    return e, g

end

function Hess_Tersoff(x, parameters, connectivity)


    ForwardDiff.hessian(x_input -> Tersoff_energy(x_input, parameters, connectivity), x)


end

function extended_Grad_Tersoff(x_input, x_fix, parameters, connectivity)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing vector field
    _, Fx = Energy_Grad_Tersoff(x, parameters, connectivity)

    ### Extending function
    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]

    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]

    ### Second alternative
    Fx = reshape(Fx, 3 * N, 1) + mu1 * T1 + mu2 * T2 + mu3 * T3 + lambda1 * I1x + lambda2 * I2x + lambda3 * I3x

    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]


end

function extended_Hess_Tersoff(x_input, x_fix, parameters, connectivity)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing Fx
    ### Computing vector field
    _, Fx = Energy_Grad_Tersoff(x, parameters, connectivity)

    ### Extending function
    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]

    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]


    ### Second alternative
    Fx = reshape(Fx, 3 * N, 1) + mu1 * T1 + mu2 * T2 + mu3 * T3 + lambda1 * I1x + lambda2 * I2x + lambda3 * I3x


    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]

    x = reshape(x, :, 3)

    ### Prelocating Matrices

    dFx = Hess_Tersoff(x, parameters, connectivity)


    ### Extending derivative
    dFx = [zeros(type, 3 * N, 6) dFx]
    dFx = [zeros(type, 6, 3 * N + 6); dFx]

    Adjusts = [zeros(type, N, N) -lambda1*Diagonal(ones(type, N)) lambda3*Diagonal(ones(type, N)); lambda1*Diagonal(ones(type, N)) zeros(type, N, N) -lambda2*Diagonal(ones(type, N)); -lambda3*Diagonal(ones(type, N)) lambda2*Diagonal(ones(type, N)) zeros(type, N, N)]
    Adjusts = [I1x I2x I3x T1 T2 T3 Adjusts]
    Adjusts = [zeros(type, 1, 6) transpose(I1xfix); zeros(type, 1, 6) transpose(I2xfix); zeros(type, 1, 6) transpose(I3xfix); zeros(type, 1, 6) transpose(T1); zeros(type, 1, 6) transpose(T2); zeros(type, 1, 6) transpose(T3); Adjusts]

    dFx = dFx + Adjusts

    return dFx

end
