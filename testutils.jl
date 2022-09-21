using Random
using Distributions
using LinearAlgebra
using SparseArrays


"""Construct a random Hermitian matrix of size N×N with spectral radius ≈ ρ."""
function random_hermitian_matrix(N, ρ=1.0)
    σ = 1 / √N
    d = Normal(0.0, σ)
    X = (rand(d, (N, N)) + rand(d, (N, N)) * 1im) / √2
    H = ρ * (X + X') / (2 * √2)
end


"""Construct a random sparse Hermitian matrix with spectral radius ≈ ρ."""
function random_hermitian_sparse_matrix(N; ρ=1.0, sparsity=0.5)
    σ = 1 / √(sparsity * N)
    d = Normal(0.0, σ)
    H1 = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    H2 = copy(H1)
    H2.nzval .= rand(d, length(H2.nzval))
    X = (H1 + H2 * 1im) / √2
    return 0.5ρ * (X + X') / √2
end


"""Return a random, normalized Hilbert space state vector of dimension `N`."""
function random_state_vector(N)
    Ψ = rand(N) .* exp.((2π * im) .* rand(N))
    Ψ ./= norm(Ψ)
    return Ψ
end

function getcontrols(generator::Tuple)
    controls = []
    slots_dict = IdDict()  # utilized as Set of controls we've seen
    for (i, part) in enumerate(generator)
        if isa(part, Tuple)
            control = part[2]
            if control in keys(slots_dict)
                push!(slots_dict[control], i)
            else
                push!(controls, control)
                slots_dict[control] = [i]
            end
        end
    end
    return Tuple(controls)
end


function init_system(make_operator; N_Hilbert=10, N_pulses=2)
    generator_list = Any[make_operator(N_Hilbert)]
    for n = 1:N_pulses
        val = rand()
        ϵ = t -> val
        Ĥ = make_operator(N_Hilbert)
        push!(generator_list, (Ĥ, ϵ))
    end
    generator = Tuple(generator_list)
    Ψ = random_state_vector(N_Hilbert)
    controls = getcontrols(generator)
    vals_dict = IdDict(ϵ => rand() for ϵ ∈ controls)
    return Ψ, generator, vals_dict
end
