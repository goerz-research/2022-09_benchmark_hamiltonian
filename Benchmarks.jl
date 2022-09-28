# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Julia 1.8.0
#     language: julia
#     name: julia-1.8
# ---

# # Benchmarks

# Consider a time-dependent Hamiltonian of the form
#
# $$\hat{H}(t) = \hat{H}_0 + ϵ_1(t) \hat{H}_1 + ϵ_2(t) \hat{H}_2 + ...$$
#
# being evaluated for some value of $t$ and then applied to a state $|\Psi⟩$.
#
# Here we benchmark the "greedy" time-evaluation
#
# $$|Φ⟩ = \left(\hat{H}_0 + ϵ_1(t) \hat{H}_1 + ϵ_2(t) \hat{H}_2 + ...\right) |Ψ⟩$$
#
# where we sum the entire Hamiltonian into a single operator before applying it vs the "lazy" evaluation
#
# $$|Φ⟩ = \hat{H}_0  |Ψ⟩  + ϵ_1(t) \hat{H}_1  |Ψ⟩  + ϵ_2(t) \hat{H}_2 |Ψ⟩ + ..$$
#
# where we apply the terms of Hamiltonian to the state one-by-one.

using DrWatson

using BenchmarkTools
using LinearAlgebra
using Statistics
using Plots
using Term
using Test
using Printf

plotlyjs()

include("./testutils.jl")

for var in [
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS"
]
    ENV[var] = "1"
end

function make_apply_generator_greedy(state::ST, generator::GT) where {ST,GT}
    @assert !(generator[1] isa Tuple)  # constant drift Hamiltonian
    Ĥ = similar(generator[1])
    ϕ::ST = similar(state)
    function apply_generator_greedy(state::ST, generator::GT, vals_dict)
        if generator[begin] isa Tuple
            Ĥ₀, ϵ₀ = generator[begin]
            axpy!(vals_dict[ϵ₀], Ĥ₀, Ĥ)
        else
            Ĥ₀ = generator[begin]
            copyto!(Ĥ, Ĥ₀)
        end
        @inbounds for part in generator[begin+1:end]
            if part isa Tuple
                Ĥₙ, ϵₙ = part
                axpy!(vals_dict[ϵₙ], Ĥₙ, Ĥ)
            else
                Ĥₙ = part
                axpy!(true, Ĥₙ, Ĥ)
            end
        end
        mul!(ϕ, Ĥ, state)
        return ϕ
    end
end


function make_apply_generator_lazy(state::ST, generator::GT) where {ST,GT}
    ϕ::ST = similar(state)
    function apply_generator_lazy(state::ST, generator::GT, vals_dict)
        part = generator[begin]
        if part isa Tuple
            Ĥ₀, control = part
            val = vals_dict[control]
            mul!(ϕ, Ĥ₀, state, val, false)
        else
            Ĥ₀ = part
            mul!(ϕ, Ĥ₀, state)
        end
        @inbounds for part in generator[begin+1:end]
            if part isa Tuple
                Ĥₙ, control = part
                val = vals_dict[control]
                mul!(ϕ, Ĥₙ, state, val, true)
            else
                Ĥₙ = part
                mul!(ϕ, Ĥₙ, state, true, true)
            end
        end
        return ϕ
    end
end

function run_benchmarks(make_operator; N_Hilbert=10, N_pulses=2)
    Ψ, generator, vals_dict = init_system(make_operator; N_Hilbert, N_pulses)
    apply_generator_greedy = make_apply_generator_greedy(Ψ, generator)
    apply_generator_lazy = make_apply_generator_lazy(Ψ, generator)
    ϕ1 = apply_generator_greedy(Ψ, generator, vals_dict)
    ϕ2 = apply_generator_lazy(Ψ, generator, vals_dict)
    @test norm(ϕ1 - ϕ2) < 1e-14
    b_greedy = @benchmarkable $apply_generator_greedy($Ψ, $generator, $vals_dict)
    b_lazy = @benchmarkable $apply_generator_lazy($Ψ, $generator, $vals_dict)
    tune!(b_greedy; seconds=5, evals=10)
    tune!(b_lazy; seconds=5, evals=10)
    evals = max(b_greedy.params.evals, b_lazy.params.evals, 10)
    samples = max(b_greedy.params.samples, b_lazy.params.samples, 1000)
    # @show evals # TODO
    # @show samples
    return run(b_greedy; evals, samples), run(b_lazy; evals, samples)
end

"""Get the average runtime for the benchmark in ns, excluding outliers.

Outliers are defined as being outside the range `(M-Δ, M+2Δ)`, where `M` is the
median runtime and `Δ` is the difference between the median and the minimum runtime.

Return the average and standard deviation.
"""
function average_times(benchmark)
    time_min = minimum(benchmark.times)
    time_median = median(benchmark.times)
    Δ = time_median - time_min
    filtered_times = filter(t -> (time_min ≤ t ≤ time_median + 2Δ), benchmark.times)
    return mean(filtered_times), std(filtered_times)
end


function as_table(data)
    Term.Tables.Table(
        hcat(
            [@sprintf("%d", N) for N in data["N_Hilbert_vals"]],
            [@sprintf("%.2e", v / 1e6) for v in data["runtime_greedy"]],
            [@sprintf("%.2e", v / 1e6) for v in data["runtime_lazy"]],
        ),
        header=["N", "greedy (ms)", "lazy (ms)"],
        columns_justify=:right
    )
end

# ## Linear Algebra Benchmarks

# +
function benchmark_mv_vs_mpm()

    N = 1000
    H = random_hermitian_matrix(N)
    H0 = random_hermitian_matrix(N)
    Ψ = random_state_vector(N)
    ϕ = random_state_vector(N)
    val = 1.15

    println("*** matrix-vector product ϕ = H Ψ")
    b_mv1 = @benchmark mul!($ϕ, $H, $Ψ)
    display(b_mv1)

    println("*** matrix-vector product ϕ += v H Ψ")
    b_mv2 = @benchmark mul!($ϕ, $H, $Ψ, $val, true)
    display(b_mv2)

    println("*** matrix-vector product ϕ += H Ψ")
    b_mv3 = @benchmark mul!($ϕ, $H, $Ψ, true, true)
    display(b_mv3)

    println("*** matrix-matrix addition H += v H0")
    b_mpm1 = @benchmark axpy!($val, $H0, $H)
    display(b_mpm1)

    println("*** matrix-matrix addition H += H0")
    b_mpm2 = @benchmark axpy!(true, $H0, $H)
    display(b_mpm2)

    println("*** matrix-copy H = H0")
    b_mpm3 = @benchmark copyto!($H, $H0)
    display(b_mpm3)

end
# -

benchmark_mv_vs_mpm()

# ## Dense Matrices

# ### 2 Pulses

function benchmark_series(; force=false)

    filename = "benchmarks_dense_2pulses.jld2"
    data, file = @produce_or_load(Dict(); filename, force) do _
        N_pulses = 2
        runtime_greedy = Float64[]
        std_greedy = Float64[]
        runtime_lazy = Float64[]
        std_lazy = Float64[]
        N_Hilbert_vals =
            [5, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000]
        for N_Hilbert in N_Hilbert_vals
            b_greedy, b_lazy = run_benchmarks(
                N -> random_hermitian_matrix(N, 1.0);
                N_Hilbert,
                N_pulses
            )
            t̄1, σ1 = average_times(b_greedy)
            push!(runtime_greedy, t̄1)
            push!(std_greedy, σ1)
            t̄2, σ2 = average_times(b_lazy)
            push!(runtime_lazy, t̄2)
            push!(std_lazy, σ2)
        end
        @strdict(N_Hilbert_vals, runtime_greedy, std_greedy, runtime_lazy, std_lazy)
    end

    fig = plot(
        data["N_Hilbert_vals"],
        data["runtime_greedy"] .* 1e-6,
        yerr=data["std_greedy"] .* 1e-6,
        label="greedy",
        marker=true
    )
    plot!(
        fig,
        data["N_Hilbert_vals"],
        data["runtime_lazy"] .* 1e-6,
        yerr=data["std_lazy"] .* 1e-6,
        label="lazy",
        marker=true
    )
    if !isnothing(match(r"^In\[[0-9]*\]$", @__FILE__))  # notebook
        display(plot!(fig; legend=:top, xlabel="Hilbert space dimension", ylabel="runtime (ms)"))
    end
    display(as_table(data))
    return data
end

benchmark_series(; force=false);
