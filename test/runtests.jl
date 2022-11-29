using MonogenicFilterFlux, ContinuousWavelets
using Flux, FFTW, CUDA, Wavelets, Zygote
using Logging, Test, LinearAlgebra

@testset "MonogenicFilterFlux.jl" begin
    include("MonogenicConvFFTConstructors.jl")
end