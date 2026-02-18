import FourierFilterFlux: getBatchSize

function getBatchSize(c::C) where {C<:MonoConvFFT}
    if typeof(c.fftPlan) <: Tuple
        return c.fftPlan[2][end]
    else
        return c.fftPlan.sz[end]
    end
end

import FourierFilterFlux: cu

function cu(mono::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A,PD,P,T,S,AL}

    # D = mono.D
    # OT = mono.OT
    σ = mono.σ
    boundary = mono.bc

    # trainable = mono.T
    scale = mono.scale

    #averagingLayer = mono.averagingLayer
    averagingLayer = AL

    cuw = cu(mono.weight)
    cuf = cu(mono.fftPlan)


    return MonoConvFFT{D,OT,typeof(σ),typeof(cuw),typeof(boundary),typeof(cuf), T, typeof(scale), typeof(averagingLayer)}(mono.σ, cuw, boundary, cuf, scale, averagingLayer)
end

import Base: size
import FourierFilterFlux: originalSize

function size(l::C) where {C<:MonoConvFFT}
    if typeof(l.fftPlan) <: Tuple
        sz = l.fftPlan[2]
    else
        sz = l.fftPlan.sz
    end
    signalSize = originalSize(sz[1:ndims(l.weight[1])], l.bc)
    return (signalSize..., sz[(ndims(l.weight[1])+1):end]...)
end

function size(l::MonoConvFFT{D,OT,A,B,C,PD,P}) where {D,OT,A,B,C,PD,P<:Tuple}
    if typeof(l.fftPlan[1]) <: Tuple
        sz = l.fftPlan[1][2]
    else
        sz = l.fftPlan[1].sz
    end
    signalSize = originalSize(sz[1:ndims(l.weight[1])], l.bc)
    return (signalSize..., sz[(ndims(l.weight[1])+1):end]...)
end