function applyBC(x, bc::FourierFilterFlux.Periodic, nd)
    return (x, axes(x)[1:nd])
end

function applyBC(x, bc::Sym, nd)
    flipThisDim = cat(x, reverse(x, dims=nd), dims=nd)
    if nd == 1
        return flipThisDim, axes(x)[1:nd]
    else
        return applyBC(flipThisDim, bc, nd - 1)[1], axes(x)[1:nd]
    end
end
