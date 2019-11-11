function grad2!(G, para)
    for i in 1:length(para)
        G[i] = central_fdm(2,1)(loss, para[i])
    end
end

loss(x, y) = x^2 + y^2
central_fdm(2,1)(loss, x)
