function z = cosin_vec(d,ds)

if iscolumn(d)
    d=d';
end
if iscolumn(ds)
    ds=ds';
end

z = sum(d .* ds) / (norm(d) * norm(ds));
end