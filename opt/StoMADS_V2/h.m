
function z = h(x, nprob)
if (isrow(x))
    x = x';
end
u = Det_bb_call(x, nprob);
v = u(2:end);
z = sum(max(0, v));
end