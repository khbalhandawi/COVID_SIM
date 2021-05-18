function z = Det_bb_call(x, p_it)

p_it_vec = repmat(1:42,1,1000);

z = Det_black_box(x, p_it_vec(p_it)); % z is a row vector

end