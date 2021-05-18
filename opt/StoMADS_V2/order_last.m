
function Z = order_last(PP, ds)

T1 = PP';
U = zeros(length(T1(:,1)),1);
for i = 1:length(T1(:,1))
    U(i) = cosin_vec(T1(i,:),ds);
end

T = [T1 U];

for i=2:length(T(:,1))
    j=i;
    v=T(i,end); w=T(i,:);
    while (j>1)&&(v>T(j-1,end))        
        T(j,:)=T(j-1,:);
        j=j-1;
    end
    T(j,:)=w;
end

Z1 = T';

Z = Z1(1:end - 1,:);

end