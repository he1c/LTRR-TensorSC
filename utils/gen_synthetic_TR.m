function X=gen_synthetic_TR(ss,TRrank)

N=length(TRrank);

TRrank=[TRrank;TRrank(1)];

U=cell(N,1);

for i=1:1:N
    U{i}=randn(TRrank(i),ss(i),TRrank(i+1));%-0.5;
end

X=fullTR(U);

end