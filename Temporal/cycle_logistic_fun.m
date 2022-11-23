function L = cycle_logistic_fun(b, x1, x2)

m = (x1 - b(2))./(b(1)-x1);

%m(m<=0) = 1;

L = abs(b(3)+abs(b(4)).*log(m)-x2);
