function h = h(x,y,sigma)
    h=(1/(2*pi*power(sigma,2)))*(exp(-(power(x,2)+power(y,2))/(2*power(sigma,2))));
end