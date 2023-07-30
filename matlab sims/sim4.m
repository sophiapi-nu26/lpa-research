% find value of v for which around 50% of trials converge to one value
% starting with N = 3125 and doubling numDoubles (-1) times

numDoubles = 6;
type = 1;

N_arr = zeros(1, numDoubles);
N_arr(1) = 3125;
v_arr = zeros(1, numDoubles);

for i = 1:numDoubles
    fprintf("current N = %d\n", N_arr(i));
    v_arr(i) = BinSearchV(type, N_arr(i), 0.01);
    fprintf("for N = %f, v is around %f\n", N_arr(i), v_arr(i));
    N_arr(i+1) = N_arr(i) * 2;
end

% plot results on log log scale
% plot N on x-axis, avg degree on y-axis (= Np = N*(N^v))
loglog(N_arr, N_arr.*(N_arr.^v_arr), '-*', 'MarkerSize', 10, 'LineWidth', 2)
xlabel('N')
ylabel('Average degree Np')
title('Calculated value of Np for N up to 6.4e5 (type = %d)', type)
% get and plot line of best fit
const = polyfit(log(N_arr), log(N_arr.*(N_arr.^v_arr)), 1);
m = const(1);
b = const(2);
hold on;
plot(N_arr, (N_arr.^m).*exp(b), 'LineWidth', 2)
