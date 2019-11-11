%% para_est
cdata = [1 2 2.5 3; 1 2 10 100; 1 2.5 20 200; 1 3 55 158];
xvalues = {'gradient_descent','Nelder_Mead','Simulated_Annealing','Particle_Swarm'};
yvalues = {'1','10','100','1000'};
h = heatmap(xvalues,yvalues,cdata);

h.Title = 'Total run time comparison for parameter estimation';
h.XLabel = 'Algorithms';
h.YLabel = 'Number of parameters';
h.Colormap = jet


%% derivatives 

cdata = [1 2 2.5 3; 1 2 5 10; 1 2.5 4, 11; 1 3 5 15];
xvalues = {'forward_ad','cont_sens','num_diff','backward_ad'};
yvalues = {'1','10','100','1000'};
h = heatmap(xvalues,yvalues,cdata);

h.Title = 'Total run time comparison for calculating derivatives';
h.XLabel = 'Methods';
h.YLabel = 'Number of parameters';
h.Colormap = jet
