mos = xlsread('C:\Zohaib\Metrics\brisque/brisque_mos_order.xlsx');
metric = xlsread('C:\Zohaib\Metrics\brisque/brisque_score.xlsx'); %metric = metric';

scatter(metric,mos,100,'+','MarkerEdgeColor','b','MarkerFaceColor','b','LineWidth',1.5)

 hold on
 fact = 20;
 beta0 = [max(mos) min(mos) fact*mean(metric) 1 0];
 
 logistic_eq =  'b1*(0.5 - 1./(1+exp(b2*(x-b3)))) + b4.*x + b5';
% logistic_eq1 =  'b1.*x^4 + b2.*x^3 + b4.*x^2 + b3.*x + b5';
 f_line = fit(metric,mos,logistic_eq,'Start', beta0)

 plot(f_line)

legend('off')

xlabel('Predicted Score','FontSize',20)
ylabel('MOS','FontSize',20)
box on

set(gca,'FontSize',18)

%saveas(gcf,'scatter.eps','epsc')


