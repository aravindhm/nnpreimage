
conn = database('', '', '', 'org.sqlite.JDBC', 'jdbc:sqlite:/var/www/html/naturalness/question_bank.sq3');

curs = exec(conn,'select * from questions');
curs = fetch(curs);

close (conn);

votes = zeros(20, 3);

num_views = zeros(20, 3); % The number of times a choice was presented to the user

for i=1:size(curs.Data,1)
   qno = curs.Data{i,2};
   q_ans = curs.Data{i,end};

   if(qno > 5 && ~isnan(q_ans))

     layer = curs.Data{i,3};
     choices = [curs.Data{i,4}, curs.Data{i,6}];

     votes(layer, choices(q_ans)) = votes(layer, choices(q_ans)) + 1;

     num_views(layer, choices(1)) = num_views(layer, choices(1)) + 1;
     num_views(layer, choices(2)) = num_views(layer, choices(2)) + 1;
   end
end

layers_of_interest = [4,10,15,20];
rate = votes(layers_of_interest,:) ./ num_views(layers_of_interest,:);
error_bars = 1.96*(rate.*(1-rate) ./ num_views(layers_of_interest,:)).^0.5;

all_colors = [0, 0, 0.9; 0, 0.9, 0; 0.9, 0, 0];

figure;
data = votes(layers_of_interest, :) ./ num_views(layers_of_interest,:); 
hb = bar(data);
hold on;
for ib = 1:numel(hb)
  hb(ib).FaceColor = all_colors(ib,:);
  hb(ib).EdgeColor = all_colors(ib,:);
  xData = hb(ib).XData+hb(ib).XOffset;
  errorbar(xData, data(:, ib), error_bars(:, ib), '.', 'LineWidth', 2, 'Color', all_colors(ib,:)*3/4);
end
legend('C=1', 'C=100', 'No reg.');
axis([0.5, 4.5, 0, 1]);

ax = gca;
ax.FontSize = 15;
ax.XTickLabel = {'MPool1', 'ReLU3', 'MPool5', 'FC8'};
xlabel('Layer Name');
ylabel('Preference Ratio');
