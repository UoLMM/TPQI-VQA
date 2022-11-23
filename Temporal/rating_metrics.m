function metrics = rating_metrics(i, y_true, y_pred)
    % 准备评价函数
    figure(i)
    y_true = y_true(:);
    y_pred = y_pred(:);

    p_plcc = round(corr(y_true, y_pred, 'type', 'Pearson'), 3);
    p_srocc = round(corr(y_true, y_pred, 'type', 'Spearman'), 3);
    p_mae = round(mean(abs(y_true - y_pred)), 3);
    p_rmse = round(sqrt(mean((y_true - y_pred).^2)), 3);

    fprintf('SRCC: %.3f | PLCC: %.3f | MAE: %.3f | RMSE: %.3f', p_srocc, p_plcc, p_mae, p_rmse);

    txt = ['SRCC: ', num2str(p_srocc), ' PLCC: ', num2str(p_plcc), ' MAE: ', num2str(p_mae) ,' RMSE: ',num2str(p_rmse)];

    scatter(y_true, y_pred, '.');
    xlabel('ground-truth');
    ylabel('predicted');
    title(txt);
    hold on


    p = polyfit(y_true, y_pred, 1);
    y1 = polyval(p, y_true);
    plot(y_true, y1);

    metrics = [p_plcc p_srocc p_mae p_rmse];
end