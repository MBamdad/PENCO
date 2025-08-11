clc;
clear;
close all;

%% ======================= Configuration =======================
% Define the time steps to plot and their corresponding indices
time_steps = [0, 50, 90];  % The actual time values
time_indices = [1, 51, 91]; % Corresponding array indices (MATLAB is 1-indexed)

% Define all case studies and their paths
case_studies = {
    'AC3D', {
        '/scratch/noqu8762/phase_field_equations_4d/AC3D/plots_FNO4d/FNO4d_AC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/AC3D/plots_TNO3d/TNO3d_AC3D_S32_T1to100_width12_modes14_q12_h6_grf3d_Mixed.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/AC3D/plots_Data_Physics_TNO3d/TNO3d_AC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d_Mixed.pt_results.mat'
    };
    'CH3D', {
        '/scratch/noqu8762/phase_field_equations_4d/CH3D/plots_FNO4d/FNO4d_CH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/CH3D/plots_TNO3d/TNO3d_CH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/CH3D/plots_Data_Physics_TNO3d/TNO3d_CH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
    };
    'SH3D', {
        '/scratch/noqu8762/phase_field_equations_4d/SH3D/plots_FNO4d/FNO4d_SH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/SH3D/plots_TNO3d/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/SH3D/plots_Data_Physics_TNO3d/TNO3d_SH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
    };
    'MBE3D', {
        '/scratch/noqu8762/phase_field_equations_4d/MBE3D/plots_FNO4d/FNO4d_MBE3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/MBE3D/plots_TNO3d/TNO3d_MBE3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/MBE3D/plots_Data_Physics_TNO3d/TNO3d_MBE3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
    };
    'PFC3D', {
        '/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_FNO4d/FNO4d_PFC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_TNO3d/TNO3d_PFC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat', ...
        '/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_Data_Physics_TNO3d/TNO3d_PFC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
    }
};

method_names = {'FNO-4d', 'MHNO', 'PI-MHNO'};
colors = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.4 0.8 0.4]; % Different colors for each method

% Font size configuration
title_fontsize = 20;
axis_label_fontsize = 20;
tick_label_fontsize = 20;
legend_fontsize = 20;
sgtitle_fontsize = 24;

%% ======================= Load and Process Data =======================
% Initialize a structure to store all error data
all_error_data = struct();

for case_idx = 1:length(case_studies)
    case_name = case_studies{case_idx, 1};
    case_paths = case_studies{case_idx, 2};

    fprintf('\nProcessing case study: %s\n', case_name);

    % Initialize error data for this case study
    case_error_data = cell(length(method_names), length(time_steps));

    for method_idx = 1:length(method_names)
        mat_filepath = case_paths{method_idx};
        fprintf('Loading data from: %s\n', mat_filepath);

        if ~exist(mat_filepath, 'file')
            warning('File not found: %s', mat_filepath);
            case_error_data(method_idx, :) = {NaN(1, 1)};
            continue;
        end

        results = load(mat_filepath);

        % Get the data arrays
        u_input_all = double(results.test_input);
        u_exact_all = double(results.test_exact);
        u_pred_all = double(results.test_prediction);

        num_samples = size(u_exact_all, 1);

        for t_idx = 1:length(time_steps)
            time_step = time_steps(t_idx);
            time_index = time_indices(t_idx);

            if time_step == 0
                % For time step 0, we compare with input (initial condition)
                u_exact = u_input_all;
                u_pred = u_input_all; % Prediction at t=0 should match input
            else
                % Check if the requested time index exists in the data
                if size(u_exact_all, 5) < time_index || size(u_pred_all, 5) < time_index
                    warning('Time index %d (t=%d) not available in data. Using NaN.', time_index, time_step);
                    case_error_data{method_idx, t_idx} = NaN(num_samples, 1);
                    continue;
                end
                u_exact = u_exact_all(:, :, :, :, time_index);
                u_pred = u_pred_all(:, :, :, :, time_index);
            end

            % Calculate relative L2 error for each sample
            errors = zeros(num_samples, 1);
            for sample_idx = 1:num_samples
                % Process the data exactly as in the visualization code
                exact_slice = squeeze(u_exact(sample_idx, :, :, :, :));
                pred_slice = squeeze(u_pred(sample_idx, :, :, :, :));

                % Permute dimensions to match visualization code
                exact_slice = permute(exact_slice, [2 1 3 4]);
                pred_slice = permute(pred_slice, [2 1 3 4]);

                if time_step == 0
                    % At t=0, error should be zero (input matches input)
                    errors(sample_idx) = 0;
                else
                    % Calculate error exactly as in visualization code
                    error_slice = pred_slice - exact_slice;
                    norm_of_error = norm(error_slice(:));
                    norm_of_exact = norm(exact_slice(:));

                    if norm_of_exact > 1e-9
                        errors(sample_idx) = norm_of_error / norm_of_exact;
                    else
                        errors(sample_idx) = 0;
                    end
                end
            end

            case_error_data{method_idx, t_idx} = errors;
        end
    end

    % Store the error data for this case study
    all_error_data.(case_name) = case_error_data;
end

%% ======================= Create Master Figure =======================
% Create a larger figure with higher DPI for better scaling
fig = figure('Position', [100, 100, 1400, 1000], 'Color', 'w', 'Units', 'inches');

% Define subplot positions
subplot_positions = {
    [0.07, 0.55, 0.28, 0.35],  % AC3D
    [0.38, 0.55, 0.28, 0.35],   % CH3D
    [0.69, 0.55, 0.28, 0.35],   % SH3D
    [0.20, 0.10, 0.28, 0.35],   % MBE3D
    [0.55, 0.10, 0.28, 0.35]    % PFC3D
};

% Plot each case study
for case_idx = 1:length(case_studies)
    case_name = case_studies{case_idx, 1};
    case_error_data = all_error_data.(case_name);

    % Create subplot
    subplot('Position', subplot_positions{case_idx});
    hold on;

    % Adjust these parameters to control the appearance
    box_width = 0.2;
    box_positions = [];
    group_positions = [];

    % Calculate positions for each group of boxes
    for t_idx = 1:length(time_steps)
        for method_idx = 1:length(method_names)
            position = t_idx + (method_idx - 2) * box_width * 1.2;
            box_positions = [box_positions, position];
            group_positions = [group_positions, t_idx];
        end
    end

    % Plot each box
    for i = 1:numel(case_error_data)
        [method_idx, t_idx] = ind2sub(size(case_error_data), i);

        % Skip if data is NaN (missing time step)
        if all(isnan(case_error_data{method_idx, t_idx}))
            continue;
        end

        % Create boxplot
        boxplot_data = case_error_data{method_idx, t_idx};

        % Remove outliers to prevent them from dominating the y-axis scale
        q = quantile(boxplot_data, [0.25 0.75]);
        iqr = q(2) - q(1);
        upper_whisker = q(2) + 1.5*iqr;
        lower_whisker = q(1) - 1.5*iqr;
        boxplot_data(boxplot_data > upper_whisker | boxplot_data < lower_whisker) = [];

        % Plot the box
        bp = boxchart(box_positions(i)*ones(size(boxplot_data)), boxplot_data, ...
            'BoxWidth', box_width, ...
            'BoxFaceColor', colors(method_idx, :), ...
            'MarkerColor', colors(method_idx, :), ...
            'LineWidth', 1.5);

        % Add median line
        median_val = median(boxplot_data);
        line([box_positions(i)-box_width/2, box_positions(i)+box_width/2], [median_val, median_val], ...
            'Color', 'k', 'LineWidth', 2);
    end

    % Customize the subplot with larger fonts
    set(gca, 'FontSize', tick_label_fontsize, 'FontWeight', 'bold', 'LineWidth', 1.5);
    set(gca, 'XTick', 1:length(time_steps), 'XTickLabel', arrayfun(@(x) sprintf('t=%d', x), time_steps, 'UniformOutput', false));

    % Add title and labels with larger fonts
    title(case_name, 'FontSize', title_fontsize, 'FontWeight', 'bold');
    if ismember(case_idx, [1, 4]) % Only label left-side plots
        ylabel('Rel. L2 Error', 'FontSize', axis_label_fontsize, 'FontWeight', 'bold');
    end

    grid on;
    box on;

    % Adjust y-axis to show all data
    all_errors = vertcat(case_error_data{:});
    valid_errors = all_errors(~isnan(all_errors));
    if ~isempty(valid_errors)
        ylim([0, max(valid_errors) * 1.1]);
    end

    % Add legend to the first subplot only with larger font
    if case_idx == 1
        legend_entries = cell(length(method_names), 1);
        for i = 1:length(method_names)
            legend_entries{i} = patch([NaN NaN], [NaN NaN], colors(i, :), 'DisplayName', method_names{i});
        end
        legend([legend_entries{:}], 'Location', 'northwest', 'FontSize', legend_fontsize, 'FontWeight', 'bold', 'Box', 'off');
    end
end

% Add overall title with very large font
sgtitle('Comparison of Relative L2 Errors Across Case Studies and Methods', ...
    'FontSize', sgtitle_fontsize, 'FontWeight', 'bold');

%% ======================= Save Plot =======================
output_dir = 'saved_plots';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save at very high resolution for Word compatibility
output_filename = fullfile(output_dir, 'All_Case_Studies_L2Error_Comparison.png');
exportgraphics(fig, output_filename, 'Resolution', 600);
fprintf('Saved master plot to: %s\n', output_filename);

% Also save as PDF for better quality in publications
output_filename_pdf = fullfile(output_dir, 'All_Case_Studies_L2Error_Comparison.pdf');
exportgraphics(fig, output_filename_pdf, 'ContentType', 'vector');
fprintf('Saved master plot (PDF) to: %s\n', output_filename_pdf);

disp('Master plot generation completed.');