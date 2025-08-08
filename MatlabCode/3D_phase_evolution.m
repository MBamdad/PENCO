clc;
clear;
close all;

%% ======================= Configuration =======================
% --- File Path ---

% You can switch between these paths. The titles will update automatically.

% SH3D

%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/SH3D/plots_TNO3d/TNO3d_SH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat';
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/SH3D/plots_Data_Physics_TNO3d/TNO3d_SH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat';
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/SH3D/plots_FNO3d/FNO3d_SH3D_S32_T1to91_width12_modes14_q12_h6_grf3d.pt_results.mat'

mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/SH3D/plots_FNO4d/FNO4d_SH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat'


%AC3D

%mat_filepath = '//scratch/noqu8762/phase_field_equations_4d/AC3D/plots_TNO3d/TNO3d_AC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/AC3D/plots_FNO3d/FNO3d_AC3D_S32_T1to91_width12_modes14_q12_h6_grf3d.pt_results.mat'
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/AC3D/plots_Data_Physics_TNO3d/TNO3d_AC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'


% AC3d FNO4d
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/AC3D/plots_FNO4d/FNO4d_AC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat'

% CH3D
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/CH3D/plots_TNO3d/TNO3d_CH3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
%mat_filepath ='/scratch/noqu8762/phase_field_equations_4d/CH3D/plots_Data_Physics_TNO3d/TNO3d_CH3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/CH3D/plots_FNO3d/FNO3d_CH3D_S32_T1to91_width12_modes14_q12_h6_grf3d.pt_results.mat'

%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/CH3D/plots_FNO4d/FNO4d_CH3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat'

% MBE3D
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/MBE3D/plots_TNO3d/TNO3d_MBE3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/MBE3D/plots_Data_Physics_TNO3d/TNO3d_MBE3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat';
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/MBE3D/plots_FNO3d/FNO3d_MBE3D_S32_T1to91_width12_modes14_q12_h6_grf3d.pt_results.mat'

% MBE3D with FNO4d
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/MBE3D/plots_FNO4d/FNO4d_MBE3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat'

% PFC
%mat_filepath ='/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_TNO3d/TNO3d_PFC3D_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'; %valid
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_Data_Physics_TNO3d/TNO3d_PFC3D_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'; %valid
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_FNO3d/FNO3d_PFC3D_S32_T1to91_width12_modes14_q12_h6_grf3d.pt_results.mat'

%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_Data_Physics_TNO3d/TNO3d_PFC3D_Mixed_Hybrid_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
%mat_filepath ='/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_TNO3d/TNO3d_PFC3D_Mixed_S32_T1to100_width12_modes14_q12_h6_grf3d.pt_results.mat'
 
% PFC3D with FNO4d
%mat_filepath = '/scratch/noqu8762/phase_field_equations_4d/PFC3D/plots_FNO4d/FNO4d_PFC3D_S32_T1to91_width12_modes14_q12_h6_grf3d_NoMixed.pt_results.mat'


% --- Visualization Parameters ---
time_steps_to_plot = [0, 50, 90];
sample_index_to_plot = 2; % set 2 for MBE model
iso_value = 0;
error_display_threshold_ratio = 0.20;

%% ======================= Automatic Title Generation =======================
path_parts = strsplit(mat_filepath, '/');
problem_name = 'UnknownProblem'; 
base_dir_idx = find(strcmp(path_parts, 'phase_field_equations_4d'));
if ~isempty(base_dir_idx) && base_dir_idx < length(path_parts)
    problem_name = path_parts{base_dir_idx + 1};
end
model_name = 'UnknownModel';
if ~isempty(base_dir_idx) && (base_dir_idx + 1) < length(path_parts)
    plot_dir_name = path_parts{base_dir_idx + 2};
    switch plot_dir_name
        case 'plots_Data_Physics_TNO3d'
            model_name = 'Hybrid';
        case 'plots_TNO3d'
            model_name = 'MHNO';
        case 'plots_FNO3d'
            model_name = 'FNO-3d';
        case 'plots_FNO4d'
            model_name = 'FNO-4d';
            
    end
end
main_plot_title = sprintf('3D Phase Evolution of %s Based on %s Model', problem_name, model_name);
loss_plot_title = sprintf('Loss Progression for %s - %s Model', problem_name, model_name);
fprintf('Generated Plot Title: "%s"\n', main_plot_title);

%% ======================= Load and Prepare Data =======================
fprintf('Loading data from: %s\n', mat_filepath);
if ~exist(mat_filepath, 'file')
    error('File not found! Please check the mat_filepath variable.');
end
results = load(mat_filepath);
disp('Data loaded successfully.');

u_input_all = double(results.test_input);
u_exact_all = double(results.test_exact);
u_pred_all  = double(results.test_prediction);
u_error_all = u_pred_all - u_exact_all;

% First, squeeze to extract the single sample we want to plot
u_input = squeeze(u_input_all(sample_index_to_plot, :, :, :, :));
u_exact = squeeze(u_exact_all(sample_index_to_plot, :, :, :, :));
u_pred = squeeze(u_pred_all(sample_index_to_plot, :, :, :, :));
u_error = squeeze(u_error_all(sample_index_to_plot, :, :, :, :));

% Second, permute the dimensions of the extracted sample from (x,y,z,t) 
% to (y,x,z,t) to match the 'meshgrid' convention required by plotting functions.
u_input = permute(u_input, [2 1 3 4]);
u_exact = permute(u_exact, [2 1 3 4]);
u_pred  = permute(u_pred,  [2 1 3 4]);
u_error = permute(u_error, [2 1 3 4]);
Lx = double(results.config_Lx);
%Lx = 2*pi %double(results.config_Lx);
Ly = Lx;
Lz = Lx;

% Get the new size after permutation (Note: Ny is now the first dimension)
[Ny, Nx, Nz, ~] = size(u_exact);
x = linspace(-Lx/2, Lx/2, Nx);
y = linspace(-Ly/2, Ly/2, Ny);
z = linspace(-Lz/2, Lz/2, Nz);

% Use 'meshgrid' to create the grid, which is expected by functions like 'slice'.
[xx, yy, zz] = meshgrid(x, y, z);


%% ======================= Plot 1: Loss Curves (Generalized) =======================
fig1 = figure('Name', 'Loss vs. Epoch', 'NumberTitle', 'off', 'Position', [100, 400, 800, 600]);
if isfield(results, 'test_loss_hybrid_log')
    hybrid_loss = double(results.test_loss_hybrid_log);
    data_loss = double(results.test_data_log);
    epochs = 1:length(hybrid_loss);
    semilogy(epochs, hybrid_loss, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Test Hybrid Loss');
    hold on;
    semilogy(epochs, data_loss, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Test Data-Only Loss');
    hold off;
    legend('show', 'Location', 'northeast');
else
    test_loss = double(results.test_l2_log);
    epochs = 1:length(test_loss);
    semilogy(epochs, test_loss, 'g-', 'LineWidth', 2.5, 'DisplayName', 'Test L2 Loss');
    legend('show', 'Location', 'northeast');
end
grid on; box on;
xlabel('Epoch', 'FontWeight', 'bold');
ylabel('L2 Relative Loss', 'FontWeight', 'bold');
title(loss_plot_title, 'FontSize', 14);
set(gca, 'FontSize', 12, 'LineWidth', 1);

%% ======================= Plot 2: 3D Subplots (FINAL - CORRECTED SPACING) =======================
num_times = length(time_steps_to_plot);
fig2 = figure('Name', '3D Field Comparison', 'NumberTitle', 'off', 'Position', [200, 100, 950, 800]); 
set(fig2, 'Color', [0.94 0.94 0.94]);

tl = tiledlayout(3, num_times, 'TileSpacing', 'compact', 'Padding', 'normal');
title(tl, {main_plot_title; ''}, 'FontSize', 22, 'FontWeight', 'bold');
custom_error_map = [linspace(1,0,256)', linspace(1,1,256)', linspace(0.2,0,256)']; % Yellow->Green

for i = 1:num_times
    t_step = time_steps_to_plot(i);
    
    if t_step == 0
        exact_slice = u_input;
        pred_slice  = u_input;
        title_time_str = 't = 0 (Input)';
        error_title_str = 't = 0';
    else
        exact_slice = u_exact(:, :, :, t_step);
        pred_slice  = u_pred(:, :, :, t_step);
        error_slice = u_error(:, :, :, t_step);
        title_time_str = sprintf('%d\\Deltat', t_step);
        norm_of_error = norm(error_slice(:));
        norm_of_exact = norm(exact_slice(:));
        relative_l2_error = (norm_of_exact > 1e-9) * (norm_of_error / norm_of_exact);
        error_title_str = sprintf('Rel. L2: %.2f', relative_l2_error);
    end
    
    % --- Exact Row ---
    ax1 = nexttile(i);
    plot_isosurface_with_fallback(ax1, xx, yy, zz, exact_slice, iso_value, [0.85, 0.25, 0.25]);
    title(ax1, title_time_str, 'FontSize', 20, 'FontWeight', 'bold'); 
    if i == 1 % for PFC only we set  ax1, -0.9, 0.5, , and the rest should be ax1, -0.7, 0.5, 
        text(ax1, -0.9, 0.5, 'Exact', 'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Rotation', 90, 'Units', 'normalized');
    end

    % --- Predicted Row ---
    ax2 = nexttile(i + num_times);
    plot_isosurface_with_fallback(ax2, xx, yy, zz, pred_slice, iso_value, [0.25, 0.5, 0.85]);
    title(ax2, title_time_str, 'FontSize', 20, 'FontWeight', 'bold');
    if i == 1
        text(ax2, -0.9, 0.5, 'Predicted', 'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Rotation', 90, 'Units', 'normalized');
    end

    % --- Error Row ---
    ax3 = nexttile(i + 2*num_times);
    if t_step == 0
        [x_grid, y_grid, z_grid] = meshgrid(linspace(min(x), max(x), 5), linspace(min(y), max(y), 5), linspace(min(z), max(z), 5));
        scatter3(ax3, x_grid(:), y_grid(:), z_grid(:), 40, [1 1 0.2], 'filled');
        colorbar(ax3, 'off');
    else
        abs_error = abs(u_error(:,:,:,t_step));
        max_abs_error = max(abs_error(:));
        error_threshold = error_display_threshold_ratio * max_abs_error;
        idx_to_plot = find(abs_error >= error_threshold);
        if isempty(idx_to_plot) || max_abs_error < 1e-9
            text(ax3, 0.5, 0.5, 'No significant error', 'HorizontalAlignment', 'center', 'FontSize', 14, 'Units', 'normalized', 'FontWeight', 'bold');
            axis(ax3, 'off');
        else
            x_err = xx(idx_to_plot); y_err = yy(idx_to_plot); z_err = zz(idx_to_plot);
            c_data = abs_error(idx_to_plot);
            scatter3(ax3, x_err, y_err, z_err, 40, c_data, 'filled');
            caxis(ax3, [error_threshold, max_abs_error]);
        end
    end
    
    title(ax3, error_title_str, 'FontSize', 16, 'FontWeight', 'bold'); 
    colormap(ax3, custom_error_map); 
    cb = colorbar(ax3); 
    ylabel(cb, 'Relative Error', 'FontSize', 14, 'FontWeight', 'bold'); 
    cb.FontSize = 14;
    cb.FontWeight = 'bold';
    
    plot_isosurface_with_fallback(ax3, [],[],[],[],[],[]); % Just use it to set axis properties
    if i == 1
        text(ax3, -0.9, 0.5, 'Error', 'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Rotation', 90, 'Units', 'normalized');
    end
end

%% ======================= Save Plots to File =======================
fprintf('\nSaving generated plots...\n');
output_dir = 'saved_plots';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
loss_plot_filename = sprintf('%s_%s_loss_curve.png', problem_name, model_name);
field_plot_filename = sprintf('%s_%s_field_comparison_FINAL.png', problem_name, model_name);

try
    exportgraphics(fig1, fullfile(output_dir, loss_plot_filename), 'Resolution', 300);
    fprintf('Saved loss plot to: %s\n', fullfile(output_dir, loss_plot_filename));
catch ME
    fprintf('Error saving loss plot: %s\n', ME.message);
end

try
    exportgraphics(fig2, fullfile(output_dir, field_plot_filename), 'Resolution', 600);
    fprintf('Saved high-quality field plot to: %s\n', fullfile(output_dir, field_plot_filename));
catch ME
    fprintf('Error saving field plot with exportgraphics: %s\n', ME.message);
end

disp('Visualization script finished.');

%% ======================= MODIFIED Local Helper Function =======================
function plot_isosurface_with_fallback(ax, xx, yy, zz, data, iso_value, color)
    % This helper function plots a 3D surface. It first tries the specified
    % iso_value. If no surface is found, it automatically tries to plot the
    % "peaks" and "valleys" of the data to show its 3D structure.

    if ~isempty(data)
        % --- Primary Plotting Attempt ---
        % Try to find the surface at the requested iso_value (usually 0).
        [faces, vertices] = isosurface(xx, yy, zz, data, iso_value);

        if ~isempty(faces)
            % SUCCESS: The u=0 surface exists. Plot it.
            p = patch(ax, 'Faces', faces, 'Vertices', vertices);
            set(p, 'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
            lighting(ax, 'gouraud'); material(ax, 'dull');
            camlight(ax);
        else
            % --- Fallback Plotting Logic ---
            % The u=0 surface does not exist. Let's visualize the data's
            % actual structure by plotting its peaks and valleys.

            % Define new isosurface values based on the data's range.
            max_val = max(data(:));
            min_val = min(data(:));
            
            % Set thresholds to avoid plotting noise near zero.
            positive_iso = 0.3 * max_val;
            negative_iso = 0.3 * min_val;
            
            % Find the "peak" and "valley" surfaces.
            [faces_pos, vertices_pos] = isosurface(xx, yy, zz, data, positive_iso);
            [faces_neg, vertices_neg] = isosurface(xx, yy, zz, data, negative_iso);

            if isempty(faces_pos) && isempty(faces_neg)
                % Fallback also failed (data is too flat). Show the old slice plot.
                slice(ax, xx, yy, zz, data, [], [], 0);
                shading(ax, 'interp'); caxis(ax, [-1.2, 1.2]); colorbar(ax);
                text(ax, 0.05, 0.95, 'No clear structure found.', 'Color', 'k', 'BackgroundColor', 'w', 'Margin', 2, 'VerticalAlignment', 'top', 'FontSize', 9, 'Units', 'normalized');
            else
                % SUCCESS: We found some structure. Plot it.
                hold(ax, 'on');
                % Plot positive surface (peaks)
                if ~isempty(faces_pos)
                    p_pos = patch(ax, 'Faces', faces_pos, 'Vertices', vertices_pos);
                    set(p_pos, 'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
                end
                % Plot negative surface (valleys) using a slightly different shade for contrast
                if ~isempty(faces_neg)
                    p_neg = patch(ax, 'Faces', faces_neg, 'Vertices', vertices_neg);
                    % Use a complementary or darker/lighter color for the second surface
                    comp_color = color * 0.6; % A simple way to make it darker
                    set(p_neg, 'FaceColor', comp_color, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
                end
                hold(ax, 'off');
                lighting(ax, 'gouraud'); material(ax, 'dull');
                camlight(ax);
            end
        end
    end
    
    % --- Universal Axis Formatting ---
    daspect(ax, [1 1 1]); view(ax, 45, 30);
    grid(ax, 'on'); box(ax, 'on'); axis(ax, 'tight');
    xlabel(ax, 'X', 'FontSize', 16, 'FontWeight', 'bold'); 
    ylabel(ax, 'Y', 'FontSize', 16, 'FontWeight', 'bold'); 
    zlabel(ax, 'Z', 'FontSize', 16, 'FontWeight', 'bold');
    ax.FontSize = 14;
    ax.FontWeight = 'bold';
end