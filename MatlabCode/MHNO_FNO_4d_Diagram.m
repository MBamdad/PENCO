% MATLAB script to generate architecture diagrams for FNO and SANO models
% (Version 34 - Corrected FNO diagram to reflect time-as-channels architecture)

clear; clc; close all;

%% --- Setup and Color Palette ---
fig = figure('Name', 'FNO and SANO Architectures', 'Position', [100, 100, 1400, 1000], 'Color', 'w');
set(fig, 'Renderer', 'painters'); % For sharp vector graphics and text

% Define a color palette
colors.box_fno = [1, 0.9, 0.9];
colors.box_sano = [0.9, 1, 0.9];
colors.input = [1, 0.9, 0.8];
colors.projection = [0.9, 0.95, 1];
colors.loss = [0.9, 0.9, 0.9];

% General plotting parameters
lineWidth = 1.5;
fontSize = 18;

%% --- Subplot a) FNO Architecture (Corrected for FNO4d) ---
ax1 = subplot(2, 1, 1);
hold on;
axis([0 40 0 10]);
axis off;
title('a) FNO Architecture (Time-as-Channels)', 'FontSize', 22, 'FontWeight', 'bold');

% --- FNO Core Block Box ---
rectangle('Position', [12, 1, 13, 8], 'FaceColor', colors.box_fno, 'EdgeColor', [1 0.5 0.5], 'LineWidth', 2, 'LineStyle', '--');
text(18.5, 0, '3D FNO Block', 'FontSize', 16, 'HorizontalAlignment', 'center');


% --- 1. Input and Grid ---
input_width = 6.5; input_height = 2.5;
input_center_x = 4.0;
drawBox([input_center_x, 7.5], input_width, input_height, 'Input a(x,y,z, t_{in})', colors.input, 'ellipse');
drawBox([input_center_x, 4.5], input_width, input_height, '3D Grid (x,y,z)', colors.input, 'ellipse');

% --- 2. Concatenation and Lifting (P) ---
concat_summer_x = input_center_x + input_width/2 + 1;
drawSummer([concat_summer_x, 6], 0.7);
text(concat_summer_x, 5, 'cat', 'FontSize', 14, 'FontWeight', 'bold');
plot([input_center_x + input_width/2, concat_summer_x - 0.7], [7.5, 6], 'k', 'LineWidth', lineWidth);
plot([input_center_x + input_width/2, concat_summer_x - 0.7], [4.5, 6], 'k', 'LineWidth', lineWidth);

p_center_x = concat_summer_x + 2; p_size = 1.2;
drawBox([p_center_x, 6], p_size, p_size, 'P', colors.input, 'ellipse');
drawArrow([concat_summer_x + 0.7, 6], [p_center_x - p_size/2, 6]);
drawArrow([p_center_x + p_size/2, 6], [12, 6]);

% --- 3. Split into W and K paths ---
plot([12, 12.4], [6.25, 8], 'k', 'LineWidth', lineWidth);
plot([12, 12.4], [6.25, 4], 'k', 'LineWidth', lineWidth);
text(12.5, 8.2, '(W)', 'FontSize', fontSize, 'FontWeight', 'bold');
text(12.5, 3.8, '(K)', 'FontSize', fontSize, 'FontWeight', 'bold');

% --- 4. Path (W): Local Transformation (Conv3d) ---
plot([12.4, 15.45], [8, 8], 'k', 'LineWidth', lineWidth);
drawResistor([16.2, 8], 1.5, 0.5, 'R1');
plot([16.95, 18.7], [8, 8], 'k', 'LineWidth', lineWidth);

% --- 5. Path (K): Non-Local Transformation (SpectralConv3d) ---
plot([12.4, 13.45], [4, 4], 'k', 'LineWidth', lineWidth);
drawResistor([14.2, 4], 1.5, 0.5, 'R2');
drawResistor([16.2, 4], 1.5, 0.5, 'R3');
drawResistor([18.2, 4], 1.5, 0.5, 'R4');
plot([18.95, 18.7], [4, 4], 'k', 'LineWidth', lineWidth);

% --- 6. Summing Junction and Activation (R5) ---
plot([18.7, 19], [8, 7], 'k', 'LineWidth', lineWidth);
plot([18.7, 19], [4, 5], 'k', 'LineWidth', lineWidth);
drawSummer([19.2, 6], 1);
plot([19.7, 20.2], [6, 6], 'k', 'LineWidth', lineWidth);
drawResistor([20.7, 6], 1, 0.5, 'R5');

% --- 7. Latent Rep and Projection Q ---
line_start_x_fno = 21.2;
line_end_x_fno = 25.0; % This is the right edge of the main box
plot([line_start_x_fno, line_end_x_fno], [6, 6], 'k', 'LineWidth', lineWidth);
text((line_start_x_fno + line_end_x_fno)/2, 6.7, 'v(x,y,z)', 'FontSize', fontSize, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', 'b');
drawBox([26.2, 6], 2, 1.5, 'Q_p', colors.projection, 'ellipse'); % Projection layer

% --- 8. FNO Output and Loss ---
loss_box_width = 7.0; loss_box_height = 2.8;
u_out_start_x_fno = 27.2;      % Right edge of Q_p
loss_arrow_start_x = 30.0;     % Point where the line ends and the Loss arrow begins.
loss_arrow_end_x = 32.0;     % End of Loss arrow.
loss_box_center_x = loss_arrow_end_x + loss_box_width / 2;

plot([u_out_start_x_fno, loss_arrow_start_x], [6, 6], 'k', 'LineWidth', lineWidth);
text((u_out_start_x_fno + loss_arrow_start_x) / 2, 6.7, 'u(x,y,z)', 'FontSize', fontSize, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
drawArrow([loss_arrow_start_x, 6], [loss_arrow_end_x, 6], 'Loss');
drawBox([loss_box_center_x, 6], loss_box_width, loss_box_height, '||G_{\theta}(a_j) - u_j||^2', colors.loss, 'rectangle', 'Interpreter', 'tex');


%% --- Subplot b) SANO Architecture ---
ax2 = subplot(2, 1, 2);
hold on;
axis([0 40 0 10]);
axis off;
title('b) SANO Architecture', 'FontSize', 22, 'FontWeight', 'bold');

% --- SPATIAL ENCODER for SANO ---
sano_encoder_offset = 0; % Keep it aligned with FNO above
% Main Box
rectangle('Position', [8.8+sano_encoder_offset, 1, 13, 8], 'FaceColor', colors.box_fno, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(15, -0.5, 'Spatial Encoder (run once)', 'FontSize', 14, 'HorizontalAlignment', 'center');
% Input and Lifting (P)
drawBox([3.0+sano_encoder_offset, 6], input_width, input_height, 'Input a(x)', colors.input, 'ellipse');
drawBox([6.8+sano_encoder_offset, 6], p_size, p_size, 'P', colors.input, 'ellipse');
drawArrow([3.0+sano_encoder_offset + input_width/2, 6], [6.8+sano_encoder_offset - p_size/2, 6]);
drawArrow([6.8+sano_encoder_offset + p_size/2, 6], [8.8+sano_encoder_offset, 6]);
% W and K paths
plot([8.8, 9.2]+sano_encoder_offset, [6.25, 8], 'k', 'LineWidth', lineWidth);
plot([8.8, 9.2]+sano_encoder_offset, [6.25, 4], 'k', 'LineWidth', lineWidth);
plot([9.2, 12.25]+sano_encoder_offset, [8, 8], 'k', 'LineWidth', lineWidth);
drawResistor([13, 8]+[sano_encoder_offset, 0], 1.5, 0.5, 'R1');
plot([13.75, 15.5]+sano_encoder_offset, [8, 8], 'k', 'LineWidth', lineWidth);
plot([9.2, 10.25]+sano_encoder_offset, [4, 4], 'k', 'LineWidth', lineWidth);
drawResistor([11, 4]+[sano_encoder_offset, 0], 1.5, 0.5, 'R2');
drawResistor([13, 4]+[sano_encoder_offset, 0], 1.5, 0.5, 'R3');
drawResistor([15, 4]+[sano_encoder_offset, 0], 1.5, 0.5, 'R4');
plot([15.75, 15.5]+sano_encoder_offset, [4, 4], 'k', 'LineWidth', lineWidth);
% Summing and Activation
plot([15.5, 15.8]+sano_encoder_offset, [8, 7], 'k', 'LineWidth', lineWidth);
plot([15.5, 15.8]+sano_encoder_offset, [4, 5], 'k', 'LineWidth', lineWidth);
drawSummer([16, 6]+[sano_encoder_offset, 0], 1);
plot([16.5, 17]+sano_encoder_offset, [6, 6], 'k', 'LineWidth', lineWidth);
drawResistor([17.5, 6]+[sano_encoder_offset, 0], 1, 0.5, 'R5');

% --- SANO Recurrent Block ---
recurrent_block_offset = 24; % Shift recurrent block to the right
% Box
rectangle('Position', [recurrent_block_offset, 1, 10.2, 8], 'FaceColor', colors.box_sano, 'EdgeColor', [0.5 1 0.5], 'LineWidth', 2, 'LineStyle', '--');
text(recurrent_block_offset + 5.1, -0.5, 'SANO Recurrent Update (for t > t_1)', 'FontSize', 16, 'HorizontalAlignment', 'center');

% Arrow from Spatial Encoder output to Recurrent block input
encoder_output_x = 18 + sano_encoder_offset;
recurrent_input_x = recurrent_block_offset;
plot([encoder_output_x, recurrent_input_x], [6, 8.2], 'k', 'LineWidth', lineWidth);
text(encoder_output_x + 1, 6.2, 'v_a(x)', 'FontSize', fontSize, 'FontWeight', 'bold', 'Color', 'b');

% Recurrent block inputs
text(recurrent_input_x - 1, 3.8, 'u(x,t-1)', 'FontSize', fontSize, 'FontWeight', 'bold', 'Color', 'r', 'HorizontalAlignment', 'right');

% Branch (Q)
plot([recurrent_input_x, recurrent_input_x + 0.7], [8.2, 8.2], 'k', 'LineWidth', lineWidth);
drawResistor([recurrent_input_x + 2, 8.2], 1.5, 0.5, 'R1');
drawResistor([recurrent_input_x + 4.5, 8.2], 1.5, 0.5, 'R5');
drawResistor([recurrent_input_x + 7, 8.2], 1.5, 0.5, 'R1');
plot([recurrent_input_x + 7.75, recurrent_input_x + 8], [8.2, 7], 'k', 'LineWidth', lineWidth);
text(recurrent_input_x, 9, '(Q)', 'FontSize', fontSize, 'FontWeight', 'bold');

% Branch (H)
plot([recurrent_input_x, recurrent_input_x + 0.7], [3.8, 3.8], 'k', 'LineWidth', lineWidth);
drawResistor([recurrent_input_x + 2, 3.8], 1.5, 0.5, 'R1');
drawResistor([recurrent_input_x + 4.5, 3.8], 1.5, 0.5, 'R5');
drawResistor([recurrent_input_x + 7, 3.8], 1.5, 0.5, 'R1');
plot([recurrent_input_x + 7.75, recurrent_input_x + 8], [3.8, 5], 'k', 'LineWidth', lineWidth);
text(recurrent_input_x, 3, '(H)', 'FontSize', fontSize, 'FontWeight', 'bold');

% Summing Junction and Output
summer_center_x = recurrent_input_x + 8.5;
drawSummer([summer_center_x, 6], 1);
u_out_start_x_sano = summer_center_x + 0.5;
plot([u_out_start_x_sano, loss_arrow_start_x], [6, 6], 'k', 'LineWidth', lineWidth);
text((u_out_start_x_sano + loss_arrow_start_x) / 2, 6.7, 'u(x,t)', 'FontSize', fontSize, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Recurrent Loop (Feedback)
plot([loss_arrow_start_x, loss_arrow_start_x, recurrent_input_x - 0.5], [6, 2, 2], 'r', 'LineStyle', '--', 'LineWidth', lineWidth);
drawArrow([recurrent_input_x - 0.5, 2], [recurrent_input_x - 0.5, 3.5], '', 'r--');
drawBox([recurrent_block_offset + 3, 2], 3, 1.2, 'Delay (z^{-1})', [1 0.9 0.9], 'rectangle');

% Loss calculation
drawArrow([loss_arrow_start_x, 6], [loss_arrow_end_x, 6], 'Loss');
drawBox([loss_box_center_x, 6], loss_box_width, loss_box_height, '||G_{\theta}(a_j) - u_j||^2', colors.loss, 'rectangle', 'Interpreter', 'tex');


%% --- Legend for all components (Corrected) ---
legend_ax = axes('Position', [0.75, 0.42, 0.22, 0.32]);
axis off; hold on;
rectangle('Position', [0, 0, 1, 1], 'EdgeColor', 'k', 'LineWidth', 1);
text(0.5, 0.94, 'Legend', 'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(0.1, 0.80, 'W: Local Path (Conv3d)', 'FontSize', fontSize);
text(0.1, 0.72, 'K: Non-Local Path (SpectralConv3d)', 'FontSize', fontSize);
text(0.1, 0.64, 'Q: Spatial Net', 'FontSize', fontSize);
text(0.1, 0.56, 'H: Temporal Net', 'FontSize', fontSize);
plot([0.05 0.95], [0.5, 0.5], 'k:');
text(0.1, 0.42, 'R1: Conv Layer', 'FontSize', fontSize);
text(0.1, 0.34, 'R2: Fourier Transform', 'FontSize', fontSize);
text(0.1, 0.26, 'R3: Modes Filter', 'FontSize', fontSize);
text(0.1, 0.18, 'R4: Inverse Fourier', 'FontSize', fontSize);
text(0.1, 0.10, 'R5: Activation', 'FontSize', fontSize);


%% --- Helper Functions ---

function drawResistor(center, width, height, label)
    fontSize = 18;
    x=center(1); y=center(2);
    xp = x + [-width/2, -width/2+width/8, -width/8, width/8, -width/8, width/8, width/2-width/8, width/2];
    yp = y + [0, 0, height/2, -height/2, height/2, -height/2, 0, 0];
    plot(xp, yp, 'k', 'LineWidth', 1.5);
    text(x, y + height, label, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
end

function drawBox(center, width, height, label, color, shape, varargin)
    fontSize = 18;
    x = center(1)-width/2; y = center(2)-height/2;
    if strcmp(shape, 'rectangle')
        rectangle('Position', [x, y, width, height], 'FaceColor', color, 'EdgeColor', 'k', 'LineWidth', 1.5);
    else
        rectangle('Position', [x, y, width, height], 'FaceColor', color, 'EdgeColor', 'k', 'LineWidth', 1.5, 'Curvature', [1 1]);
    end
    text(center(1), center(2), label, 'FontSize', fontSize, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', varargin{:});
end

function drawArrow(start_point, end_point, label, line_style)
    fontSize = 18;
    if nargin < 3, label = ''; end
    if nargin < 4, line_style = 'k-'; end

    dp = end_point - start_point;
    quiver(start_point(1), start_point(2), dp(1), dp(2), 0, line_style, 'LineWidth', 1.5, 'MaxHeadSize', 0.5/norm(dp));
    if ~isempty(label)
        text(start_point(1)+dp(1)/2, start_point(2)+dp(2)/2+0.6, label, 'FontSize',fontSize, 'HorizontalAlignment','center','BackgroundColor','w');
    end
end

function drawSummer(center, radius)
    x=center(1); y=center(2);
    plot([x-radius, x+radius], [y, y], 'k', 'LineWidth', 1.5);
    plot([x, x], [y-radius, y+radius], 'k', 'LineWidth', 1.5);
    rectangle('Position', [x-radius, y-radius, 2*radius, 2*radius], 'Curvature', [1 1], 'EdgeColor', 'k', 'LineWidth', 1.5);
end