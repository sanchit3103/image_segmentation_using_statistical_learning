clc; clear; close all;

load("Alpha.mat")
load("Prior_1.mat")
load("Prior_2.mat")
load("TrainingSamplesDCT_subsets_8.mat")

% Strategy Declaration
mu0_BG(1,1)             = 2;
mu0_FG(1,1)             = 2;

% Initialize Variables
error_Pred_Dist         = zeros(max(size(alpha)), 1);
error_MAP               = zeros(max(size(alpha)), 1);
error_MLE               = zeros(max(size(alpha)), 1);
num_Features            = 64;

% Training Sample Set Declaration
Trainsample_FG          = D1_FG;
Trainsample_BG          = D1_BG;
length_TrainSampleFG    = length(Trainsample_FG);
length_TrainSampleBG    = length(Trainsample_BG);

% Calculation of Prior Probabilities
P_cheetah               = length_TrainSampleFG / (length_TrainSampleFG + length_TrainSampleBG);
P_grass                 = length_TrainSampleBG / (length_TrainSampleFG + length_TrainSampleBG);

% Calculation of mean for both the datasets
mean_FG                 = mean(Trainsample_FG);
mean_BG                 = mean(Trainsample_BG);

% Calculation of covariance for both the datasets
cov_FG                  = cov(Trainsample_FG);
cov_BG                  = cov(Trainsample_BG);
det_cov_FG              = det(cov_FG);
det_cov_BG              = det(cov_BG);

% Load Input Image and Declare Variables
inputImg    = imread("cheetah.bmp");
inputImg    = im2double(inputImg);
img_Size    = size(inputImg);
img_Width   = img_Size(1);
img_Height  = img_Size(2);
winSize     = 8;
A           = zeros(img_Width - winSize + 1, img_Height - winSize + 1);

fileID      = fopen('Zig-Zag Pattern.txt','r');
global zigzag
zigzag      = fscanf(fileID, '%d');

% Part a: Solution using parameters of predictive distribution

for k = 1:max(size(alpha))
    % Calculation of mu_n and sigma_n
    cov0_FG                 = diag(W0) * alpha(k);
    cov0_BG                 = diag(W0) * alpha(k);

    mu_n_FG                 = ( (length_TrainSampleFG * cov0_FG) / (cov_FG + (length_TrainSampleFG * cov0_FG)) ) * mean_FG' + ( cov_FG / (cov_FG + (length_TrainSampleFG * cov0_FG)) ) * mu0_FG';
    mu_n_BG                 = ( (length_TrainSampleBG * cov0_BG) / (cov_BG + (length_TrainSampleBG * cov0_BG)) ) * mean_BG' + ( cov_BG / (cov_BG + (length_TrainSampleBG * cov0_BG)) ) * mu0_BG';

    cov_n_FG                = (cov_FG * cov0_FG) / (cov_FG + (length_TrainSampleFG * cov0_FG));
    cov_n_BG                = (cov_BG * cov0_BG) / (cov_BG + (length_TrainSampleBG * cov0_BG));

    det_cov_n_FG            = det(cov_n_FG);
    det_cov_n_BG            = det(cov_n_BG);

    % Calculation of sigma_combined
    cov_combined_FG         = cov_FG + cov_n_FG;
    cov_combined_BG         = cov_BG + cov_n_BG;

    det_cov_combined_FG     = det(cov_combined_FG);
    det_cov_combined_BG     = det(cov_combined_BG);

    for j = 1:img_Height - winSize + 1
        for i = 1:img_Width - winSize + 1
            block                   = inputImg(i:i+winSize-1, j:j+winSize-1);
            block_DCT               = dct2(block);
            dct_Vector              = matrix_to_zigzag_vector(block_DCT);
            P_x_FG                  = exp( -0.5*( (dct_Vector - mu_n_FG') * inv(cov_combined_FG) * (dct_Vector - mu_n_FG')' ) ) / (sqrt( ((2*pi)^num_Features)*det_cov_combined_FG ) );
            P_x_FG                  = log(P_x_FG) + log(P_cheetah);
            P_x_BG                  = exp( -0.5*( (dct_Vector - mu_n_BG') * inv(cov_combined_BG) * (dct_Vector - mu_n_BG')' ) ) / (sqrt( ((2*pi)^num_Features)*det_cov_combined_BG ) );
            P_x_BG                  = log(P_x_BG) + log(P_grass);
        
            if P_x_FG > P_x_BG
                A(i,j)  = 1;
            else
                A(i,j)  = 0;
            end
        end
    end
    imagesc(A)
    colormap(gray(255))
    filename = sprintf('%d.png', k);
    saveas(gcf,filename)

    % Calculation of probability of error
    ground_Truth_Mask   = imread("cheetah_mask.bmp");
    ground_Truth_Mask   = im2double(ground_Truth_Mask);
    
    error               = sum( abs(A - ground_Truth_Mask(1:img_Width - winSize + 1, 1:img_Height - winSize + 1)), "all" );
    error_Pred_Dist(k)  = error / (img_Width * img_Height);
end

% Part b: Solution using ML Procedure

for k = 1:max(size(alpha))
    for j = 1:img_Height - winSize + 1
        for i = 1:img_Width - winSize + 1
            block                   = inputImg(i:i+winSize-1, j:j+winSize-1);
            block_DCT               = dct2(block);
            dct_Vector              = matrix_to_zigzag_vector(block_DCT);
            P_x_FG                  = exp( -0.5*( (dct_Vector - mean_FG) * inv(cov_FG) * (dct_Vector - mean_FG)' ) ) / (sqrt( ((2*pi)^num_Features)*det_cov_FG ) );
            P_x_FG                  = log(P_x_FG) + log(P_cheetah);
            P_x_BG                  = exp( -0.5*( (dct_Vector - mean_BG) * inv(cov_BG) * (dct_Vector - mean_BG)' ) ) / (sqrt( ((2*pi)^num_Features)*det_cov_BG ) );
            P_x_BG                  = log(P_x_BG) + log(P_grass);
        
            if P_x_FG > P_x_BG
                A(i,j)  = 1;
            else
                A(i,j)  = 0;
            end
        end
    end

    % Calculation of probability of error   
    error               = sum( abs(A - ground_Truth_Mask(1:img_Width - winSize + 1, 1:img_Height - winSize + 1)), "all" );
    error_MLE(k)        = error / (img_Width * img_Height);
end

% Part c: Solution using MAP estimate

for k = 1:max(size(alpha))
    % Calculation of mu_n
    cov0_FG                 = diag(W0) * alpha(k);
    cov0_BG                 = diag(W0) * alpha(k);

    mu_n_FG                 = ( (length_TrainSampleFG * cov0_FG) / (cov_FG + (length_TrainSampleFG * cov0_FG)) ) * mean_FG' + ( cov_FG / (cov_FG + (length_TrainSampleFG * cov0_FG)) ) * mu0_FG';
    mu_n_BG                 = ( (length_TrainSampleBG * cov0_BG) / (cov_BG + (length_TrainSampleBG * cov0_BG)) ) * mean_BG' + ( cov_BG / (cov_BG + (length_TrainSampleBG * cov0_BG)) ) * mu0_BG';

    for j = 1:img_Height - winSize + 1
        for i = 1:img_Width - winSize + 1
            block                   = inputImg(i:i+winSize-1, j:j+winSize-1);
            block_DCT               = dct2(block);
            dct_Vector              = matrix_to_zigzag_vector(block_DCT);
            P_x_FG                  = exp( -0.5*( (dct_Vector - mu_n_FG') * inv(cov_FG) * (dct_Vector - mu_n_FG')' ) ) / (sqrt( ((2*pi)^num_Features)*det_cov_FG ) );
            P_x_FG                  = log(P_x_FG) + log(P_cheetah);
            P_x_BG                  = exp( -0.5*( (dct_Vector - mu_n_BG') * inv(cov_BG) * (dct_Vector - mu_n_BG')' ) ) / (sqrt( ((2*pi)^num_Features)*det_cov_BG ) );
            P_x_BG                  = log(P_x_BG) + log(P_grass);
        
            if P_x_FG > P_x_BG
                A(i,j)  = 1;
            else
                A(i,j)  = 0;
            end
        end
    end

    % Calculation of probability of error
    error               = sum( abs(A - ground_Truth_Mask(1:img_Width - winSize + 1, 1:img_Height - winSize + 1)), "all" );
    error_MAP(k)        = error / (img_Width * img_Height);
end

figure;
plot(alpha, error_Pred_Dist)
hold on
plot(alpha, error_MLE)
hold on 
plot(alpha, error_MAP)
hold off
set(gca, 'XScale', 'log')
xlabel('Alpha - Values')
ylabel('Probability of Error')
title('Error vs Alpha Plot for Set D1 and Strategy 2')
legend('Predictive Distribution', 'MLE', 'MAP')
saveas(gca,'Plot_D1_S2.jpg')

function dct_vector = matrix_to_zigzag_vector(img_dct_block)
    dct_vector  = zeros(1,64);
    global zigzag
    for i = 1:8
        for j = 1:8
            index = zigzag( (i-1)*8 + j ) + 1;
            dct_vector(index) = img_dct_block(i,j);
        end 
    end
end