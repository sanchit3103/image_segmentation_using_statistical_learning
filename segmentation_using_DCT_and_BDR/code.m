clc; clear; close all;
load("TrainingSamplesDCT_8.mat")

% Part a: Calculation of Prior Probabilities
length_TrainSampleFG    = length(TrainsampleDCT_FG);
length_TrainSampleBG    = length(TrainsampleDCT_BG);

P_cheetah               = length_TrainSampleFG / (length_TrainSampleFG + length_TrainSampleBG);
P_grass                 = length_TrainSampleBG / (length_TrainSampleFG + length_TrainSampleBG);

% Part b: Computation and plot of index histograms

index_List_BG           = zeros(length_TrainSampleBG,1);
index_List_FG           = zeros(length_TrainSampleFG,1);
class_Conditionals_BG   = zeros(64,1);
class_Conditionals_FG   = zeros(64,1);

for i = 1:length_TrainSampleBG
    row                     = abs(TrainsampleDCT_BG(i,:));
    temp_array              = sort( row , 'descend' );
    sec_largest_eng_val     = temp_array(2); 
    ind_sec_largest_eng_val = find(row == sec_largest_eng_val);
    index_List_BG(i)        = ind_sec_largest_eng_val;
end

for i = 1:length_TrainSampleFG
    row                     = abs(TrainsampleDCT_FG(i,:));
    temp_array              = sort( row , 'descend' );
    sec_largest_eng_val     = temp_array(2); 
    ind_sec_largest_eng_val = find(row == sec_largest_eng_val);
    index_List_FG(i)        = ind_sec_largest_eng_val;
end

figure;
hist_BG = histogram(index_List_BG, 1:65);
title('Index Histogram for Background')
xlabel('Co-efficient of 2nd largest energy value for each block')
ylabel('Count of each co-efficient')
saveas(gcf,'Index Histogram_BG.png')

figure;
hist_FG = histogram(index_List_FG, 1:65);
title('Index Histogram for Foreground')
xlabel('Co-efficient of 2nd largest energy value for each block')
ylabel('Count of each co-efficient')
saveas(gcf,'Index Histogram_FG.png')

% Computation of class conditional probabilities

sum_BG  = sum(hist_BG.Values);
sum_FG  = sum(hist_FG.Values);

for i = 1:64
    class_Conditionals_BG(i) = hist_BG.Values(i) / sum_BG;
    class_Conditionals_FG(i) = hist_FG.Values(i) / sum_FG;
end

figure;
hist_BG = bar(class_Conditionals_BG);
title('Conditional Probability Chart for Background')
xlabel('Co-efficient of 2nd largest energy value for each block')
ylabel('Conditional probability of each co-efficient')
saveas(gcf,'Cond_Prob_BG.png')

figure;
hist_FG = bar(class_Conditionals_FG);
title('Conditional Probability Chart for Foreground')
xlabel('Co-efficient of 2nd largest energy value for each block')
ylabel('Conditional probability of each co-efficient')
saveas(gcf,'Cond_Prob_FG.png')

% Creation of mask for the given image

inputImg    = imread("cheetah.bmp");
inputImg    = im2double(inputImg);
img_Size    = size(inputImg);
img_Width   = img_Size(1);
img_Height  = img_Size(2);
winSize     = 8;
feature_X   = zeros(img_Width - winSize + 1, img_Height - winSize + 1);
state_Y_BG  = zeros(img_Width - winSize + 1, img_Height - winSize + 1);
state_Y_FG  = zeros(img_Width - winSize + 1, img_Height - winSize + 1);
A           = zeros(img_Width - winSize + 1, img_Height - winSize + 1);

fileID      = fopen('Zig-Zag Pattern.txt','r');
global zigzag
zigzag      = fscanf(fileID, '%d');

for j = 1:img_Height - winSize + 1
    for i = 1:img_Width - winSize + 1
        block                   = inputImg(i:i+winSize-1, j:j+winSize-1);
        block_DCT               = dct2(block);
        dct_Vector              = matrix_to_zigzag_vector(block_DCT);
        temp_array              = sort( dct_Vector , 'descend' );
        sec_largest_eng_val     = temp_array(2); 
        ind_sec_largest_eng_val = find(dct_Vector == sec_largest_eng_val);
        feature_X(i,j)          = ind_sec_largest_eng_val;
    end
end

figure;
hist_X = histogram(feature_X, 1:65);
sum_X  = sum(hist_X.Values);

for j = 1:img_Height - winSize + 1
    for i = 1:img_Width - winSize + 1
        P_x             = hist_X.Values(feature_X(i,j)) / sum_X;
        state_Y_FG(i,j) =  ( class_Conditionals_FG(feature_X(i,j)) * P_cheetah ) / P_x;
        state_Y_BG(i,j) =  ( class_Conditionals_BG(feature_X(i,j)) * P_grass ) / P_x;

        if state_Y_FG(i,j) > state_Y_BG(i,j)
            A(i,j)  = 1;
        else
            A(i,j)  = 0;
        end
    end
end

imagesc(A)
colormap(gray(255))
saveas(gcf,'Mask.png')

% Calculation of error
ground_Truth_Mask   = imread("cheetah_mask.bmp");
ground_Truth_Mask   = im2double(ground_Truth_Mask);

error               = sum( abs(A - ground_Truth_Mask(img_Width - winSize + 1, img_Height - winSize + 1)), "all" );
error               = error / (img_Width * img_Height);

function dct_vector = matrix_to_zigzag_vector(img_dct_block)
    dct_vector  = zeros(64,1);
    global zigzag
    for i = 1:8
        for j = 1:8
            index = zigzag( (i-1)*8 + j ) + 1;
            dct_vector(index) = img_dct_block(i,j);
        end 
    end
end